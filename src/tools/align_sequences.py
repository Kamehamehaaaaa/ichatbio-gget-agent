import json
import os
import shutil
import tempfile

import gget
import gget.gget_muscle as _gget_muscle
import httpx

from context import current_context
from tools.get_sequence import FetchError, fetch_fasta
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

# gget's bundled macOS MUSCLE binary is Intel-only and fails on Apple Silicon ("Bad CPU type").
# Prefer a native MUSCLE 5 if one is available: MUSCLE_BIN env var, else `muscle` on PATH.
_native_muscle = os.getenv("MUSCLE_BIN") or shutil.which("muscle")
if _native_muscle:
    _gget_muscle.PRECOMPILED_MUSCLE_PATH = _native_muscle

MIN_SEQUENCES = 2
MAX_SEQUENCES = 50
MAX_TOTAL_RESIDUES = 500_000
MAX_SEQUENCE_LENGTH = 20_000  # MUSCLE 5 gets very slow beyond gene-length sequences
SUPER5_THRESHOLD = 30  # MUSCLE's faster mode for larger sets
MAX_ALIGN_SECONDS = 300
MAX_VARIABLE_SITES_SHOWN = 25
FULL_MATRIX_MAX_SEQUENCES = 10
NAME_MAX_CHARS = 30
FASTA_LINE_WIDTH = 70

NUCLEOTIDE_CHARS = set("ACGTUNRYKMSWBDHV")
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ*")


class InputError(Exception):
    pass


def _parse_fasta_text(text: str) -> list[tuple[str, str]]:
    """Multi-FASTA text -> [(name, sequence)]. A bare sequence with no header becomes 'seq1'."""
    records, name, chunks = [], None, []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None or chunks:
                records.append((name or f"seq{len(records) + 1}", "".join(chunks)))
            name, chunks = line[1:].strip() or None, []
        else:
            chunks.append("".join(line.split()).upper())
    if name is not None or chunks:
        records.append((name or f"seq{len(records) + 1}", "".join(chunks)))
    return records


def _unique_names(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Short, unique, FASTA-safe names (first word of the header)."""
    seen, out = {}, []
    for name, seq in records:
        base = (name.split()[0] if name.split() else "seq")[:NAME_MAX_CHARS]
        seen[base] = seen.get(base, 0) + 1
        out.append((base if seen[base] == 1 else f"{base}_{seen[base]}", seq))
    return out


def _validate(records: list[tuple[str, str]]) -> str:
    """Checks counts, characters and molecule type. Returns 'nucleotide' or 'protein'."""
    if len(records) < MIN_SEQUENCES:
        raise InputError(f"need at least {MIN_SEQUENCES} sequences to align; got {len(records)}.")
    if len(records) > MAX_SEQUENCES:
        raise InputError(f"at most {MAX_SEQUENCES} sequences per alignment; got {len(records)}.")
    total = sum(len(s) for _, s in records)
    if total > MAX_TOTAL_RESIDUES:
        raise InputError(
            f"total length {total:,} residues exceeds the {MAX_TOTAL_RESIDUES:,} limit; "
            "align fewer or shorter sequences (e.g. one gene instead of whole genomes)."
        )
    for name, seq in records:
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise InputError(
                f"sequence '{name}' is {len(seq):,} residues; the limit is {MAX_SEQUENCE_LENGTH:,} per sequence. "
                "Align a gene or region instead of a whole genome."
            )
        bad = set(seq) - ALLOWED_CHARS
        if not seq or bad:
            reason = "is empty" if not seq else f"contains invalid characters {sorted(bad)}"
            raise InputError(
                f"sequence '{name}' {reason}. It may be a placeholder; ask the user for the full sequence."
            )
    kinds = {"nucleotide" if set(seq) <= NUCLEOTIDE_CHARS else "protein" for _, seq in records}
    if len(kinds) > 1:
        raise InputError("the sequences mix nucleotide and protein; align one type at a time.")
    return kinds.pop()


def _read_afa(path: str) -> dict[str, str]:
    return dict(_parse_fasta_text(open(path).read()))


def _pairwise_identity(a: str, b: str) -> float | None:
    """% identity over columns where neither sequence has a gap."""
    compared = matches = 0
    for x, y in zip(a, b):
        if x != "-" and y != "-":
            compared += 1
            matches += x == y
    return round(100 * matches / compared, 1) if compared else None


def summarize_alignment(aligned: dict[str, str]) -> dict:
    names = list(aligned)
    rows = [aligned[n] for n in names]
    length = len(rows[0])

    conserved = variable = gapped = 0
    variable_sites = []
    for i in range(length):
        column = [r[i] for r in rows]
        residues = set(column) - {"-"}
        if "-" in column:
            gapped += 1
        if len(residues) == 1 and "-" not in column:
            conserved += 1
        elif len(residues) > 1:
            variable += 1
            if len(variable_sites) < MAX_VARIABLE_SITES_SHOWN:
                variable_sites.append({"position": i + 1, **dict(zip(names, column))})

    matrix = {
        a: {b: (100.0 if a == b else _pairwise_identity(aligned[a], aligned[b])) for b in names}
        for a in names
    }
    pairs = [
        (matrix[a][b], a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if matrix[a][b] is not None
    ]
    pairs.sort()

    summary = {
        "sequences": len(names),
        "alignment_length": length,
        "ungapped_lengths": {n: len(aligned[n].replace("-", "")) for n in names},
        "conserved_columns": conserved,
        "variable_columns": variable,
        "columns_with_gaps": gapped,
        "percent_conserved": round(100 * conserved / length, 1) if length else 0,
    }
    if pairs:
        summary["mean_pairwise_identity"] = round(sum(p[0] for p in pairs) / len(pairs), 1)
        summary["most_similar_pair"] = {"pair": [pairs[-1][1], pairs[-1][2]], "identity": pairs[-1][0]}
        summary["least_similar_pair"] = {"pair": [pairs[0][1], pairs[0][2]], "identity": pairs[0][0]}
    if len(names) <= FULL_MATRIX_MAX_SEQUENCES:
        summary["pairwise_identity_percent"] = matrix
    summary["first_variable_sites"] = variable_sites
    if variable > len(variable_sites):
        summary["more_variable_sites"] = variable - len(variable_sites)
    return summary, matrix


def _identity_csv(matrix: dict) -> str:
    names = list(matrix)
    lines = ["," + ",".join(names)]
    for a in names:
        lines.append(a + "," + ",".join("" if matrix[a][b] is None else str(matrix[a][b]) for b in names))
    return "\n".join(lines) + "\n"


@context_tool
async def align_sequences(
    fasta: str | None = None,
    accessions: list[str] | None = None,
) -> str:
    """
    Aligns 2-50 nucleotide OR protein sequences with MUSCLE (via gget) and reports how similar they
    are: alignment length, conserved and variable columns, pairwise % identity, most and least similar
    pair, and the first variable positions with each sequence's residue.

    Use this when the user wants to align or compare several sequences, find where they differ, or
    see which are most similar. Do NOT use it to identify an unknown sequence (use run_blast_sync).

    Provide sequences pasted by the user as multi-FASTA text in `fasta` (headers become the names),
    and/or NCBI nucleotide accessions in `accessions` (fetched automatically, named by accession).
    Keep to genes or short regions; whole genomes over the size limit are rejected.

    Args:
        fasta: Multi-FASTA text of the user's sequences, copied exactly.
        accessions: NCBI nucleotide accessions to fetch and include, e.g. ["MN908947", "NC_004718.3"].

    Returns JSON with alignment statistics, or an ERROR message.
    """
    context = current_context.get()

    async with context.begin_process("Aligning sequences with MUSCLE") as process:
        records: list[tuple[str, str]] = _parse_fasta_text(fasta) if fasta else []

        for acc in [a.strip() for a in (accessions or []) if a.strip()]:
            try:
                _, seq = await fetch_fasta(acc)
            except (FetchError, httpx.HTTPError) as e:
                await process.log(f"Could not fetch {acc}: {e}")
                return f"ERROR: could not fetch sequence for '{acc}': {e}"
            records.append((acc, seq.upper()))
            await process.log(f"Fetched {acc} ({len(seq):,} residues)")

        records = _unique_names(records)
        try:
            kind = _validate(records)
        except InputError as e:
            await process.log(f"Rejected input: {e}")
            return f"ERROR: {e}"

        super5 = len(records) > SUPER5_THRESHOLD
        await process.log(
            f"Aligning {len(records)} {kind} sequences "
            f"({sum(len(s) for _, s in records):,} residues total)"
            + (" with MUSCLE Super5" if super5 else "")
        )

        with tempfile.TemporaryDirectory(prefix="gget_muscle_") as tmp:
            in_path = os.path.join(tmp, "input.fa")
            out_path = os.path.join(tmp, "aligned.afa")
            with open(in_path, "w") as f:
                f.writelines(f">{name}\n{seq}\n" for name, seq in records)

            try:
                await run_gget(
                    process,
                    gget.muscle,
                    in_path,
                    super5=super5,
                    out=out_path,
                    verbose=False,
                    label="MUSCLE alignment",
                    timeout_seconds=MAX_ALIGN_SECONDS,
                )
            except GgetTimeout:
                await process.log(f"Alignment did not finish within {MAX_ALIGN_SECONDS // 60} minutes")
                return (
                    f"ERROR: alignment timed out after {MAX_ALIGN_SECONDS // 60} minutes; "
                    "suggest fewer or shorter sequences."
                )
            except Exception as e:
                await process.log(f"MUSCLE failed: {e}")
                return f"ERROR: MUSCLE alignment failed: {e}"

            if not os.path.isfile(out_path):
                await process.log("MUSCLE produced no output")
                return "ERROR: MUSCLE produced no alignment output."
            raw = _read_afa(out_path)

        # MUSCLE reorders sequences; restore the user's order
        aligned = {name: raw[name] for name, _ in records if name in raw}
        if len(aligned) != len(records):
            return "ERROR: the alignment output is missing some sequences."

        summary, matrix = summarize_alignment(aligned)
        await process.log(
            f"Alignment length {summary['alignment_length']:,}; "
            f"{summary['percent_conserved']}% of columns fully conserved"
        )

        afa = "".join(
            f">{n}\n"
            + "\n".join(s[i : i + FASTA_LINE_WIDTH] for i in range(0, len(s), FASTA_LINE_WIDTH))
            + "\n"
            for n, s in aligned.items()
        )
        await process.create_artifact(
            mimetype="text/x-fasta",
            description=f"MUSCLE alignment of {len(aligned)} {kind} sequences (aligned FASTA)",
            content=afa.encode("utf-8"),
            metadata={"sequences": list(aligned), "molecule": kind, "super5": super5},
        )
        await process.create_artifact(
            mimetype="text/csv",
            description=f"Pairwise % identity matrix for {len(aligned)} aligned sequences",
            content=_identity_csv(matrix).encode("utf-8"),
            metadata={"sequences": list(aligned)},
        )

    return json.dumps(
        {
            "molecule": kind,
            **summary,
            "note": "Aligned FASTA and full identity matrix saved as artifacts for the user.",
        },
        default=str,
    )


align_sequences.handle_validation_error = True