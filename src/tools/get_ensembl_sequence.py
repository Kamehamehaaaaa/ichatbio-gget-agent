import json

import gget

from context import current_context
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

MAX_IDS = 20
PREVIEW_LEN = 60
FASTA_LINE_WIDTH = 70


def parse_fasta_list(lines: list[str] | None) -> list[tuple[str, str]]:
    """gget.seq returns [">header1", "SEQ1", ">header2", "SEQ2", ...] -> [(header, seq), ...]"""
    records = []
    header = None
    for line in lines or []:
        line = str(line)
        if line.startswith(">"):
            header = line[1:].strip()
        elif header is not None:
            records.append((header, line.strip()))
            header = None
    return records


async def fetch_ensembl_sequences(
    process, ens_ids: list[str], translate: bool = False, isoforms: bool = False
) -> list[tuple[str, str]]:
    """
    Plain helper (not a tool): fetches sequences for Ensembl IDs via gget.seq.
    Returns [(header, sequence), ...]. Raises on failure. Reusable from run_blast.py.
    """
    lines = await run_gget(
        process,
        gget.seq,
        ens_ids,
        translate=translate,
        isoforms=isoforms,
        verbose=False,
        label="Ensembl sequence fetch",
    )
    return parse_fasta_list(lines)


@context_tool
async def get_ensembl_sequence(
    ens_ids: list[str],
    translate: bool = False,
    isoforms: bool = False,
) -> str:
    """
    Fetches sequences for Ensembl gene or transcript IDs (ENSG..., ENST...). By default returns
    the nucleotide sequence; with translate=true returns the amino acid (protein) sequence from
    UniProt. For a gene ID, the canonical transcript is used unless isoforms=true.

    Use this when the user wants the DNA/cDNA or protein sequence of an Ensembl gene/transcript.
    If the user gave a gene name instead of an ID, call ensembl_search first.

    Args:
        ens_ids: One or more Ensembl IDs, e.g. ["ENSG00000130234"].
        translate: True for protein (amino acid) sequences, False for nucleotide sequences.
        isoforms: True to return all transcript isoforms of a gene, not just the canonical one.

    Returns JSON with each sequence's header, length and a short preview, or an ERROR message.
    """
    context = current_context.get()
    ids = [i.strip() for i in ens_ids if i.strip()]
    if not ids:
        return "ERROR: get_ensembl_sequence needs at least one Ensembl ID."
    if len(ids) > MAX_IDS:
        return f"ERROR: get_ensembl_sequence accepts at most {MAX_IDS} IDs at a time; got {len(ids)}."

    kind = "protein" if translate else "nucleotide"
    async with context.begin_process(f"Fetching {kind} sequence(s) for {', '.join(ids)}") as process:
        try:
            records = await fetch_ensembl_sequences(process, ids, translate=translate, isoforms=isoforms)
        except GgetTimeout as e:
            await process.log(str(e))
            return f"ERROR: {e}. Ensembl/UniProt may be busy; try again later."
        except Exception as e:
            await process.log(f"Sequence fetch failed: {e}")
            return f"ERROR: sequence fetch failed: {e}"

        if not records:
            await process.log("No sequences returned")
            return (
                f"No {kind} sequences found for {ids}. Check the IDs are valid; "
                "non-coding genes have no protein sequence."
            )

        await process.log(f"Retrieved {len(records)} {kind} sequence(s)")

        fasta = "".join(
            f">{h}\n"
            + "\n".join(s[i : i + FASTA_LINE_WIDTH] for i in range(0, len(s), FASTA_LINE_WIDTH))
            + "\n"
            for h, s in records
        )
        await process.create_artifact(
            mimetype="text/x-fasta",
            description=f"Ensembl {kind} sequence(s) for {', '.join(ids)} ({len(records)} records)",
            content=fasta.encode("utf-8"),
            metadata={"ens_ids": ids, "translate": translate, "isoforms": isoforms},
        )

    return json.dumps(
        {
            "sequence_type": kind,
            "count": len(records),
            "sequences": [
                {
                    "header": h,
                    "length": len(s),
                    "preview": s[:PREVIEW_LEN] + ("..." if len(s) > PREVIEW_LEN else ""),
                }
                for h, s in records
            ],
            "note": "Full sequences saved as a FASTA artifact for the user.",
        }
    )


get_ensembl_sequence.handle_validation_error = True