import json
from typing import Literal

import gget
import httpx
from pydantic import BaseModel, Field

from context import current_context
from tools.get_ensembl_sequence import fetch_ensembl_sequences
from tools.get_sequence import FetchError, fetch_fasta
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

MIN_SEQUENCE_LENGTH = 20
ALLOWED_SEQUENCE_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ*-")  # IUPAC nucleotide + amino acid codes
TOP_N = 5
# Give up waiting after this long (NCBI web BLAST is usually 30s–5min)
MAX_BLAST_SECONDS = 600
# Columns from gget.blast's DataFrame worth showing the LLM (only those present are used)
SUMMARY_COLUMNS = [
    "Scientific Name",
    "Description",
    "Per. Ident",
    "Query Cover",
    "E value",
    "Accession",
]


class BlastSettings(BaseModel):
    """Optional BLAST settings. Only fill in what the user explicitly asked for."""

    program: Literal["blastn", "blastp", "blastx", "tblastn", "tblastx"] | None = Field(
        None,
        description=(
            "BLAST program. Leave empty to auto-detect (blastn for nucleotide, blastp for protein). "
            "Set only if the user names a program."
        ),
    )
    database: str | None = Field(
        None,
        description=(
            "NCBI database to search. Nucleotide: 'nt', 'refseq_rna', 'pdbnt'. "
            "Protein: 'nr', 'refseq_protein', 'swissprot', 'pdbaa'. "
            "Leave empty for the default."
        ),
    )
    limit: int | None = Field(
        None,
        ge=1,
        le=500,
        description="Maximum number of hits to return (default 50). Set if the user asks for e.g. 'top 10'.",
    )
    expect: float | None = Field(
        None,
        gt=0,
        description="E-value threshold (default 10.0). Set only if the user specifies one.",
    )


def _clean_sequence(raw: str) -> str:
    """Strips FASTA headers, whitespace and line breaks from a pasted sequence."""
    lines = raw.strip().splitlines()
    body = "".join(line for line in lines if not line.startswith(">"))
    return "".join(body.split()).upper()


@context_tool
async def run_blast_sync(
    sequence: str | None = None,
    accession: str | None = None,
    ensembl_id: str | None = None,
    protein: bool = False,
    settings: BlastSettings | None = None,
) -> str:
    """
    Runs NCBI BLAST via gget to identify the likely organism of a RAW sequence (no accession), or to
    find sequences SIMILAR to a sequence, accession or Ensembl gene/transcript (closest relatives,
    homologs). Provide ONE of: a raw sequence, an NCBI accession, or an Ensembl ID (the sequence is
    fetched automatically). Optionally pass BLAST settings the user asked for. Can take a few minutes.

    Do NOT use this just to find which organism an accession comes from; get_sequence returns that
    directly from the record.

    Args:
        sequence: Raw nucleotide or protein sequence (FASTA headers are fine).
        accession: NCBI nucleotide accession, used if no sequence is given.
        ensembl_id: Ensembl gene or transcript ID (ENSG..., ENST..., ENSMUSG...), used if no sequence or
            accession is given; the sequence is fetched from Ensembl. E.g. "BLAST the human ACE2 protein"
            -> ensembl_search, then ensembl_id=<ACE2 gene ID>, protein=true.
        protein: With ensembl_id: true = BLAST the protein (blastp), false = the nucleotide sequence.
            For a gene ID, false means the whole gene region (introns included); prefer protein=true or
            a transcript ID (ENST...) for cDNA.
        settings: Optional BLAST settings; omit entirely if the user specified none.

    Returns JSON with the top hits (species, % identity, query cover, e-value, accession),
    or an ERROR message.
    """
    context = current_context.get()

    if not sequence and not accession and not ensembl_id:
        return "ERROR: run_blast_sync needs a sequence, an NCBI accession or an Ensembl ID."
    query_label = "user-provided sequence"

    async with context.begin_process("Identifying species with BLAST") as process:
        if not sequence and not accession and ensembl_id:
            ensembl_id = ensembl_id.strip()
            kind = "protein" if protein else "nucleotide"
            try:
                records = await fetch_ensembl_sequences(process, [ensembl_id], translate=protein)
            except Exception as e:
                await process.log(f"Could not fetch {ensembl_id} from Ensembl: {e}")
                return f"ERROR: could not fetch the {kind} sequence for '{ensembl_id}' from Ensembl: {e}"
            if not records:
                return (
                    f"ERROR: Ensembl returned no {kind} sequence for '{ensembl_id}'"
                    + (" (it may be non-coding)." if protein else ".")
                )
            _, sequence = records[0]  # canonical transcript / representative isoform
            query_label = f"{ensembl_id} ({kind})"
            await process.log(f"Fetched {query_label} from Ensembl ({len(sequence):,} residues)")
        elif not sequence:
            accession = accession.strip()
            query_label = accession
            try:
                _, sequence = await fetch_fasta(accession)
            except (FetchError, httpx.HTTPError) as e:
                await process.log(f"Could not fetch {accession}: {e}")
                return f"ERROR: could not fetch sequence for '{accession}': {e}"
            await process.log(f"Fetched {accession} from NCBI ({len(sequence):,} residues)")

        sequence = _clean_sequence(sequence)
        invalid = set(sequence) - ALLOWED_SEQUENCE_CHARS
        if invalid or len(sequence) < MIN_SEQUENCE_LENGTH:
            reason = (
                f"contains invalid characters {sorted(invalid)}" if invalid
                else f"is only {len(sequence)} residues (minimum {MIN_SEQUENCE_LENGTH})"
            )
            await process.log(f"Rejected sequence: {reason}")
            return (
                f"ERROR: the sequence '{sequence[:30]}' {reason}, so it does not look like a real "
                "sequence. It may be a placeholder; the full sequence was probably not included in the "
                "request. Abort and ask the user to provide the complete sequence."
            )

        # Only pass settings that were actually set; gget's defaults handle the rest
        kwargs = settings.model_dump(exclude_none=True) if settings else {}

        await process.log(
            f"Submitting {len(sequence):,}-residue query to NCBI BLAST "
            f"with {kwargs or 'default settings'}. This can take a few minutes."
        )

        try:
            # Runs gget.blast in a thread with "still running" updates and a time limit
            results = await run_gget(
                process,
                gget.blast,
                sequence,
                label="BLAST",
                timeout_seconds=MAX_BLAST_SECONDS,
                **kwargs,
            )
        except GgetTimeout:
            await process.log(f"BLAST did not finish within {MAX_BLAST_SECONDS // 60} minutes")
            return (
                f"ERROR: BLAST timed out after {MAX_BLAST_SECONDS // 60} minutes. NCBI may be busy; "
                "suggest the user try again later or use a shorter sequence."
            )
        except Exception as e:
            await process.log(f"BLAST failed: {e}")
            return f"ERROR: BLAST failed: {e}"

        if results is None or len(results) == 0:
            await process.log("BLAST returned no significant hits")
            return "BLAST completed but found no significant hits for this sequence."

        await process.log(f"BLAST returned {len(results)} hits")

        await process.create_artifact(
            mimetype="application/json",
            description=f"BLAST hits for {query_label} ({len(results)} hits)",
            content=results.to_json(orient="records").encode("utf-8"),
            metadata={
                "query": query_label,
                "query_length": len(sequence),
                "blast_settings": {k: str(v) for k, v in kwargs.items()},
            },
        )

    wanted = {c.lower() for c in SUMMARY_COLUMNS}
    columns = [c for c in results.columns if c.lower() in wanted]
    top_hits = results[columns].head(TOP_N).to_dict(orient="records")

    return json.dumps(
        {
            "query": query_label,
            "query_length": len(sequence),
            "total_hits": len(results),
            "settings_used": kwargs or "defaults",
            "top_hits": top_hits,
            "note": "Full hit table saved as an artifact for the user.",
        },
        default=str,
    )


# If the LLM sends invalid settings (e.g. limit=5000), return the validation error to it
# so it can retry, instead of crashing the run.
run_blast_sync.handle_validation_error = True