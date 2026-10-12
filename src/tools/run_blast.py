import asyncio
import json
import time
from typing import Literal

import gget
import httpx
from pydantic import BaseModel, Field

from context import current_context
from tools.get_sequence import FetchError, fetch_fasta
from tools.util import context_tool

from tools.gget_runner import GgetTimeout, run_gget

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
    settings: BlastSettings | None = None,
) -> str:
    """
    Runs NCBI BLAST via gget to identify the likely organism of a RAW sequence (no accession), or to
    find sequences SIMILAR to a sequence or accession (closest relatives, homologs). Provide EITHER a
    raw sequence OR an accession (the sequence is fetched automatically). Optionally pass BLAST
    settings the user asked for. Can take a few minutes.

    Do NOT use this just to find which organism an accession comes from; get_sequence returns that
    directly from the record.

    Args:
        sequence: Raw nucleotide or protein sequence (FASTA headers are fine).
        accession: NCBI nucleotide accession, used if no sequence is given.
        settings: Optional BLAST settings; omit entirely if the user specified none.

    Returns JSON with the top hits (species, % identity, query cover, e-value, accession),
    or an ERROR message.
    """
    context = current_context.get()

    if not sequence and not accession:
        return "ERROR: run_blast_sync needs either a sequence or an accession number."

    async with context.begin_process("Identifying species with BLAST") as process:
        if not sequence:
            accession = accession.strip()
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
        
        if not sequence:
            return "ERROR: the provided sequence is empty after removing headers and whitespace."

        # Only pass settings that were actually set; gget's defaults handle the rest
        kwargs = settings.model_dump(exclude_none=True) if settings else {}

        await process.log(
            f"Submitting {len(sequence):,}-residue query to NCBI BLAST "
            f"with {kwargs or 'default settings'}. This can take a few minutes."
        )

        # gget.blast is blocking; run it in a thread so the server stays responsive,
        # and send periodic progress updates so the request isn't seen as idle.
        # start = time.monotonic()
        # blast_task = asyncio.create_task(asyncio.to_thread(gget.blast, sequence, **kwargs))
        # elapsed = 0
        # while True:
        #     done, _ = await asyncio.wait({blast_task}, timeout=HEARTBEAT_SECONDS)
        #     if done:
        #         break
        #     elapsed += HEARTBEAT_SECONDS
        #     if elapsed >= MAX_BLAST_SECONDS:
        #         blast_task.cancel()  # stops waiting; the background thread finishes on its own
        #         await process.log(f"BLAST did not finish within {MAX_BLAST_SECONDS // 60} minutes")
        #         return (
        #             f"ERROR: BLAST timed out after {MAX_BLAST_SECONDS // 60} minutes. NCBI may be busy; "
        #             "suggest the user try again later or use a shorter sequence."
        #         )
        #     minutes, seconds = divmod(elapsed, 60)
        #     await process.log(
        #         f"BLAST still running on NCBI servers ({minutes}m {seconds:02d}s elapsed)..."
        #     )

        # try:
        #     results = blast_task.result()
        # except Exception as e:
        #     await process.log(f"BLAST failed: {e}")
        #     return f"ERROR: BLAST failed: {e}"

        # await process.log(f"BLAST finished in {time.monotonic() - start:.0f}s")

        try:
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
            description=(
                f"BLAST hits for {accession or 'user-provided sequence'} "
                f"({len(results)} hits)"
            ),
            content=results.to_json(orient="records").encode("utf-8"),
            metadata={
                "query": accession or "user-provided sequence",
                "query_length": len(sequence),
                "blast_settings": {k: str(v) for k, v in kwargs.items()},
            },
        )

    wanted = {c.lower() for c in SUMMARY_COLUMNS}
    columns = [c for c in results.columns if c.lower() in wanted]
    top_hits = results[columns].head(TOP_N).to_dict(orient="records")

    return json.dumps(
        {
            "query": accession or "user-provided sequence",
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



# import asyncio
# import json

# import gget
# import httpx

# from context import current_context
# from models.BlastInput import BlastInput
# from tools.get_sequence import FetchError, fetch_fasta
# from tools.util import context_tool

# TOP_N = 5
# # Columns from gget.blast's DataFrame worth showing the LLM (only those present are used)
# SUMMARY_COLUMNS = [
#     "Scientific Name",
#     "Description",
#     "Per. ident",
#     "Query Cover",
#     "E value",
#     "Accession",
# ]
# BLAST_PARAMS = ("program", "database", "limit", "expect")


# def _clean_sequence(raw: str) -> str:
#     """Strips FASTA headers, whitespace and line breaks from a pasted sequence."""
#     lines = raw.strip().splitlines()
#     body = "".join(line for line in lines if not line.startswith(">"))
#     return "".join(body.split()).upper()


# @context_tool("run_blast_sync", args_schema=BlastInput, description="...")
# async def run_blast_sync(sequence=None, accession=None, program=None, limit=50):
#     """
#     Identifies the likely source organism(s) of a nucleotide or protein sequence by running
#     NCBI BLAST via gget. Provide EITHER a raw sequence pasted by the user OR an NCBI accession
#     number (the sequence is fetched automatically). Can take a few minutes.

#     :param sequence: Raw nucleotide or protein sequence (FASTA headers are fine).
#     :param accession: NCBI nucleotide accession, used if no sequence is given.
#     :param parameters: Optional BLAST settings from extract_parameters (program, database, limit, expect).
#     :return: JSON with the top hits (species, % identity, e-value), or an ERROR message.
#     """
#     context = current_context.get()

#     if not sequence and not accession:
#         return "ERROR: run_blast_sync needs either a sequence or an accession number."

#     async with context.begin_process("Identifying species with BLAST") as process:
#         if not sequence:
#             accession = accession.strip()
#             try:
#                 _, sequence = await fetch_fasta(accession)
#             except (FetchError, httpx.HTTPError) as e:
#                 await process.log(f"Could not fetch {accession}: {e}")
#                 return f"ERROR: could not fetch sequence for '{accession}': {e}"
#             await process.log(f"Fetched {accession} from NCBI ({len(sequence):,} residues)")

#         sequence = _clean_sequence(sequence)
#         if not sequence:
#             return "ERROR: the provided sequence is empty after removing headers and whitespace."

#         # Only pass settings that were actually set; gget's defaults handle the rest
#         kwargs = {}
#         if parameters is not None:
#             for name in BLAST_PARAMS:
#                 value = getattr(parameters, name, None)
#                 if value is not None:
#                     kwargs[name] = value

#         await process.log(
#             f"Submitting {len(sequence):,}-residue query to NCBI BLAST "
#             f"with {kwargs or 'default settings'}. This can take a few minutes."
#         )

#         try:
#             # gget.blast is blocking; run it in a thread so the server stays responsive
#             results = await asyncio.to_thread(gget.blast, sequence, **kwargs)
#         except Exception as e:
#             await process.log(f"BLAST failed: {e}")
#             return f"ERROR: BLAST failed: {e}"

#         if results is None or len(results) == 0:
#             await process.log("BLAST returned no significant hits")
#             return "BLAST completed but found no significant hits for this sequence."

#         await process.log(f"BLAST returned {len(results)} hits")

#         await process.create_artifact(
#             mimetype="application/json",
#             description=(
#                 f"BLAST hits for {accession or 'user-provided sequence'} "
#                 f"({len(results)} hits)"
#             ),
#             content=results.to_json(orient="records").encode("utf-8"),
#             metadata={
#                 "query": accession or "user-provided sequence",
#                 "query_length": len(sequence),
#                 "blast_settings": {k: str(v) for k, v in kwargs.items()},
#             },
#         )

#     columns = [c for c in SUMMARY_COLUMNS if c in results.columns]
#     top_hits = results[columns].head(TOP_N).to_dict(orient="records")

#     return json.dumps(
#         {
#             "query": accession or "user-provided sequence",
#             "query_length": len(sequence),
#             "total_hits": len(results),
#             "top_hits": top_hits,
#             "note": "Full hit table saved as an artifact for the user.",
#         },
#         default=str,
#     )




# from models.SpeciesIdentification import SpeciesIdentification
# from tools.util import context_tool
# import gget

# @context_tool
# def run_blast_sync(sequence: str, parameters: SpeciesIdentification):
#     """
#     Runs BLAST on a given nucleotide or protein sequence using the gget.blast function.
#     """
#     kwargs: dict = {}
#     if parameters.program:
#         kwargs["program"] = parameters.program
#     if parameters.database:
#         kwargs["database"] = parameters.database

#     return gget.blast(
#         sequence,
#         limit=parameters.limit,
#         expect=parameters.expect,
#         **kwargs,
#     )