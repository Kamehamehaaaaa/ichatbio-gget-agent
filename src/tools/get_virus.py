import asyncio
import glob
import json
import os
import tempfile
import time
from typing import Literal

import gget
import pandas as pd
from pydantic import BaseModel, Field

from context import current_context
from tools.util import context_tool

from tools.gget_runner import GgetTimeout, run_gget

# Give up waiting after this long
MAX_VIRUS_SECONDS = 900
# Don't upload a FASTA artifact bigger than this; the metadata table is still saved
MAX_FASTA_ARTIFACT_BYTES = 20 * 1024 * 1024
EXAMPLE_RECORDS = 5
TOP_VALUES = 5


class VirusFilters(BaseModel):
    """Optional NCBI Virus filters. Only fill in what the user explicitly asked for."""

    host: str | None = Field(
        None, description='Host organism, e.g. "Homo sapiens", "Gallus gallus", "bat". "human" also works.'
    )
    nuc_completeness: Literal["complete", "partial"] | None = Field(
        None, description='"complete" for complete genomes only, "partial" for partial only.'
    )
    source_database: Literal["refseq", "genbank"] | None = Field(
        None, description='"refseq" for curated reference genomes only (small result sets), "genbank" for all submissions.'
    )
    geographic_location: str | None = Field(
        None, description='Where the sample was collected, e.g. "USA", "Brazil", "Africa".'
    )
    min_collection_date: str | None = Field(None, description="Earliest collection date, YYYY-MM-DD.")
    max_collection_date: str | None = Field(None, description="Latest collection date, YYYY-MM-DD.")
    min_seq_length: int | None = Field(None, ge=1, description="Minimum sequence length (nt).")
    max_seq_length: int | None = Field(None, ge=1, description="Maximum sequence length (nt).")
    lineage: str | None = Field(
        None, description='Virus lineage, e.g. Pango lineage "B.1.1.7" for SARS-CoV-2.'
    )
    segment: str | None = Field(
        None, description='Genome segment for segmented viruses, e.g. "4" or "HA" for influenza.'
    )


def _find(folder: str, suffix: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(folder, f"*{suffix}")))
    return matches[0] if matches else None


def _col(df: pd.DataFrame, name: str) -> str | None:
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None


def summarize_metadata(df: pd.DataFrame) -> dict:
    """Compact summary of the gget.virus metadata table for the LLM."""
    summary: dict = {"total_records": len(df)}

    for field, key in [
        ("Organism Name", "top_organisms"),
        ("Host", "top_hosts"),
        ("Geographic Location", "top_locations"),
        ("Nuc Completeness", "completeness"),
        ("Virus Lineage", "top_lineages"),
    ]:
        col = _col(df, field)
        if col and df[col].notna().any():
            summary[key] = df[col].value_counts().head(TOP_VALUES).to_dict()

    length = _col(df, "Length")
    if length:
        lengths = pd.to_numeric(df[length], errors="coerce").dropna()
        if len(lengths):
            summary["length_range_nt"] = [int(lengths.min()), int(lengths.max())]

    date = _col(df, "Collection Date")
    if date:
        dates = df[date].dropna().astype(str)
        if len(dates):
            summary["collection_date_range"] = [dates.min(), dates.max()]

    example_cols = [
        c
        for name in ["Accession", "Organism Name", "Host", "Geographic Location", "Collection Date", "Length"]
        if (c := _col(df, name))
    ]
    summary["examples"] = df[example_cols].head(EXAMPLE_RECORDS).to_dict(orient="records")
    return summary


@context_tool
async def get_virus(
    virus: str,
    is_accession: bool = False,
    filters: VirusFilters | None = None,
) -> str:
    """
    Downloads virus genome sequences and metadata from the NCBI Virus database via gget.virus.
    Use this to find or count virus sequences matching criteria (host, location, collection date,
    completeness, lineage, etc.), or to get the metadata for a virus accession.

    Broad queries (e.g. all "Influenza A virus" or "SARS-CoV-2") can return hundreds of thousands
    of records and take a long time. Narrow them with filters when the user's request allows.

    Args:
        virus: Virus name (e.g. "Zika virus"), NCBI taxonomy ID (e.g. "64320"), or accession number.
        is_accession: True if `virus` is an accession number (e.g. "NC_045512.2").
        filters: Optional NCBI Virus filters; omit entirely if the user specified none.
    
    If no filters are given, only RefSeq reference genomes are searched; pass filters (or
    source_database="genbank") to search all submitted records.

    Returns JSON with record counts, top hosts/locations/lineages, date and length ranges and example
    records, or an ERROR message.
    """
    context = current_context.get()
    virus = virus.strip()
    kwargs = filters.model_dump(exclude_none=True) if filters else {}

    # Guard: a bare name query ("measles virus", "SARS-CoV-2") can download tens of thousands of records.
    # With no filters, search only RefSeq reference genomes and tell the agent how to widen the search.
    defaulted_to_refseq = False
    if not is_accession and not kwargs:
        kwargs = {"source_database": "refseq"}
        defaulted_to_refseq = True

    async with context.begin_process(f"Downloading {virus} data from NCBI Virus") as process:
        await process.log(f"Querying NCBI Virus for '{virus}' with {kwargs or 'no filters'}")

        with tempfile.TemporaryDirectory(prefix="gget_virus_") as outfolder:
            # gget.virus is blocking; run it in a thread and send progress updates while it runs
            # start = time.monotonic()
            # task = asyncio.create_task(
            #     asyncio.to_thread(
            #         gget.virus,
            #         virus,
            #         is_accession=is_accession,
            #         outfolder=outfolder,
            #         verbose=False,
            #         **kwargs,
            #     )
            # )
            # elapsed = 0
            # while True:
            #     done, _ = await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
            #     if done:
            #         break
            #     elapsed += HEARTBEAT_SECONDS
            #     if elapsed >= MAX_VIRUS_SECONDS:
            #         task.cancel()  # stops waiting; the thread finishes on its own
            #         await process.log(f"Download did not finish within {MAX_VIRUS_SECONDS // 60} minutes")
            #         return (
            #             f"ERROR: NCBI Virus download for '{virus}' timed out after "
            #             f"{MAX_VIRUS_SECONDS // 60} minutes. Suggest narrowing the query with filters "
            #             "(e.g. source_database='refseq', nuc_completeness='complete', host, date range)."
            #         )
            #     minutes, seconds = divmod(elapsed, 60)
            #     await process.log(f"Still downloading from NCBI Virus ({minutes}m {seconds:02d}s elapsed)...")

            # try:
            #     task.result()
            # except Exception as e:
            #     await process.log(f"gget.virus failed: {e}")
            #     return f"ERROR: gget.virus failed for '{virus}': {e}"

            # await process.log(f"Download finished in {time.monotonic() - start:.0f}s")

            try:
                await run_gget(
                    process,
                    gget.virus,
                    virus,
                    is_accession=is_accession,
                    outfolder=outfolder,
                    verbose=False,
                    label="NCBI Virus download",
                    timeout_seconds=MAX_VIRUS_SECONDS,
                    **kwargs,
                )
            except GgetTimeout:
                await process.log(f"Download did not finish within {MAX_VIRUS_SECONDS // 60} minutes")
                return (
                    f"ERROR: NCBI Virus download for '{virus}' timed out after "
                    f"{MAX_VIRUS_SECONDS // 60} minutes. Suggest narrowing the query with filters "
                    "(e.g. source_database='refseq', nuc_completeness='complete', host, date range)."
                )
            except Exception as e:
                await process.log(f"gget.virus failed: {e}")
                return f"ERROR: gget.virus failed for '{virus}': {e}"
            
            metadata_csv = _find(outfolder, "_metadata.csv")
            fasta_path = _find(outfolder, "_sequences.fasta")

            if not metadata_csv:
                await process.log("No records matched the query")
                return f"No NCBI Virus records found for '{virus}' with filters {kwargs or 'none'}."

            df = pd.read_csv(metadata_csv)
            if df.empty:
                await process.log("No records matched the query")
                return f"No NCBI Virus records found for '{virus}' with filters {kwargs or 'none'}."

            await process.log(f"Found {len(df):,} records")

            await process.create_artifact(
                mimetype="text/csv",
                description=f"NCBI Virus metadata for '{virus}' ({len(df):,} records)",
                content=open(metadata_csv, "rb").read(),
                metadata={"query": virus, "filters": {k: str(v) for k, v in kwargs.items()}},
            )

            fasta_note = "No FASTA file was produced."
            if fasta_path:
                size = os.path.getsize(fasta_path)
                if size <= MAX_FASTA_ARTIFACT_BYTES:
                    await process.create_artifact(
                        mimetype="text/x-fasta",
                        description=f"NCBI Virus sequences for '{virus}' ({len(df):,} records)",
                        content=open(fasta_path, "rb").read(),
                        metadata={"query": virus, "filters": {k: str(v) for k, v in kwargs.items()}},
                    )
                    fasta_note = "Sequences saved as a FASTA artifact."
                else:
                    fasta_note = (
                        f"FASTA was {size / 1024 / 1024:.0f} MB, too large to attach; "
                        "suggest narrowing the query with filters to get the sequences."
                    )
                    await process.log(fasta_note)

    summary = summarize_metadata(df)
    summary.update(
        {
            "query": virus,
            "filters_used": (
                {"filter_note": "No filters were given, so only RefSeq reference genomes were searched. "
                 "To search all records, add filters (host, location, dates, completeness) or source_database='genbank'."}
                if defaulted_to_refseq else {}
            ),
            "note": f"Full metadata table saved as a CSV artifact. {fasta_note}",
        }
    )
    return json.dumps(summary, default=str)


# If the LLM sends invalid filters, return the validation error to it so it can retry
get_virus.handle_validation_error = True