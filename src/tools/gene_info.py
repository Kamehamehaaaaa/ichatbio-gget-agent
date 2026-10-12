import json

import gget
import pandas as pd

from context import current_context
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

MAX_IDS = 20
TEXT_CHARS = 300
LIST_ITEMS = 10

SUMMARY_FIELDS = [
    "primary_gene_name",
    "species",
    "assembly_name",
    "object_type",
    "biotype",
    "canonical_transcript",
    "uniprot_id",
    "ncbi_gene_id",
    "pdb_id",
    "synonyms",
    "protein_names",
    "ensembl_description",
    "uniprot_description",
    "ncbi_description",
    "subcellular_localisation",
]


def _compact(value):
    """Trim long text and long lists so the LLM gets a readable summary."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value[:LIST_ITEMS] + ([f"... {len(value) - LIST_ITEMS} more"] if len(value) > LIST_ITEMS else [])
    text = str(value)
    return text if len(text) <= TEXT_CHARS else text[:TEXT_CHARS] + "..."


@context_tool
async def gene_info(ens_ids: list[str], include_pdb: bool = False) -> str:
    """
    Looks up detailed information for Ensembl gene or transcript IDs (ENSG..., ENST...): gene
    name, species, biotype, genomic location, canonical transcript, UniProt and NCBI Gene IDs,
    protein names, functional descriptions and transcripts.

    Use this when the user gives Ensembl IDs, or after ensembl_search to describe the matches.

    Args:
        ens_ids: One or more Ensembl IDs, e.g. ["ENSG00000130234", "ENST00000252519"].
        include_pdb: Also look up PDB structure IDs (slower). Set only if the user asks about structures.

    Returns JSON with a summary per ID, or an ERROR message.
    """
    context = current_context.get()
    ids = [i.strip() for i in ens_ids if i.strip()]
    if not ids:
        return "ERROR: gene_info needs at least one Ensembl ID."
    if len(ids) > MAX_IDS:
        return f"ERROR: gene_info accepts at most {MAX_IDS} IDs at a time; got {len(ids)}."

    async with context.begin_process(f"Looking up {', '.join(ids)} in Ensembl") as process:
        try:
            df = await run_gget(
                process,
                gget.info,
                ids,
                pdb=include_pdb,
                verbose=False,
                label="Ensembl info lookup",
            )
        except GgetTimeout as e:
            await process.log(str(e))
            return f"ERROR: {e}. Ensembl/UniProt may be busy; try again later."
        except Exception as e:
            await process.log(f"Lookup failed: {e}")
            return f"ERROR: gene info lookup failed: {e}"

        if df is None or len(df) == 0:
            await process.log("No information found")
            return f"No Ensembl information found for {ids}. Check that the IDs are valid Ensembl IDs."

        df = pd.DataFrame(df)  # one row per ID, indexed by Ensembl ID
        await process.log(f"Retrieved information for {len(df)} ID(s)")

        await process.create_artifact(
            mimetype="application/json",
            description=f"Ensembl gene/transcript info for {', '.join(map(str, df.index))}",
            content=df.to_json(orient="index", default_handler=str).encode("utf-8"),
            metadata={"ens_ids": ids, "include_pdb": include_pdb},
        )

    results = {}
    for ens_id, row in df.iterrows():
        entry = {f: _compact(row[f]) for f in SUMMARY_FIELDS if f in row.index}
        if {"seq_region_name", "start", "end"} <= set(row.index):
            strand = row.get("strand")
            entry["location"] = (
                f"chr{row['seq_region_name']}:{row['start']}-{row['end']}"
                + (f" (strand {strand})" if strand is not None else "")
            )
        if "all_transcripts" in row.index and isinstance(row["all_transcripts"], list):
            entry["transcript_count"] = len(row["all_transcripts"])
        results[str(ens_id)] = {k: v for k, v in entry.items() if v not in (None, "", [])}

    missing = [i for i in ids if not any(i.split(".")[0] == str(k).split(".")[0] for k in results)]
    return json.dumps(
        {
            "results": results,
            **({"not_found": missing} if missing else {}),
            "note": "Full details (all transcripts, exons) saved as an artifact for the user.",
        },
        default=str,
    )


gene_info.handle_validation_error = True