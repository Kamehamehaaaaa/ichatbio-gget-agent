import json
from typing import Literal

import gget
import pandas as pd

from context import current_context
from tools.gget_runner import GgetTimeout, normalize_species, run_gget
from tools.util import context_tool

TOP_N = 10
DESCRIPTION_CHARS = 150


@context_tool
async def ensembl_search(
    searchwords: list[str],
    species: str,
    id_type: Literal["gene", "transcript"] = "gene",
    andor: Literal["or", "and"] = "or",
    limit: int | None = None,
) -> str:
    """
    Searches Ensembl for genes (or transcripts) whose name or description matches free-text
    search words, and returns their Ensembl IDs. Use this to turn a gene name, symbol or
    keyword (e.g. "ace2", "angiotensin converting enzyme 2") into Ensembl IDs (ENSG.../ENST...)
    for gene_info or get_ensembl_sequence.

    Args:
        searchwords: One or more search words, e.g. ["ace2", "angiotensin converting enzyme 2"].
        species: Ensembl species name, e.g. "homo_sapiens", "mus_musculus", "danio_rerio".
            "human" and "mouse" also work.
        id_type: "gene" for gene IDs (ENSG...), "transcript" for transcript IDs (ENST...).
        andor: "or" returns matches for ANY search word; "and" requires ALL of them.
        limit: Maximum number of results. Leave empty unless the user asks for a number.

    Returns JSON with the top matches (Ensembl ID, gene name, description, biotype), or an ERROR message.
    """
    context = current_context.get()
    species = normalize_species(species)
    words = [w.strip() for w in searchwords if w.strip()]
    if not words:
        return "ERROR: ensembl_search needs at least one search word."

    async with context.begin_process(f"Searching Ensembl ({species}) for {', '.join(words)}") as process:
        try:
            df = await run_gget(
                process,
                gget.search,
                words,
                species=species,
                id_type=id_type,
                andor=andor,
                limit=limit,
                verbose=False,
                label="Ensembl search",
            )
        except GgetTimeout as e:
            await process.log(str(e))
            return f"ERROR: {e}. Ensembl may be busy; try again later."
        except Exception as e:
            await process.log(f"Ensembl search failed: {e}")
            return f"ERROR: Ensembl search failed: {e}. Check that '{species}' is a valid Ensembl species name."

        if df is None or len(df) == 0:
            await process.log("No matches found")
            return f"No Ensembl {id_type}s in {species} matched {words} (andor='{andor}')."

        df = pd.DataFrame(df)
        await process.log(f"Found {len(df)} matching {id_type}s")

        await process.create_artifact(
            mimetype="application/json",
            description=f"Ensembl {id_type} search results for {words} in {species} ({len(df)} matches)",
            content=df.to_json(orient="records").encode("utf-8"),
            metadata={"species": species, "searchwords": words, "id_type": id_type, "andor": andor},
        )

    keep = [c for c in ["ensembl_id", "gene_name", "ensembl_description", "biotype"] if c in df.columns]
    top = df[keep].head(TOP_N).copy()
    if "ensembl_description" in top.columns:
        top["ensembl_description"] = top["ensembl_description"].astype(str).str.slice(0, DESCRIPTION_CHARS)

    return json.dumps(
        {
            "species": species,
            "total_matches": len(df),
            "top_matches": top.to_dict(orient="records"),
            "note": "Full results saved as an artifact for the user.",
        },
        default=str,
    )


ensembl_search.handle_validation_error = True