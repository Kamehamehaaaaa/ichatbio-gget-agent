import json
from typing import Literal

import gget
import pandas as pd

from context import current_context
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

MAX_BGEE_SECONDS = 120
TOP_SPECIES = 30
TOP_TISSUES = 15


def _species_name(row) -> str:
    genus, species = str(row.get("genus") or "").strip(), str(row.get("species") or "").strip()
    return f"{genus} {species}".strip() or "unknown"


# Common names -> scientific names, so species_filter=["zebrafish"] matches Bgee's "Danio rerio"
COMMON_NAMES = {
    "human": "homo sapiens", "mouse": "mus musculus", "rat": "rattus norvegicus",
    "zebrafish": "danio rerio", "chicken": "gallus gallus", "cow": "bos taurus", "cattle": "bos taurus",
    "pig": "sus scrofa", "dog": "canis lupus familiaris", "cat": "felis catus", "horse": "equus caballus",
    "sheep": "ovis aries", "goat": "capra hircus", "rabbit": "oryctolagus cuniculus",
    "chimpanzee": "pan troglodytes", "macaque": "macaca mulatta", "rhesus macaque": "macaca mulatta",
    "fruit fly": "drosophila melanogaster", "fly": "drosophila melanogaster",
    "worm": "caenorhabditis elegans", "c elegans": "caenorhabditis elegans",
    "frog": "xenopus tropicalis", "platypus": "ornithorhynchus anatinus", "opossum": "monodelphis domestica",
}


def _normalize(name: str) -> str:
    n = name.strip().lower().replace("_", " ").replace(".", "")
    return COMMON_NAMES.get(n, n)


def _matches(name: str, wanted: list[str]) -> bool:
    n = name.lower()
    return any(_normalize(w) in n for w in wanted if w.strip())


def _summarize_orthologs(df: pd.DataFrame, species_filter: list[str] | None) -> dict:
    df = df.copy()
    df["organism"] = [_species_name(r) for _, r in df.iterrows()]
    by_species: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        name = r.get("gene_name")
        label = f"{name} ({r['gene_id']})" if name and str(name) != "nan" else str(r["gene_id"])
        by_species.setdefault(r["organism"], []).append(label)
    result = {"total_orthologs": len(df), "species_count": len(by_species)}

    if species_filter:
        found = {sp: genes for sp, genes in by_species.items() if _matches(sp, species_filter)}
        missing = [w for w in species_filter if not any(_matches(sp, [w]) for sp in by_species)]
        result["requested_species"] = {"found": found, "no_ortholog_listed": missing}
    else:
        names = sorted(by_species)
        result["orthologs_by_species"] = {sp: by_species[sp] for sp in names[:TOP_SPECIES]}
        if len(names) > TOP_SPECIES:
            result["more_species"] = len(names) - TOP_SPECIES
    return result


def _summarize_expression(df: pd.DataFrame) -> dict:
    df = df.copy()
    # Normalize e.g. "EXPRESSED" / "NOT_EXPRESSED" / "not expressed"
    states = (
        df["expression_state"].astype(str).str.lower().str.replace("_", " ")
        if "expression_state" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    expressed = df[states == "expressed"].sort_values("score", ascending=False)
    keep = [c for c in ["anat_entity_name", "anat_entity_id", "score", "score_confidence"] if c in df.columns]
    return {
        "conditions_total": len(df),
        "expressed_in": int((states == "expressed").sum()),
        "absent_in": int((states == "not expressed").sum()),
        "top_tissues": expressed[keep].head(TOP_TISSUES).to_dict(orient="records"),
        "score_note": "Bgee expression score, 0-100; higher means more highly expressed relative to other genes and tissues.",
    }


@context_tool
async def bgee_lookup(
    ensembl_gene_id: str,
    data: Literal["orthologs", "expression"] = "orthologs",
    species_filter: list[str] | None = None,
) -> str:
    """
    Looks up a gene in Bgee (via gget) for either:
    - "orthologs": orthologous genes of this gene across the ~50 animal species Bgee covers, grouped by
      species (gene name + Ensembl ID). Use for "which species have an ortholog of X", "what is the
      zebrafish/mouse version of X", or conservation across species.
    - "expression": tissues/organs where the gene is expressed, ranked by Bgee expression score.
      Use for "where is X expressed" or "which tissues express X".

    Needs an Ensembl GENE ID (ENSG..., ENSMUSG..., ENSDARG...). If the user gives a gene name, call
    ensembl_search first and pass the matching gene ID.

    Args:
        ensembl_gene_id: One Ensembl gene ID, e.g. "ENSG00000130234" (human ACE2).
        data: "orthologs" (default) or "expression".
        species_filter: For orthologs only: species to check, e.g. ["zebrafish", "Mus musculus"] or
            ["danio rerio"]. Omit to list orthologs in all species.

    Returns JSON with the summary, or an ERROR message.
    """
    context = current_context.get()
    gene_id = ensembl_gene_id.strip()
    if not gene_id.upper().startswith("ENS") or gene_id.upper().startswith(("ENST", "ENSP")):
        return (
            f"ERROR: '{gene_id}' is not an Ensembl gene ID. Bgee needs a gene ID (ENSG..., ENSMUSG..., "
            "ENSDARG...); call ensembl_search to find it from a gene name."
        )
    gene_id = gene_id.split(".")[0]  # Bgee uses unversioned IDs

    label = "orthologs" if data == "orthologs" else "expression data"
    async with context.begin_process(f"Fetching {label} for {gene_id} from Bgee") as process:
        try:
            df = await run_gget(
                process,
                gget.bgee,
                gene_id,
                type=data,
                verbose=False,
                label="Bgee lookup",
                timeout_seconds=MAX_BGEE_SECONDS,
            )
        except GgetTimeout:
            await process.log("Bgee timed out")
            return f"ERROR: Bgee did not respond within {MAX_BGEE_SECONDS}s; try again later."
        except Exception as e:
            await process.log(f"Bgee lookup failed: {e}")
            return (
                f"ERROR: Bgee lookup failed for {gene_id}: {e}. The gene may not be in Bgee "
                "(it covers ~50 animal species) or the ID may be invalid."
            )

        if df is None or len(df) == 0:
            await process.log(f"No {label} found")
            return f"No {label} found in Bgee for {gene_id}."

        df = pd.DataFrame(df)
        await process.log(f"Retrieved {len(df)} {label} records")

        await process.create_artifact(
            mimetype="text/csv",
            description=f"Bgee {label} for {gene_id} ({len(df)} records)",
            content=df.to_csv(index=False).encode("utf-8"),
            metadata={"gene_id": gene_id, "data": data},
        )

    summary = _summarize_orthologs(df, species_filter) if data == "orthologs" else _summarize_expression(df)
    return json.dumps(
        {"gene_id": gene_id, "data": data, **summary, "note": f"Full {label} table saved as a CSV artifact."},
        default=str,
    )


bgee_lookup.handle_validation_error = True