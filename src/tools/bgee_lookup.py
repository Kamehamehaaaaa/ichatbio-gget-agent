import json
import re
from typing import Literal

import gget
import pandas as pd

from context import current_context
from tools.gget_runner import GgetTimeout, normalize_species, run_gget
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


# Ensembl gene-ID prefix -> species, to know which species a gene ID belongs to
ID_PREFIX_SPECIES = {
    "ENSG": "homo sapiens", "ENSMUSG": "mus musculus", "ENSRNOG": "rattus norvegicus",
    "ENSDARG": "danio rerio", "ENSGALG": "gallus gallus", "ENSBTAG": "bos taurus", "ENSSSCG": "sus scrofa",
    "ENSCAFG": "canis lupus familiaris", "ENSMMUG": "macaca mulatta", "ENSPTRG": "pan troglodytes",
    "ENSXETG": "xenopus tropicalis", "ENSECAG": "equus caballus", "ENSOARG": "ovis aries",
    "ENSFCAG": "felis catus", "ENSOCUG": "oryctolagus cuniculus",
}


class LookupError_(Exception):
    pass


def _species_of_id(gene_id: str) -> str | None:
    m = re.match(r"(ENS[A-Z]*?G)\d", gene_id.upper())
    return ID_PREFIX_SPECIES.get(m.group(1)) if m else None


async def _resolve_gene(process, gene_name: str, species: str) -> str:
    """Gene name + species -> Ensembl gene ID, accepting only an exact gene-name match."""
    try:
        df = await run_gget(
            process, gget.search, [gene_name], species=normalize_species(species), id_type="gene",
            verbose=False, label="Ensembl gene search", timeout_seconds=MAX_BGEE_SECONDS,
        )
    except Exception as e:
        raise LookupError_(f"could not search Ensembl for '{gene_name}' in {species}: {e}")
    if df is None or len(df) == 0:
        raise LookupError_(f"no Ensembl gene named '{gene_name}' found in {species}")
    df = pd.DataFrame(df)
    exact = df[df["gene_name"].astype(str).str.lower() == gene_name.strip().lower()]
    if exact.empty:
        candidates = ", ".join(df["gene_name"].astype(str).head(5))
        raise LookupError_(
            f"no gene named exactly '{gene_name}' in {species} (closest names: {candidates}). "
            "Check the gene symbol and that `species` is the species the gene comes FROM."
        )
    return str(exact.iloc[0]["ensembl_id"])


@context_tool
async def bgee_lookup(
    gene_name: str | None = None,
    species: str = "human",
    ensembl_gene_id: str | None = None,
    data: Literal["orthologs", "expression"] = "orthologs",
    species_filter: list[str] | None = None,
) -> str:
    """
    Looks up a gene in Bgee (via gget) for either:
    - "orthologs": orthologous genes of this gene across the ~50 animal species Bgee covers, grouped by
      species. Use for "which species have an ortholog of X" or "does zebrafish have an ortholog of X".
    - "expression": tissues/organs where the gene is expressed, ranked by Bgee expression score.

    Identify the gene by NAME + the species it comes FROM (preferred; the tool finds the exact Ensembl ID),
    or by an Ensembl gene ID. For ortholog questions, the gene's species is the one named after "of":
      "Does zebrafish have an ortholog of human TP53?" -> gene_name="TP53", species="human",
                                                          species_filter=["zebrafish"]
    Never look the gene up in the species being asked about.

    Args:
        gene_name: Gene symbol, e.g. "TP53", "ACE2", "Ins2".
        species: Species the gene comes FROM, e.g. "human" (default), "mouse", "zebrafish".
        ensembl_gene_id: Alternatively, an Ensembl gene ID (ENSG..., ENSMUSG..., ENSDARG...).
        data: "orthologs" (default) or "expression".
        species_filter: For orthologs only: species to check, e.g. ["zebrafish", "chicken"].
            Omit to list orthologs in all species.

    Returns JSON with the summary, or an ERROR message.
    """
    context = current_context.get()
    if not gene_name and not ensembl_gene_id:
        return "ERROR: bgee_lookup needs gene_name (+ species) or ensembl_gene_id."

    label = "orthologs" if data == "orthologs" else "expression data"
    async with context.begin_process(f"Fetching {label} for {gene_name or ensembl_gene_id} from Bgee") as process:
        if ensembl_gene_id:
            gene_id = ensembl_gene_id.strip()
            if not gene_id.upper().startswith("ENS") or gene_id.upper().startswith(("ENST", "ENSP")):
                return (
                    f"ERROR: '{gene_id}' is not an Ensembl gene ID. Pass gene_name and species instead, "
                    "e.g. gene_name='TP53', species='human'."
                )
            gene_id = gene_id.split(".")[0]  # Bgee uses unversioned IDs
        else:
            try:
                gene_id = await _resolve_gene(process, gene_name, species)
            except LookupError_ as e:
                await process.log(str(e))
                return f"ERROR: {e}"
            await process.log(f"Resolved {gene_name} ({species}) to {gene_id}")

        own_species = _species_of_id(gene_id) or _normalize(species)
        if data == "orthologs" and species_filter and own_species and any(
            _normalize(w) in own_species for w in species_filter
        ):
            return (
                f"ERROR: species_filter {species_filter} includes the gene's own species ({own_species}), so "
                "there is nothing to compare. For 'does <species B> have an ortholog of <species A> GENE', look "
                "up the gene in species A (gene_name='GENE', species='<A>') and set species_filter=['<B>']."
            )

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
            description=f"Bgee {label} for {gene_name or gene_id} ({gene_id}, {len(df)} records)",
            content=df.to_csv(index=False).encode("utf-8"),
            metadata={"gene_id": gene_id, "gene_name": gene_name, "data": data},
        )

    summary = _summarize_orthologs(df, species_filter) if data == "orthologs" else _summarize_expression(df)
    query = {"gene_id": gene_id, "gene_species": own_species}
    if gene_name:
        query["gene_name"] = gene_name
    return json.dumps(
        {**query, "data": data, **summary, "note": f"Full {label} table saved as a CSV artifact."},
        default=str,
    )


bgee_lookup.handle_validation_error = True
