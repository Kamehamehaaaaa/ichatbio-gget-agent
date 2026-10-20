import json

import gget
import httpx
import pandas as pd

from context import current_context
from tools.get_sequence import FetchError, fetch_fasta
from tools.gget_runner import GgetTimeout, run_gget
from tools.util import context_tool

MIN_LENGTH = 20
MAX_LENGTH = 8000  # UCSC BLAT limit; gget silently truncates longer input
TOP_N = 10
MAX_BLAT_SECONDS = 120

NUCLEOTIDE_CHARS = set("ACGTUNRYKMSWBDHV")
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ*")

# Common species -> current UCSC assembly. Any other UCSC code (e.g. "hg19") is passed through.
UCSC_ASSEMBLIES = {
    "human": "hg38", "homo_sapiens": "hg38",
    "mouse": "mm39", "mus_musculus": "mm39",
    "rat": "rn7", "rattus_norvegicus": "rn7",
    "zebrafish": "danRer11", "danio_rerio": "danRer11",
    "chicken": "galGal6", "gallus_gallus": "galGal6",
    "zebra_finch": "taeGut2", "zebrafinch": "taeGut2", "taeniopygia_guttata": "taeGut2",
    "fruit_fly": "dm6", "drosophila_melanogaster": "dm6",
    "worm": "ce11", "c_elegans": "ce11", "caenorhabditis_elegans": "ce11",
    "yeast": "sacCer3", "saccharomyces_cerevisiae": "sacCer3",
    "dog": "canFam6", "canis_familiaris": "canFam6", "canis_lupus_familiaris": "canFam6",
    "pig": "susScr11", "sus_scrofa": "susScr11",
    "cow": "bosTau9", "bos_taurus": "bosTau9",
    "rhesus_macaque": "rheMac10", "macaca_mulatta": "rheMac10",
    "chimpanzee": "panTro6", "pan_troglodytes": "panTro6",
    "frog": "xenTro10", "xenopus_tropicalis": "xenTro10",
}


def _resolve_assembly(assembly: str) -> str:
    key = assembly.strip().lower().replace(" ", "_").replace(".", "")
    return UCSC_ASSEMBLIES.get(key, assembly.strip())


def _clean(raw: str) -> str:
    lines = raw.strip().splitlines()
    body = "".join(line for line in lines if not line.startswith(">"))
    return "".join(body.split()).upper()


def _browser_link(db: str, chrom: str, start, end) -> str:
    return f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={db}&position={chrom}:{start}-{end}"


@context_tool
async def map_to_genome(
    sequence: str | None = None,
    accession: str | None = None,
    assembly: str = "human",
) -> str:
    """
    Finds where a nucleotide or protein sequence is located in a reference genome using UCSC BLAT
    (via gget): chromosome, strand, start/end coordinates, % of the query aligned and % matched,
    plus a UCSC Genome Browser link for each hit. Handles spliced mRNA/cDNA (exons map separately).

    Use this when the user asks where a sequence, primer, read or transcript lies in a genome, which
    chromosome it is on, or its genomic coordinates. Not for species identification (use
    run_blast_sync) or comparing sequences (use align_sequences).

    Args:
        sequence: Raw sequence pasted by the user (FASTA header allowed), 20-8,000 residues.
        accession: NCBI nucleotide accession to fetch instead of a pasted sequence (first 8,000 bases used).
        assembly: Genome to search. A species name ("human", "mouse", "rat", "zebrafish", "chicken",
            "fruit_fly", "worm", "yeast", "dog", "pig", "cow", "rhesus_macaque", "chimpanzee", "frog")
            or a UCSC assembly code (e.g. "hg19", "mm10"). Default "human" (hg38).

    Returns JSON with the top genomic hits, or an ERROR message.
    """
    context = current_context.get()
    db = _resolve_assembly(assembly)

    if not sequence and not accession:
        return "ERROR: map_to_genome needs either a sequence or an accession."

    async with context.begin_process(f"Locating sequence in the {db} genome with BLAT") as process:
        query_label = "user-provided sequence"
        if not sequence:
            accession = accession.strip()
            query_label = accession
            try:
                _, sequence = await fetch_fasta(accession)
            except (FetchError, httpx.HTTPError) as e:
                await process.log(f"Could not fetch {accession}: {e}")
                return f"ERROR: could not fetch sequence for '{accession}': {e}"
            await process.log(f"Fetched {accession} ({len(sequence):,} residues)")

        sequence = _clean(sequence)
        invalid = set(sequence) - ALLOWED_CHARS
        if invalid or len(sequence) < MIN_LENGTH:
            reason = (
                f"contains invalid characters {sorted(invalid)}" if invalid
                else f"is only {len(sequence)} residues (BLAT needs at least {MIN_LENGTH})"
            )
            await process.log(f"Rejected sequence: {reason}")
            return (
                f"ERROR: the sequence '{sequence[:30]}' {reason}. It may be a placeholder; "
                "ask the user for the complete sequence."
            )

        truncated = len(sequence) > MAX_LENGTH
        original_length = len(sequence)
        if truncated:
            sequence = sequence[:MAX_LENGTH]
            await process.log(f"Sequence is {original_length:,} residues; using the first {MAX_LENGTH:,} (BLAT limit)")

        seqtype = "DNA" if set(sequence) <= NUCLEOTIDE_CHARS else "protein"
        await process.log(f"Submitting {len(sequence):,}-residue {seqtype} query to UCSC BLAT ({db})")

        try:
            df = await run_gget(
                process,
                gget.blat,
                sequence,
                seqtype=seqtype,
                assembly=db,
                verbose=False,
                label="UCSC BLAT",
                timeout_seconds=MAX_BLAT_SECONDS,
            )
        except GgetTimeout:
            await process.log("BLAT timed out")
            return f"ERROR: UCSC BLAT did not respond within {MAX_BLAT_SECONDS}s; try again later."
        except Exception as e:
            await process.log(f"BLAT failed: {e}")
            return f"ERROR: BLAT failed: {e}"

        if df is None or len(df) == 0:
            await process.log("No BLAT hits (or the UCSC request failed)")
            return (
                f"No BLAT hits for this {seqtype} sequence in {db}. Either it is not present in that "
                f"genome, '{db}' is not a valid UCSC assembly, or the UCSC server could not be reached."
            )

        df = pd.DataFrame(df)
        genome_used = str(df["genome"].iloc[0]) if "genome" in df.columns else db
        if genome_used != db:
            await process.log(f"UCSC did not recognize '{db}' and searched {genome_used} instead")
            return (
                f"ERROR: '{assembly}' is not a recognized UCSC assembly (UCSC fell back to {genome_used}). "
                "Use a species name from the list or a UCSC code such as hg38, mm39, danRer11."
            )

        sort_cols = [c for c in ["matches", "%_matched"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        df["ucsc_browser"] = [
            _browser_link(genome_used, r["chromosome"], r["start"], r["end"]) for _, r in df.iterrows()
        ]
        await process.log(f"Found {len(df)} BLAT hit(s) in {genome_used}")

        await process.create_artifact(
            mimetype="text/csv",
            description=f"UCSC BLAT hits for {query_label} in {genome_used} ({len(df)} hits)",
            content=df.to_csv(index=False).encode("utf-8"),
            metadata={"query": query_label, "assembly": genome_used, "seqtype": seqtype, "truncated": truncated},
        )

    keep = [c for c in ["chromosome", "strand", "start", "end", "%_aligned", "%_matched", "matches",
                        "mismatches", "aligned_start", "aligned_end", "ucsc_browser"] if c in df.columns]
    result = {
        "query": query_label,
        "query_length_submitted": len(sequence),
        "assembly": genome_used,
        "seqtype": seqtype,
        "total_hits": len(df),
        "top_hits": df[keep].head(TOP_N).to_dict(orient="records"),
        "note": "All hits saved as a CSV artifact for the user.",
    }
    if truncated:
        result["warning"] = f"Query was {original_length:,} residues; only the first {MAX_LENGTH:,} were searched."
    return json.dumps(result, default=str)


map_to_genome.handle_validation_error = True