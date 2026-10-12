import json
import os

import httpx

from context import current_context
from tools.util import context_tool

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PREVIEW_LEN = 60
FASTA_LINE_WIDTH = 70


class FetchError(Exception):
    pass


async def fetch_fasta(accession: str) -> tuple[str, str]:
    """
    Plain helper (not a tool): fetches one nuccore record as FASTA.
    Returns (header, sequence). Raises FetchError or httpx.HTTPError on failure.
    Import this from run_blast.py so BLAST can fetch sequences itself.
    """
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "tool": "ichatbio-gget-agent",
    }
    if api_key := os.getenv("NCBI_API_KEY"):
        params["api_key"] = api_key
    if email := os.getenv("NCBI_EMAIL"):
        params["email"] = email

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as http:
        response = await http.get(EFETCH_URL, params=params)

    if not response.is_success:
        raise FetchError(f"NCBI returned HTTP {response.status_code} for '{accession}'")

    text = response.text.strip()
    if not text.startswith(">"):
        # NCBI sometimes returns 200 with an error message instead of FASTA
        raise FetchError(f"No FASTA record found for '{accession}': {text[:200]}")

    lines = text.splitlines()
    header = lines[0][1:].strip()
    seq_lines = []
    for line in lines[1:]:
        if line.startswith(">"):  # stop at a second record, if any
            break
        seq_lines.append(line.strip())
    sequence = "".join(seq_lines)

    if not sequence:
        raise FetchError(f"Record '{accession}' contains no sequence")

    return header, sequence

async def fetch_summary(accession: str) -> dict:
    """Plain helper: record metadata (organism, taxid, ...) via NCBI esummary. Returns {} on failure."""
    params = {"db": "nuccore", "id": accession, "retmode": "json", "tool": "ichatbio-gget-agent"}
    if api_key := os.getenv("NCBI_API_KEY"):
        params["api_key"] = api_key
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as http:
            response = await http.get(ESUMMARY_URL, params=params)
        result = response.json().get("result", {})
        uid = (result.get("uids") or [None])[0]
        doc = result.get(uid, {}) if uid else {}
    except (httpx.HTTPError, ValueError):
        return {}
    return {
        k: v
        for k, v in {
            "organism": doc.get("organism"),
            "taxid": doc.get("taxid"),
            "strain": doc.get("strain") or None,
            "molecule_type": doc.get("biomol") or doc.get("moltype"),
            "completeness": doc.get("completeness") or None,
            "source_db": doc.get("sourcedb"),
        }.items()
        if v not in (None, "", 0)
    }

@context_tool
async def get_sequence(accession: str) -> str:
    """
    Retrieves an NCBI nuccore record by accession number (e.g. "NC_045512.2", "MN908947"): its
    description, source organism, taxonomy ID, molecule type and sequence. Saves the full FASTA
    for the user.

    Use this when the user wants a sequence/record, OR asks which organism/species an accession
    is from — the record states it directly, no BLAST needed.

    :param accession: An NCBI nucleotide accession number.
    :return: JSON summary (accession, description, length, short preview), or an ERROR message.
    """
    context = current_context.get()
    accession = accession.strip()

    async with context.begin_process(f"Retrieving {accession} from NCBI nuccore") as process:
        try:
            header, sequence = await fetch_fasta(accession)
        except (FetchError, httpx.HTTPError) as e:
            await process.log(f"Retrieval failed: {e}")
            return f"ERROR: could not retrieve '{accession}': {e}"

        await process.log(f"Retrieved {header} ({len(sequence):,} bp)")
        summary = await fetch_summary(accession)

        wrapped = "\n".join(
            sequence[i : i + FASTA_LINE_WIDTH] for i in range(0, len(sequence), FASTA_LINE_WIDTH)
        )
        await process.create_artifact(
            mimetype="text/x-fasta",
            description=f"FASTA sequence for {accession}: {header}",
            content=f">{header}\n{wrapped}\n".encode("utf-8"),
            metadata={"accession": accession, "length": len(sequence)},
        )

    preview = sequence[:PREVIEW_LEN] + ("..." if len(sequence) > PREVIEW_LEN else "")
    return json.dumps(
        {
            "accession": accession,
            "description": header,
            **summary,
            "length": len(sequence),
            "preview": preview,
            "note": "Full FASTA saved as an artifact for the user.",
        }
    )


# from typing import Optional
# import httpx

# from tools.util import context_tool


# @context_tool
# async def get_sequence(accession_number: str) -> Optional[str]:
#     """
#     gets the nucleotide sequence for a given accession number from NCBI's efetch API. 
#     Returns the sequence as a string, or None if the request fails.
#     """
#     url = (
#         "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
#         f"?db=nuccore&id={accession_number}&rettype=fasta&retmode=text"
#     )
#     async with httpx.AsyncClient(follow_redirects=True) as http:
#         response = await http.get(url)
#         if not response.is_success:
#             return None
#         lines = response.text.strip().splitlines()
#         return "".join(line for line in lines if not line.startswith(">"))
 