import json
from typing import Literal

import gget

from context import current_context
from tools.gget_runner import GgetTimeout, normalize_species, run_gget
from tools.util import context_tool

FileType = Literal["dna", "cdna", "cds", "ncrna", "pep", "gtf"]


def _human_size(size) -> str:
    try:
        n = float(size)
    except (TypeError, ValueError):
        return str(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@context_tool
async def reference_genome(
    species: str,
    which: list[FileType] | None = None,
    release: int | None = None,
) -> str:
    """
    Gets download links (FTP URLs) for Ensembl reference genome and annotation files for a species,
    with release number, release date and file size. Does NOT download the files.

    Args:
        species: Ensembl species name, e.g. "homo_sapiens", "mus_musculus". "human" and "mouse" also work.
        which: File types to return; omit for all. "dna" = genome FASTA, "cdna" = transcriptome,
            "cds" = coding sequences, "ncrna" = non-coding RNAs, "pep" = protein sequences,
            "gtf" = gene annotation (GTF).
        release: Ensembl release number, e.g. 110. Omit for the latest release.

    Returns JSON with one entry per file (URL, release, date, size), or an ERROR message.
    """
    context = current_context.get()
    species = normalize_species(species)
    which_arg = list(dict.fromkeys(which)) if which else "all"

    async with context.begin_process(f"Fetching Ensembl reference links for {species}") as process:
        try:
            ref = await run_gget(
                process,
                gget.ref,
                species,
                which=which_arg,
                release=release,
                ftp=False,  # dict with URLs + metadata; never downloads the files
                verbose=False,
                label="Ensembl reference lookup",
            )
        except GgetTimeout as e:
            await process.log(str(e))
            return f"ERROR: {e}. Ensembl FTP may be busy; try again later."
        except Exception as e:
            await process.log(f"Reference lookup failed: {e}")
            return (
                f"ERROR: reference lookup failed for '{species}': {e}. "
                "Check the species name (e.g. 'homo_sapiens') and release number."
            )

        if not ref:
            return f"No Ensembl reference files found for '{species}'."

        await process.create_artifact(
            mimetype="application/json",
            description=f"Ensembl reference file links for {species}"
            + (f" (release {release})" if release else ""),
            content=json.dumps(ref, indent=2).encode("utf-8"),
            metadata={"species": species, "which": which_arg, "release": release},
        )

        files = []
        for sp, entries in ref.items():
            for file_type, info in entries.items():
                files.append(
                    {
                        "file": file_type,
                        "url": info.get("ftp"),
                        "ensembl_release": info.get("ensembl_release"),
                        "release_date": info.get("release_date"),
                        "size": _human_size(info.get("bytes")),
                    }
                )
        await process.log(f"Found {len(files)} reference file link(s)")

    return json.dumps(
        {
            "species": species,
            "files": files,
            "note": "Links only; nothing was downloaded. Full details saved as an artifact for the user.",
        },
        default=str,
    )


reference_genome.handle_validation_error = True