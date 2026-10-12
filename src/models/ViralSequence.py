from pydantic import BaseModel, Field
from typing import Literal, Optional

class ViralSequence(BaseModel):
    """
    Retrieve viral nucleotide sequences and metadata from NCBI Virus (gget.virus).
    """
    virus: str = Field(
        description="Virus taxon name (e.g. 'Dengue virus'), NCBI taxon ID, or accession number(s)."
    )
    is_accession: bool = Field(
        default=False,
        description="True if `virus` is an accession number rather than a taxon name/ID."
    )
    host: Optional[str] = Field(
        default=None,
        description="Filter by host organism name or NCBI Taxonomy ID, e.g. 'Homo sapiens' or 'Aedes aegypti'."
    )
    nuc_completeness: Optional[Literal["complete", "partial"]] = Field(default=None)
    min_seq_length: Optional[int] = Field(default=None, description="Minimum sequence length in bases.")
    max_seq_length: Optional[int] = Field(default=None, description="Maximum sequence length in bases.")
    segment: Optional[str] = Field(default=None, description="Segment filter, for segmented viruses (e.g. influenza).")
    source_database: Optional[Literal["genbank", "refseq"]] = Field(default=None)
    genbank_metadata: bool = Field(default=False, description="Fetch detailed GenBank metadata alongside sequences.")

    # Deliberately NOT exposed to the LLM extractor: --download_all_accessions.
    # Unfiltered, this can request the entire Viruses taxonomy (hours, large disk).
    # Gate it behind an explicit confirmation step in your orchestration code, not free-text extraction.
