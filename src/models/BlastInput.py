from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class BlastInput(BaseModel):
    """
    Identify the likely source organism(s) of a nucleotide or protein sequence via BLAST
    (gget.blast). Use when the user provides a raw sequence, or references an accession/
    record they want identified/matched against known sequences.
    """
    sequence: Optional[str] = Field(
        default=None,
        description="Raw nucleotide or amino acid sequence text pasted or described by the user. "
                    "Strip FASTA headers/whitespace if present."
    )
    accession_number: Optional[str] = Field(
        default=None,
        description="An NCBI accession number the user refers to instead of pasting a sequence "
                    "(e.g. 'is MN908947 similar to anything'). The agent must resolve this to "
                    "sequence text via get_sequence_record before calling gget.blast."
    )
    program: Optional[Literal["blastn", "blastp", "blastx", "tblastn", "tblastx"]] = Field(
        default=None,
        description="BLAST program. Leave unset to auto-detect. Use 'blastn' for DNA-vs-DNA "
                    "species-identification queries."
    )
    database: Optional[Literal["nt", "refseq_rna", "pdbnt", "nr", "swissprot", "pdbaa", "refseq_protein"]] = Field(
        default=None,
        description="Target BLAST database. 'nt' is the standard choice for identifying an "
                    "unknown DNA sequence's source organism."
    )
    limit: int = Field(default=3, ge=1, le=50, description="Max hits to return.")
    expect: float = Field(default=10.0, description="E-value cutoff; lower = stricter match.")

    @model_validator(mode="after")
    def one_input_required(self):
        if not self.sequence and not self.accession_number:
            raise ValueError("Either sequence or accession_number must be provided.")
        return self
