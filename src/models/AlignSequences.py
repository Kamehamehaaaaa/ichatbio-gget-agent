from pydantic import BaseModel, Field
from typing import Literal, Optional

class AlignSequences(BaseModel):
    """
    Align two or more nucleotide/protein sequences the user has already retrieved
    (via get_sequence_record) using MUSCLE5 (gget.muscle).
    """
    accession_numbers: list[str] = Field(
        min_length=2,
        description="Two or more NCBI accession numbers or sequence IDs the user wants aligned "
                    "or compared, e.g. ['MN908947', 'MT012345']. The agent fetches each sequence "
                    "via get_sequence_record before calling gget.muscle."
    )
    super5: bool = Field(
        default=False,
        description="Set True when aligning more than ~50 sequences, for speed."
    )