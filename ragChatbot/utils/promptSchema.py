from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    pdf_number: int = Field(..., description="PDF number from metadata")
    page: int = Field(..., description="Page number in the PDF")
    file_name: str = Field(..., description="Source PDF file name")
    quote: str = Field(..., description="Exact quote from the document")


class LegalAnswer(BaseModel):
    answer_found: bool = Field(
        ..., description="True if the answer is explicitly found in the context"
    )
    act_name: Optional[str] = Field(
        None, description="Name of the Act, if explicitly mentioned"
    )
    section: Optional[str] = Field(
        None, description="Section number, if explicitly mentioned"
    )
    explanation: Optional[str] = Field(
        None,
        description="Briefly restate what the Act defines.",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of supporting citations from the context",
    )
