import typing
import pydantic


class Citation(pydantic.BaseModel):
    pdf_number: int = pydantic.Field(..., description="PDF number from metadata")
    page: int = pydantic.Field(..., description="Page number in the PDF")
    file_name: str = pydantic.Field(..., description="Source PDF file name")
    quote: str = pydantic.Field(..., description="Exact quote from the document")


class LegalAnswer(pydantic.BaseModel):
    answer_found: bool = pydantic.Field(
        ..., description="True if the answer is explicitly found in the context"
    )
    act_name: typing.Optional[str] = pydantic.Field(
        None, description="Name of the Act, if explicitly mentioned"
    )
    section: typing.Optional[str] = pydantic.Field(
        None, description="Section number, if explicitly mentioned"
    )
    explanation: typing.Optional[str] = pydantic.Field(
        None,
        description="Briefly restate what the Act defines.",
    )
    citations: typing.List[Citation] = pydantic.Field(
        default_factory=list,
        description="List of supporting citations from the context",
    )
