"""
Text Extraction Schemas

Data models representing page-level text extraction and OCR results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """
    Extracted text content for an individual document page.

    Attributes:
        page_number: 1-indexed page number within the document.
        text: Raw extracted plain text for this specific page.
    """

    page_number: int = Field(..., description="1-indexed page number within the document.", ge=1)
    text: str = Field(..., description="Raw text extracted from this page.")
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw image bytes if this page is scanned / image-based.",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of page image (e.g. 'image/jpeg', 'image/png').",
    )
    is_scanned: bool = Field(
        default=False,
        description="True if page contains no digital text and was processed as an image.",
    )


class TextExtractionResult(BaseModel):
    """
    Structured outcome returned by the text extraction and OCR pipeline.

    Attributes:
        extraction_status: Outcome status ('SUCCESS', 'PARTIAL_SUCCESS', or 'FAILED').
        pages: List of extracted pages with page numbers and text content.
        errors: Diagnostic error messages explaining extraction failures.
        warnings: Informational messages (e.g., OCR fallback notes).
    """

    extraction_status: str = Field(
        ...,
        description="Extraction lifecycle outcome: 'SUCCESS', 'PARTIAL_SUCCESS', or 'FAILED'.",
        examples=["SUCCESS", "FAILED"],
    )
    pages: List[ExtractedPage] = Field(
        default_factory=list,
        description="Sequential list of document pages with extracted text.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of fatal error messages encountered during extraction.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of non-fatal warnings encountered during extraction.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result model to a standard dictionary."""
        return self.model_dump()
