"""
Document Schemas

Pydantic models for document processing endpoints.
"""

from pydantic import BaseModel, Field

from app.schemas.file_validation import ValidationResult


class DocumentProcessResponse(BaseModel):
    """
    Response model for the document processing and file validation endpoint.

    Attributes:
        document_name: The original name of the uploaded document.
        processing_status: Pipeline status (e.g., 'VALIDATED' or 'VALIDATION_FAILED').
        file_validation: Detailed structural validation diagnostics.
    """

    document_name: str = Field(
        ...,
        description="Original filename of the uploaded document.",
        examples=["invoice.pdf"],
    )
    processing_status: str = Field(
        ...,
        description="Document lifecycle status after file-level validation.",
        examples=["VALIDATED", "VALIDATION_FAILED"],
    )
    file_validation: ValidationResult = Field(
        ...,
        description="Structured validation outcome including format, page count, errors, and warnings.",
    )
