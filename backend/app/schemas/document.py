"""
Document Schemas

Pydantic models for document processing endpoints, single document retrieval,
and document collection listing.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from app.schemas.file_validation import ValidationResult


class DocumentProcessResponse(BaseModel):
    """
    Response model for the end-to-end document processing endpoint.

    Attributes:
        id: Database primary key identifier (if persisted).
        document_name: The original filename of the uploaded document.
        document_type: Document category (invoice, balance_sheet, etc.).
        processing_status: Pipeline lifecycle status.
        file_validation: Detailed structural file validation diagnostics.
        extracted_data: Extracted structured financial data and line items.
        validations: Deterministic financial validation results.
        metadata: Summary metrics, audit timestamps, and diagnostic notes.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: Optional[int] = Field(
        default=None,
        description="Database primary key identifier.",
    )
    document_name: str = Field(
        ...,
        description="Original filename of the uploaded document.",
        examples=["invoice.pdf"],
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Document category classification.",
        examples=["invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement"],
    )
    processing_status: str = Field(
        ...,
        description="Document lifecycle status across the processing pipeline.",
        examples=["COMPLETED", "VALIDATION_FAILED", "EXTRACTION_FAILED", "PROCESSING_FAILED"],
    )
    file_validation: Optional[Union[ValidationResult, Dict[str, Any]]] = Field(
        default=None,
        description="Structured file validation outcome.",
    )
    extracted_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured financial fields extracted by the LLM.",
    )
    validations: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Deterministic mathematical and accounting reconciliation checks.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("metadata", "doc_metadata"),
        description="Contextual metadata, page counts, or processing diagnostics.",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the document was persisted.",
    )


class DocumentSummaryResponse(BaseModel):
    """
    Summary representation of a persisted document for list views.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int = Field(..., description="Database record ID")
    document_name: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Financial document category")
    processing_status: str = Field(..., description="Lifecycle status")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("metadata", "doc_metadata"),
        description="Summary metadata",
    )


class DocumentListResponse(BaseModel):
    """
    Paginated list of processed document summaries.
    """

    total: int = Field(..., description="Total records returned")
    documents: List[DocumentSummaryResponse] = Field(
        ...,
        description="List of document summaries",
    )
