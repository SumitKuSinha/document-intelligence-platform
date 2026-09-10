"""
Document ORM Model

This module defines the SQLAlchemy 2.x ORM model for storing and tracking
uploaded documents, their extraction results, validation status, and metadata.
"""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Document(Base):
    """
    SQLAlchemy ORM model mapped to the PostgreSQL 'documents' table.

    Represents a processed document record containing raw file metadata,
    extracted structured data, validation rules execution status, and audit timestamps.
    """

    __tablename__ = "documents"

    # Primary key identifier
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        doc="Unique primary key identifier for the document record.",
    )

    # Document filename or original name
    document_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="Original filename or document title; indexed for fast lookups.",
    )

    # Document classification type (e.g., invoice, receipt, contract, id_card)
    document_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Document category or type classification.",
    )

    # Lifecycle state (e.g., PENDING, PROCESSING, COMPLETED, FAILED)
    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
        doc="Current pipeline status of the document; defaults to PENDING.",
    )

    # Pre-processing file validation results (mime-type check, size, page count, etc.)
    file_validation: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        doc="JSONB payload capturing file integrity and format checks.",
    )

    # Extracted structured data output (OCR / LLM extraction fields)
    extracted_data: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        doc="JSONB payload containing key-value pairs extracted from the document.",
    )

    # Business rule validation outputs (e.g., math checks, date validations)
    validations: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        doc="JSONB payload detailing rule evaluation outcomes and validation flags.",
    )

    # Additional contextual metadata (e.g., source channel, upload user, tags)
    # Note: 'metadata' is a reserved attribute name on DeclarativeBase classes in SQLAlchemy,
    # so the Python attribute is named 'doc_metadata' while mapping to column 'metadata' in PostgreSQL.
    doc_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
        doc="PostgreSQL JSONB column named 'metadata' for arbitrary document attributes.",
    )

    # Record creation audit timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        doc="Timestamp (with timezone) when the document record was created.",
    )

    def __repr__(self) -> str:
        """Informative string representation for debugging and logging."""
        return (
            f"<Document(id={self.id}, "
            f"document_name='{self.document_name}', "
            f"document_type='{self.document_type}', "
            f"processing_status='{self.processing_status}', "
            f"created_at={self.created_at})>"
        )
