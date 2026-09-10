"""
Document Repository

Provides database access methods for Document records using SQLAlchemy 2.x.
Separates persistence logic from API routes and business services.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    """Repository for managing Document records in PostgreSQL."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        document_name: str,
        document_type: str,
        processing_status: str,
        file_validation: Optional[Dict[str, Any]] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
        validations: Optional[Dict[str, Any]] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Create and persist a new Document record.

        Args:
            document_name: Filename of the document.
            document_type: Category (invoice, balance_sheet, etc.).
            processing_status: Pipeline status (COMPLETED, VALIDATION_FAILED, etc.).
            file_validation: JSONB payload of file validation outcome.
            extracted_data: JSONB payload of extracted financial fields.
            validations: JSONB payload of deterministic validation checks.
            doc_metadata: Additional context/metadata.

        Returns:
            Document: Persisted SQLAlchemy ORM instance with assigned ID and timestamps.
        """
        doc = Document(
            document_name=document_name,
            document_type=document_type,
            processing_status=processing_status,
            file_validation=file_validation,
            extracted_data=extracted_data,
            validations=validations,
            doc_metadata=doc_metadata,
        )
        try:
            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)
            return doc
        except Exception:
            self.db.rollback()
            raise

    def get_by_id(self, document_id: int) -> Optional[Document]:
        """Fetch a document by its primary key ID."""
        stmt = select(Document).where(Document.id == document_id)
        return self.db.scalars(stmt).first()

    def get_by_name(self, document_name: str) -> Optional[Document]:
        """
        Fetch the most recent document by its filename/title.
        Orders by creation time descending to return latest upload.
        """
        stmt = (
            select(Document)
            .where(Document.document_name == document_name)
            .order_by(Document.created_at.desc(), Document.id.desc())
        )
        return self.db.scalars(stmt).first()

    def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        document_type: Optional[str] = None,
        processing_status: Optional[str] = None,
    ) -> List[Document]:
        """Fetch a paginated list of documents with optional category or status filters."""
        stmt = select(Document).order_by(Document.created_at.desc(), Document.id.desc())
        if document_type:
            stmt = stmt.where(Document.document_type == document_type)
        if processing_status:
            stmt = stmt.where(Document.processing_status == processing_status)
        stmt = stmt.offset(skip).limit(limit)
        return list(self.db.scalars(stmt).all())
