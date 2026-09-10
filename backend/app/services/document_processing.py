"""
Document Processing Pipeline Service

Coordinates the complete end-to-end processing pipeline:
1. File Validation (FileValidationService)
2. Text / OCR Extraction (TextExtractionService)
3. Financial Field Extraction (FinancialExtractionService + Gemini)
4. Deterministic Financial Validation (FinancialValidationService)
5. Database Persistence (DocumentRepository)
"""

import logging
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from app.repositories.document import DocumentRepository
from app.schemas.document import DocumentProcessResponse
from app.services.file_validation import FileValidationService
from app.services.financial_extraction import FinancialExtractionService
from app.services.financial_validation import FinancialValidationService
from app.services.text_extraction import TextExtractionService

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service orchestrating the end-to-end document intelligence pipeline."""

    ALLOWED_DOCUMENT_TYPES = {
        "invoice",
        "balance_sheet",
        "profit_and_loss",
        "cash_flow_statement",
    }

    @classmethod
    def process_document(
        cls,
        file_bytes: bytes,
        filename: str,
        document_type: str,
        db: Session,
    ) -> Tuple[DocumentProcessResponse, int]:
        """
        Execute the end-to-end processing flow on an uploaded document.

        Args:
            file_bytes: Raw binary content of the uploaded document.
            filename: Original name of the document.
            document_type: Category (invoice, balance_sheet, etc.).
            db: Active SQLAlchemy database session.

        Returns:
            Tuple[DocumentProcessResponse, int]: Structured API response payload and HTTP status code.
        """
        repo = DocumentRepository(db)
        doc_type_clean = (document_type or "invoice").strip().lower()

        # Step 1: File Validation
        file_val = FileValidationService.validate_file(file_bytes, filename)
        if not file_val.is_valid:
            # Do NOT run extraction; return HTTP 400 Bad Request
            response = DocumentProcessResponse(
                id=None,
                document_name=filename,
                document_type=doc_type_clean,
                processing_status="VALIDATION_FAILED",
                file_validation=file_val,
                extracted_data=None,
                validations=None,
                metadata={"file_errors": file_val.errors},
                created_at=None,
            )
            return response, 400

        # Step 2: Text / OCR Extraction
        text_result = TextExtractionService.extract_text(file_bytes, filename)
        full_text = "\n\n".join(p.text for p in text_result.pages if p.text).strip()
        if text_result.extraction_status == "FAILED" or not full_text:
            diag = {
                "extraction_status": text_result.extraction_status,
                "errors": text_result.errors,
                "warnings": text_result.warnings,
            }
            try:
                saved_doc = repo.create(
                    document_name=filename,
                    document_type=doc_type_clean,
                    processing_status="EXTRACTION_FAILED",
                    file_validation=file_val.model_dump(),
                    extracted_data=None,
                    validations=None,
                    doc_metadata=diag,
                )
                doc_id = saved_doc.id
                created_at = saved_doc.created_at
            except Exception as exc:
                logger.error(f"Failed to persist document after extraction failure: {exc}")
                doc_id = None
                created_at = None

            response = DocumentProcessResponse(
                id=doc_id,
                document_name=filename,
                document_type=doc_type_clean,
                processing_status="EXTRACTION_FAILED",
                file_validation=file_val.model_dump(),
                extracted_data=None,
                validations=None,
                metadata=diag,
                created_at=created_at,
            )
            return response, 200

        # Step 3: Financial Field Extraction (Gemini / LLM)
        extraction_result = FinancialExtractionService.extract_financial_data(
            document_type=doc_type_clean,
            extracted_text=full_text,
            pages=text_result.pages,
        )

        if extraction_result.status == "FAILED" or not extraction_result.data:
            diag = {
                "extraction_status": extraction_result.status,
                "errors": extraction_result.errors,
                "warnings": extraction_result.warnings,
            }
            try:
                saved_doc = repo.create(
                    document_name=filename,
                    document_type=doc_type_clean,
                    processing_status="EXTRACTION_FAILED",
                    file_validation=file_val.model_dump(),
                    extracted_data=None,
                    validations=None,
                    doc_metadata=diag,
                )
                doc_id = saved_doc.id
                created_at = saved_doc.created_at
            except Exception as exc:
                logger.error(f"Failed to persist document after LLM extraction failure: {exc}")
                doc_id = None
                created_at = None

            response = DocumentProcessResponse(
                id=doc_id,
                document_name=filename,
                document_type=doc_type_clean,
                processing_status="EXTRACTION_FAILED",
                file_validation=file_val.model_dump(),
                extracted_data=None,
                validations=None,
                metadata=diag,
                created_at=created_at,
            )
            return response, 200

        # Step 4: Deterministic Financial Validation
        validation_result = FinancialValidationService.validate(
            document_type=doc_type_clean,
            data=extraction_result.data,
        )

        processing_status = "COMPLETED" if validation_result.is_valid else "VALIDATION_FAILED"

        metadata = {
            "page_count": len(text_result.pages),
            "evidence_count": len(extraction_result.evidence),
            "validation_summary": validation_result.summary.model_dump(),
        }

        # Step 5: Database Persistence
        try:
            saved_doc = repo.create(
                document_name=filename,
                document_type=doc_type_clean,
                processing_status=processing_status,
                file_validation=file_val.model_dump(),
                extracted_data=extraction_result.data,
                validations=validation_result.model_dump(),
                doc_metadata=metadata,
            )
            doc_id = saved_doc.id
            created_at = saved_doc.created_at
        except Exception as exc:
            logger.error(f"Database persistence failed: {exc}")
            doc_id = None
            created_at = None

        response = DocumentProcessResponse(
            id=doc_id,
            document_name=filename,
            document_type=doc_type_clean,
            processing_status=processing_status,
            file_validation=file_val.model_dump(),
            extracted_data=extraction_result.data,
            validations=validation_result.model_dump(),
            metadata=metadata,
            created_at=created_at,
        )
        return response, 200
