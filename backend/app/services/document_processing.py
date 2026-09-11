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
    def reconcile_document_type(cls, requested_type: str, text: str) -> str:
        """
        Reconcile the requested document type against the actual semantic content
        of the extracted text (without using filename or vendor-specific heuristics).

        If a user/client sends default 'invoice', but the text explicitly reports
        a Balance Sheet, Profit & Loss, or Cash Flow Statement, route it to the
        matching financial statement type.
        """
        if not text:
            return requested_type

        text_upper = text[:4000].upper()

        is_bs = any(h in text_upper for h in [
            "BALANCE SHEET",
            "CONSOLIDATED BALANCE SHEET",
            "STATEMENT OF FINANCIAL POSITION",
            "STATEMENT OF ASSETS AND LIABILITIES",
            "CAPITAL AND LIABILITIES",
        ])
        is_pnl = any(h in text_upper for h in [
            "PROFIT AND LOSS",
            "PROFIT & LOSS",
            "STATEMENT OF PROFIT AND LOSS",
            "INCOME STATEMENT",
            "STATEMENT OF OPERATIONS",
            "STATEMENT OF EARNINGS",
            "STATEMENT OF COMPREHENSIVE INCOME",
        ])
        is_cf = any(h in text_upper for h in [
            "CASH FLOW STATEMENT",
            "STATEMENT OF CASH FLOWS",
            "CASH FLOWS STATEMENT",
        ])
        is_inv = any(h in text_upper for h in [
            "TAX INVOICE",
            "COMMERCIAL INVOICE",
            "BILL OF SUPPLY",
            "PROFORMA INVOICE",
            "INVOICE NUMBER",
            "INVOICE DATE",
            "BILL TO",
            "SHIP TO",
            "GSTIN",
        ])

        if is_bs and not is_inv:
            return "balance_sheet"
        if is_pnl and not is_inv:
            return "profit_and_loss"
        if is_cf and not is_inv:
            return "cash_flow_statement"
        if is_inv and not is_bs and not is_pnl and not is_cf:
            return "invoice"

        return requested_type

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
        doc_type_clean = cls.reconcile_document_type(doc_type_clean, full_text)
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
