"""
API Integration Tests for End-to-End Document Processing & Persistence

Tests cover:
- End-to-end processing: Upload -> Validation -> OCR -> Gemini Extraction -> Math Validation -> DB Persistence -> JSON Response
- Support for all four document types: invoice, balance_sheet, profit_and_loss, cash_flow_statement
- Statuses: COMPLETED, VALIDATION_FAILED, EXTRACTION_FAILED
- Error handling: invalid files, unsupported formats, corrupt files, empty files, limit violations
- Failure handling: text extraction errors, LLM failures, invalid document_type
- Retrieval: GET /api/v1/documents/{document_name} and 404 handling
- Collection: GET /api/v1/documents with pagination and filtering
- Persistence behavior: verify database state using DocumentRepository
- Zero external LLM calls (deterministic offline test fixtures)
"""

from pathlib import Path
import sys
import unittest

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models.document import Document
from app.repositories.document import DocumentRepository
from app.services.llm_client import LLMClient, LLMClientError
from tests.test_file_validation import (
    create_mock_jpeg,
    create_mock_pdf,
    create_mock_png,
)
from tests.test_text_extraction import create_text_pdf


class TestDocumentProcessingAPI(unittest.TestCase):
    """End-to-End API integration test suite with isolated SQLite database and mock LLM."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize in-memory SQLite engine and session factory for tests."""
        cls.engine = create_engine(
            "sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(cls.engine)
        cls.TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)

        def override_get_db():
            db = cls.TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up dependency overrides and drop tables."""
        app.dependency_overrides.clear()
        Base.metadata.drop_all(cls.engine)

    def setUp(self) -> None:
        """Reset mock responder and clean database before each test."""
        LLMClient.set_mock_responder(None)
        db = self.TestingSessionLocal()
        try:
            db.query(Document).delete()
            db.commit()
        finally:
            db.close()

    def tearDown(self) -> None:
        """Reset mock responder after each test."""
        LLMClient.set_mock_responder(None)

    # -------------------------------------------------------------------------
    # 1. Successful Processing: Invoice
    # -------------------------------------------------------------------------
    def test_process_invoice_success(self) -> None:
        """Verify successful processing, validation, and persistence of an invoice."""
        mock_payload = {
            "extracted_data": {
                "invoice_number": "INV-2024-001",
                "subtotal": 1000.0,
                "tax_amount": 80.0,
                "tax_rate": 0.08,
                "discount_amount": 0.0,
                "shipping_amount": 0.0,
                "total_amount": 1080.0,
                "currency": "USD",
                "line_items": [
                    {
                        "description": "Consulting Services",
                        "quantity": 10.0,
                        "unit_price": 100.0,
                        "total_price": 1000.0,
                    }
                ],
                "additional_fields": {},
            },
            "evidence": {
                "invoice_number": {"source_snippet": "INV-2024-001", "page_number": 1}
            },
        }
        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        pdf_bytes = create_text_pdf(["Invoice INV-2024-001 Total $1080.00"])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("invoice_sample.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_name"], "invoice_sample.pdf")
        self.assertEqual(data["document_type"], "invoice")
        self.assertEqual(data["processing_status"], "COMPLETED")
        self.assertIsNotNone(data["id"])
        self.assertTrue(data["file_validation"]["is_valid"])
        self.assertEqual(data["extracted_data"]["invoice_number"], "INV-2024-001")
        self.assertTrue(data["validations"]["is_valid"])
        self.assertEqual(data["validations"]["summary"]["failed_checks"], 0)

        # Verify database persistence
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            saved = repo.get_by_id(data["id"])
            self.assertIsNotNone(saved)
            self.assertEqual(saved.document_name, "invoice_sample.pdf")
            self.assertEqual(saved.processing_status, "COMPLETED")
            self.assertEqual(saved.extracted_data["invoice_number"], "INV-2024-001")
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 2. Successful Processing: Balance Sheet
    # -------------------------------------------------------------------------
    def test_process_balance_sheet_success(self) -> None:
        """Verify successful processing of a balance sheet."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Acme Holdings",
                "statement_date": "2023-12-31",
                "current_assets": 400000.0,
                "non_current_assets": 600000.0,
                "total_assets": 1000000.0,
                "current_liabilities": 200000.0,
                "non_current_liabilities": 300000.0,
                "total_liabilities": 500000.0,
                "retained_earnings": 300000.0,
                "share_capital": 200000.0,
                "total_equity": 500000.0,
                "total_liabilities_and_equity": 1000000.0,
                "line_items": [],
                "additional_fields": {},
            },
            "evidence": {},
        }
        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        pdf_bytes = create_text_pdf(["Acme Holdings Balance Sheet 2023..."])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("balance_sheet.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "balance_sheet"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_type"], "balance_sheet")
        self.assertEqual(data["processing_status"], "COMPLETED")
        self.assertTrue(data["validations"]["is_valid"])

    # -------------------------------------------------------------------------
    # 3. Successful Processing: Profit and Loss
    # -------------------------------------------------------------------------
    def test_process_profit_and_loss_success(self) -> None:
        """Verify successful processing of a profit & loss statement."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Beta Tech",
                "total_revenue": 500000.0,
                "cost_of_goods_sold": 200000.0,
                "gross_profit": 300000.0,
                "operating_expenses": 100000.0,
                "operating_income": 200000.0,
                "interest_expense": 10000.0,
                "tax_expense": 40000.0,
                "net_income": 150000.0,
                "line_items": [],
                "additional_fields": {},
            },
            "evidence": {},
        }
        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        pdf_bytes = create_text_pdf(["Beta Tech Statement of Operations..."])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("pnl_report.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "profit_and_loss"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_type"], "profit_and_loss")
        self.assertEqual(data["processing_status"], "COMPLETED")
        self.assertTrue(data["validations"]["is_valid"])

    # -------------------------------------------------------------------------
    # 4. Successful Processing: Cash Flow Statement
    # -------------------------------------------------------------------------
    def test_process_cash_flow_statement_success(self) -> None:
        """Verify successful processing of a cash flow statement."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Solaris Energy",
                "net_cash_from_operating_activities": 600000.0,
                "net_cash_from_investing_activities": -250000.0,
                "net_cash_from_financing_activities": -50000.0,
                "net_change_in_cash": 300000.0,
                "beginning_cash_balance": 200000.0,
                "ending_cash_balance": 500000.0,
                "line_items": [],
                "additional_fields": {},
            },
            "evidence": {},
        }
        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        pdf_bytes = create_text_pdf(["Solaris Statement of Cash Flows..."])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("cash_flow.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "cash_flow_statement"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_type"], "cash_flow_statement")
        self.assertEqual(data["processing_status"], "COMPLETED")
        self.assertTrue(data["validations"]["is_valid"])

    # -------------------------------------------------------------------------
    # 5. Mathematical Validation Failure
    # -------------------------------------------------------------------------
    def test_process_mathematical_validation_failure(self) -> None:
        """When math checks fail, status should be VALIDATION_FAILED and persisted."""
        mock_payload = {
            "extracted_data": {
                "invoice_number": "INV-MATH-ERROR",
                "subtotal": 1000.0,
                "tax_amount": 100.0,
                "total_amount": 1500.0,  # 1000 + 100 = 1100 != 1500
                "line_items": [],
            },
            "evidence": {},
        }
        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        pdf_bytes = create_text_pdf(["Invoice INV-MATH-ERROR"])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("math_err.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["validations"]["is_valid"])
        self.assertGreater(data["validations"]["summary"]["failed_checks"], 0)

        # Verify persisted with VALIDATION_FAILED status
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            saved = repo.get_by_name("math_err.pdf")
            self.assertIsNotNone(saved)
            self.assertEqual(saved.processing_status, "VALIDATION_FAILED")
        finally:
            db.close()

    # -------------------------------------------------------------------------
    # 6. File Validation Failures (HTTP 400)
    # -------------------------------------------------------------------------
    def test_process_invalid_file_empty(self) -> None:
        """Verify that an empty file returns HTTP 400 and does not run extraction."""
        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["file_validation"]["is_valid"])
        self.assertIn("File is empty (0 bytes).", data["file_validation"]["errors"])

        # Verify not persisted
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            self.assertIsNone(repo.get_by_name("empty.pdf"))
        finally:
            db.close()

    def test_process_invalid_file_oversized_pages(self) -> None:
        """Verify that a 4-page PDF returns HTTP 400."""
        pdf_bytes = create_mock_pdf(page_count=4)

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("oversized.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertEqual(data["file_validation"]["page_count"], 4)

    def test_process_invalid_file_corrupt(self) -> None:
        """Verify that a corrupted PDF returns HTTP 400."""
        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("corrupt.pdf", b"corrupted pdf bytes", "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data["file_validation"]["is_valid"])

    def test_process_invalid_file_unsupported_extension(self) -> None:
        """Verify that unsupported file extensions return HTTP 400."""
        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 400)

    def test_process_invalid_document_type(self) -> None:
        """Verify that an unsupported document_type returns HTTP 400."""
        pdf_bytes = create_text_pdf(["Some text"])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("doc.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "tax_return"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid document_type", response.json()["detail"])

    # -------------------------------------------------------------------------
    # 7. Text Extraction & LLM Failures
    # -------------------------------------------------------------------------
    def test_process_text_extraction_failure(self) -> None:
        """Empty text from valid mock PDF should yield EXTRACTION_FAILED."""
        # create_mock_pdf produces valid PDF with zero text/images
        pdf_bytes = create_mock_pdf(page_count=1)

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("no_text.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing_status"], "EXTRACTION_FAILED")
        self.assertIsNone(data["extracted_data"])

        # Check persisted
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            saved = repo.get_by_name("no_text.pdf")
            self.assertIsNotNone(saved)
            self.assertEqual(saved.processing_status, "EXTRACTION_FAILED")
        finally:
            db.close()

    def test_process_llm_failure_handling(self) -> None:
        """When LLM raises an error, pipeline returns EXTRACTION_FAILED gracefully."""

        def failing_responder(sys_p, usr_p):
            raise LLMClientError("Gemini rate limit exceeded.")

        LLMClient.set_mock_responder(failing_responder)

        pdf_bytes = create_text_pdf(["Invoice text content"])

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("llm_fail.pdf", pdf_bytes, "application/pdf")},
            data={"document_type": "invoice"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["processing_status"], "EXTRACTION_FAILED")
        self.assertIsNone(data["extracted_data"])

    # -------------------------------------------------------------------------
    # 8. Document Retrieval: GET /api/v1/documents/{document_name}
    # -------------------------------------------------------------------------
    def test_get_document_by_name_success(self) -> None:
        """Retrieve a previously persisted document by name."""
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            repo.create(
                document_name="target_invoice.pdf",
                document_type="invoice",
                processing_status="COMPLETED",
                file_validation={"is_valid": True},
                extracted_data={"invoice_number": "INV-777"},
                validations={"is_valid": True},
                doc_metadata={"pages": 1},
            )
        finally:
            db.close()

        response = self.client.get("/api/v1/documents/target_invoice.pdf")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_name"], "target_invoice.pdf")
        self.assertEqual(data["document_type"], "invoice")
        self.assertEqual(data["processing_status"], "COMPLETED")
        self.assertEqual(data["extracted_data"]["invoice_number"], "INV-777")
        self.assertEqual(data["metadata"]["pages"], 1)

    def test_get_document_by_name_not_found(self) -> None:
        """Requesting a nonexistent document returns HTTP 404."""
        response = self.client.get("/api/v1/documents/non_existent.pdf")

        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 9. Document Collection: GET /api/v1/documents
    # -------------------------------------------------------------------------
    def test_list_documents(self) -> None:
        """Verify list endpoint returns summary metadata and supports filtering."""
        db = self.TestingSessionLocal()
        try:
            repo = DocumentRepository(db)
            repo.create(
                document_name="doc1.pdf",
                document_type="invoice",
                processing_status="COMPLETED",
            )
            repo.create(
                document_name="doc2.pdf",
                document_type="balance_sheet",
                processing_status="COMPLETED",
            )
            repo.create(
                document_name="doc3.pdf",
                document_type="invoice",
                processing_status="EXTRACTION_FAILED",
            )
        finally:
            db.close()

        # List all
        res_all = self.client.get("/api/v1/documents")
        self.assertEqual(res_all.status_code, 200)
        data_all = res_all.json()
        self.assertEqual(data_all["total"], 3)
        self.assertEqual(len(data_all["documents"]), 3)

        # Filter by document_type
        res_inv = self.client.get("/api/v1/documents?document_type=invoice")
        self.assertEqual(res_inv.status_code, 200)
        data_inv = res_inv.json()
        self.assertEqual(data_inv["total"], 2)

        # Filter by processing_status
        res_fail = self.client.get("/api/v1/documents?processing_status=EXTRACTION_FAILED")
        self.assertEqual(res_fail.status_code, 200)
        data_fail = res_fail.json()
        self.assertEqual(data_fail["total"], 1)
        self.assertEqual(data_fail["documents"][0]["document_name"], "doc3.pdf")

    # -------------------------------------------------------------------------
    # 10. Health Check Regression
    # -------------------------------------------------------------------------
    def test_health_check_operational(self) -> None:
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})


if __name__ == "__main__":
    unittest.main()
