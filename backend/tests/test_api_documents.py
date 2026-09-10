"""
API Integration Tests for Document Processing Endpoint

Tests POST /api/v1/documents/process using FastAPI TestClient:
- Valid PDF documents (single and multi-page within limit)
- Valid image documents (PNG, JPG/JPEG)
- Unsupported file formats (.txt, .exe, .csv)
- Empty files (0 bytes)
- PDFs exceeding the 3-page business limit
- Corrupted / malformed files
- Health endpoint regression check
"""

from pathlib import Path
import sys
import unittest

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import pytest
except ImportError:
    pytest = None

from fastapi.testclient import TestClient

from app.main import app
from tests.test_file_validation import (
    create_mock_jpeg,
    create_mock_pdf,
    create_mock_png,
)


class TestDocumentProcessAPI(unittest.TestCase):
    """Integration test suite for POST /api/v1/documents/process."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize FastAPI TestClient."""
        cls.client = TestClient(app)

    # --- Success Cases (HTTP 200) ---

    def test_process_valid_pdf(self) -> None:
        """Verify that a valid PDF returns HTTP 200 and VALIDATED status."""
        pdf_bytes = create_mock_pdf(page_count=2)

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("invoice_sample.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_name"], "invoice_sample.pdf")
        self.assertEqual(data["processing_status"], "VALIDATED")
        self.assertTrue(data["file_validation"]["is_valid"])
        self.assertEqual(data["file_validation"]["file_type"], "pdf")
        self.assertEqual(data["file_validation"]["page_count"], 2)
        self.assertEqual(data["file_validation"]["errors"], [])

    def test_process_valid_png(self) -> None:
        """Verify that a valid PNG returns HTTP 200 and null page count."""
        png_bytes = create_mock_png()

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("receipt.png", png_bytes, "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_name"], "receipt.png")
        self.assertEqual(data["processing_status"], "VALIDATED")
        self.assertTrue(data["file_validation"]["is_valid"])
        self.assertEqual(data["file_validation"]["file_type"], "png")
        self.assertIsNone(data["file_validation"]["page_count"])
        self.assertEqual(data["file_validation"]["errors"], [])

    def test_process_valid_jpeg(self) -> None:
        """Verify that valid JPG/JPEG images return HTTP 200."""
        for filename in ("bill.jpg", "bill.jpeg"):
            jpeg_bytes = create_mock_jpeg()

            response = self.client.post(
                "/api/v1/documents/process",
                files={"file": (filename, jpeg_bytes, "image/jpeg")},
            )

            self.assertEqual(response.status_code, 200, f"Failed for {filename}")
            data = response.json()
            self.assertEqual(data["document_name"], filename)
            self.assertEqual(data["processing_status"], "VALIDATED")
            self.assertTrue(data["file_validation"]["is_valid"])
            self.assertEqual(data["file_validation"]["file_type"], "jpeg")
            self.assertIsNone(data["file_validation"]["page_count"])

    # --- Error Cases (HTTP 400) ---

    def test_process_unsupported_file(self) -> None:
        """Verify that an unsupported file format returns HTTP 400 with diagnostics."""
        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("notes.txt", b"Unsupported content", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["document_name"], "notes.txt")
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["file_validation"]["is_valid"])
        self.assertTrue(any("Unsupported file type" in err for err in data["file_validation"]["errors"]))

    def test_process_empty_file(self) -> None:
        """Verify that a 0-byte file returns HTTP 400."""
        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("empty_invoice.pdf", b"", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["document_name"], "empty_invoice.pdf")
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["file_validation"]["is_valid"])
        self.assertIn("File is empty (0 bytes).", data["file_validation"]["errors"])

    def test_process_pdf_exceeding_pages(self) -> None:
        """Verify that a PDF with 4 pages returns HTTP 400."""
        pdf_bytes = create_mock_pdf(page_count=4)

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("oversized_contract.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["document_name"], "oversized_contract.pdf")
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["file_validation"]["is_valid"])
        self.assertEqual(data["file_validation"]["page_count"], 4)
        self.assertTrue(
            any("exceeds maximum allowed page count" in err for err in data["file_validation"]["errors"])
        )

    def test_process_corrupted_file(self) -> None:
        """Verify that a corrupted PDF returns HTTP 400."""
        corrupted_bytes = b"%PDF-1.4 garbage corrupted data without objects or eof"

        response = self.client.post(
            "/api/v1/documents/process",
            files={"file": ("corrupt_doc.pdf", corrupted_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["document_name"], "corrupt_doc.pdf")
        self.assertEqual(data["processing_status"], "VALIDATION_FAILED")
        self.assertFalse(data["file_validation"]["is_valid"])
        self.assertTrue(
            any("Corrupted or unreadable PDF" in err for err in data["file_validation"]["errors"])
        )

    # --- Health Check Regression ---

    def test_health_endpoint(self) -> None:
        """Ensure existing health check endpoint is operational."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
