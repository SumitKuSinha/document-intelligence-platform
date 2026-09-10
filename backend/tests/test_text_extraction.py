"""
Automated Tests for Text Extraction and OCR Service

Covers all business and architectural requirements:
- Digital text-based PDF extraction (page-by-page, preserved page numbers)
- Multi-page PDF extraction
- Raster image OCR for PNG, JPG, JPEG
- Scanned PDF OCR fallback (when pages contain images but no digital text)
- Extraction failures (corrupted files, empty files, limit violations, OCR errors)
- Verbatim text preservation (no modification or synthetic additions)
"""

from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import pypdf

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import pytest
except ImportError:
    pytest = None

from app.schemas.text_extraction import TextExtractionResult
from app.services.ocr_service import OCRError, OCRService
from app.services.text_extraction import TextExtractionService, extract_document_text
from tests.test_file_validation import (
    create_mock_jpeg,
    create_mock_pdf,
    create_mock_png,
)


def create_text_pdf(pages_text: list[str]) -> bytes:
    """Create a multi-page PDF containing actual digital text on each page."""
    writer = pypdf.PdfWriter()

    for text in pages_text:
        stream_data = f"BT\n/F1 12 Tf\n72 712 Td\n({text}) Tj\nET".encode("latin1")
        stream_len = len(stream_data)
        page_raw = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length " + str(stream_len).encode() + b" >>\nstream\n" + stream_data + b"\nendstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n500\n%%EOF\n"
        )
        r = pypdf.PdfReader(BytesIO(page_raw))
        writer.add_page(r.pages[0])

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_scanned_image_pdf(text_description: str = "scanned receipt") -> bytes:
    """Create a PDF with no digital text containing only an embedded raster image."""
    img = Image.new("RGB", (300, 200), color="white")
    buf = BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


class TestTextExtractionService(unittest.TestCase):
    """Test suite for TextExtractionService, PDFTextService, and OCRService."""

    def setUp(self) -> None:
        """Create a temporary directory and ensure OCR engine is reset."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        OCRService.set_ocr_engine(None)

    def tearDown(self) -> None:
        """Clean up temporary files and reset custom OCR engine."""
        self.temp_dir.cleanup()
        OCRService.set_ocr_engine(None)

    # --- Digital PDF Extraction Tests ---

    def test_text_based_pdf_single_page(self) -> None:
        """Verify that a digital 1-page PDF extracts exact text with correct page number."""
        sample_text = "Invoice Number: INV-2026-001 Total Due: $1,250.00"
        pdf_bytes = create_text_pdf([sample_text])

        result: TextExtractionResult = TextExtractionService.extract_text(
            pdf_bytes, filename="invoice.pdf"
        )

        self.assertEqual(result.extraction_status, "SUCCESS")
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertIn("Invoice Number: INV-2026-001", result.pages[0].text)
        self.assertIn("$1,250.00", result.pages[0].text)
        self.assertEqual(result.errors, [])

    def test_multi_page_pdf_extraction(self) -> None:
        """Verify that a 3-page PDF extracts text page-by-page preserving page numbers."""
        pages_content = [
            "Vendor: TechCorp Solutions Inc.",
            "Line Item 1: Cloud Server $400.00 Line Item 2: Support $100.00",
            "Total Amount: $500.00 Payment Terms: Net 30",
        ]
        pdf_bytes = create_text_pdf(pages_content)

        result = TextExtractionService.extract_text(pdf_bytes, filename="vendor_bill.pdf")

        self.assertEqual(result.extraction_status, "SUCCESS")
        self.assertEqual(len(result.pages), 3)

        for i, expected_text in enumerate(pages_content):
            page = result.pages[i]
            self.assertEqual(page.page_number, i + 1)
            self.assertIn(expected_text, page.text)

        self.assertEqual(result.errors, [])

    def test_verbatim_text_preservation(self) -> None:
        """Verify that text extraction does not invent or modify extracted characters."""
        exact_text = "Subtotal: $1,499.99 Tax (8.25%): $123.75 Total: $1,623.74"
        pdf_bytes = create_text_pdf([exact_text])

        result = extract_document_text(pdf_bytes, filename="tax_doc.pdf")

        self.assertEqual(result.extraction_status, "SUCCESS")
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].text.strip(), exact_text)

    # --- OCR Tests (Images & Scanned Documents) ---

    def test_png_ocr_extraction(self) -> None:
        """Verify that a PNG image is processed using OCR and returns page 1."""
        expected_ocr_text = "Store: SuperMart\nDate: 2026-09-10\nTotal: $45.60"
        OCRService.set_ocr_engine(lambda img: expected_ocr_text)

        png_bytes = create_mock_png()
        result = TextExtractionService.extract_text(png_bytes, filename="receipt.png")

        self.assertEqual(result.extraction_status, "SUCCESS")
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertEqual(result.pages[0].text, expected_ocr_text)
        self.assertEqual(result.errors, [])

    def test_jpg_ocr_extraction(self) -> None:
        """Verify that JPG and JPEG images are processed using OCR."""
        expected_ocr_text = "Gas Station #12\nFuel: $35.00"
        OCRService.set_ocr_engine(lambda img: expected_ocr_text)

        for ext in ("jpg", "jpeg"):
            jpeg_bytes = create_mock_jpeg()
            result = TextExtractionService.extract_text(
                jpeg_bytes, filename=f"fuel_bill.{ext}"
            )

            self.assertEqual(result.extraction_status, "SUCCESS")
            self.assertEqual(len(result.pages), 1)
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(result.pages[0].text, expected_ocr_text)

    def test_scanned_pdf_ocr_fallback(self) -> None:
        """Verify that a scanned PDF with no digital text triggers OCR on embedded images."""
        expected_ocr_text = "Scanned Tax Form W-2\nEmployer Identification: 12-3456789"
        OCRService.set_ocr_engine(lambda img: expected_ocr_text)

        scanned_pdf_bytes = create_scanned_image_pdf()
        result = TextExtractionService.extract_text(
            scanned_pdf_bytes, filename="scanned_w2.pdf"
        )

        self.assertEqual(result.extraction_status, "SUCCESS")
        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertEqual(result.pages[0].text, expected_ocr_text)
        self.assertTrue(any("extracted via OCR" in w for w in result.warnings))

    # --- Failure Cases ---

    def test_extraction_failure_empty_file(self) -> None:
        """Verify that an empty file fails at validation stage and returns FAILED status."""
        result = TextExtractionService.extract_text(b"", filename="empty.pdf")

        self.assertEqual(result.extraction_status, "FAILED")
        self.assertEqual(len(result.pages), 0)
        self.assertTrue(any("File is empty" in err for err in result.errors))

    def test_extraction_failure_unsupported_format(self) -> None:
        """Verify that an unsupported file format fails and returns FAILED status."""
        result = TextExtractionService.extract_text(
            b"Sample text file content", filename="report.txt"
        )

        self.assertEqual(result.extraction_status, "FAILED")
        self.assertEqual(len(result.pages), 0)
        self.assertTrue(any("Unsupported file type" in err for err in result.errors))

    def test_extraction_failure_pdf_exceeds_pages(self) -> None:
        """Verify that a 4-page PDF is rejected due to page count constraints."""
        oversized_pdf = create_mock_pdf(page_count=4)
        result = TextExtractionService.extract_text(
            oversized_pdf, filename="oversized.pdf"
        )

        self.assertEqual(result.extraction_status, "FAILED")
        self.assertEqual(len(result.pages), 0)
        self.assertTrue(any("exceeds maximum allowed page count" in err for err in result.errors))

    def test_extraction_failure_ocr_engine_error(self) -> None:
        """Verify that OCR engine failures return structured FAILED outcome with diagnostics."""
        def broken_ocr_engine(img):
            raise OCRError("Tesseract OCR binary not found in system PATH.")

        OCRService.set_ocr_engine(broken_ocr_engine)

        png_bytes = create_mock_png()
        result = TextExtractionService.extract_text(png_bytes, filename="receipt.png")

        self.assertEqual(result.extraction_status, "FAILED")
        self.assertEqual(len(result.pages), 0)
        self.assertTrue(any("OCR processing failed" in err for err in result.errors))

    # --- Tesseract Configuration Tests ---

    def test_tesseract_configuration_via_env(self) -> None:
        """Verify that TESSERACT_CMD environment variable is recognized, stripped, and applied."""
        import os
        from unittest.mock import patch
        import pytesseract

        test_path = r'"C:\Custom\Tesseract-OCR\tesseract.exe"'
        expected_path = r"C:\Custom\Tesseract-OCR\tesseract.exe"

        with patch.dict(os.environ, {"TESSERACT_CMD": test_path}):
            OCRService._configure_tesseract()
            self.assertEqual(pytesseract.pytesseract.tesseract_cmd, expected_path)

    def test_live_tesseract_ocr_if_available(self) -> None:
        """Verify that real Tesseract execution works when configured."""
        import os
        from unittest.mock import patch
        from pathlib import Path

        default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        tess_path = os.getenv("TESSERACT_CMD") or (default_win_path if Path(default_win_path).exists() else None)

        if tess_path:
            with patch.dict(os.environ, {"TESSERACT_CMD": tess_path}):
                OCRService._configure_tesseract()
                self.assertTrue(OCRService.is_tesseract_available())

                test_img = Image.new("RGB", (100, 40), color="white")
                text = OCRService.extract_text_from_image(test_img)
                self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
