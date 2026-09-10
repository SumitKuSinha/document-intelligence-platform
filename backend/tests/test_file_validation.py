"""
Tests for File Validation Service

Covers all business and integrity constraints:
- Valid PDF (1, 2, 3 pages)
- PDF with more than 3 pages (> MAX_PDF_PAGES)
- Corrupted PDF (missing headers, truncated, malformed)
- Empty files (0 bytes)
- Valid PNG and JPG/JPEG images
- Corrupted images
- Unsupported file types (.txt, .exe, .docx)
- Non-existent file paths
- Stream and in-memory bytes input validation
"""

import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    import pytest
except ImportError:
    pytest = None

from app.schemas.file_validation import ValidationResult
from app.services.file_validation import FileValidationService, validate_document_file


# ---------------------------------------------------------------------------
# Test Data Builders (Generate compliant binary payloads in memory)
# ---------------------------------------------------------------------------

def create_mock_pdf(page_count: int) -> bytes:
    """Generate a valid ISO 32000-1 minimal PDF with the given number of pages."""
    objs = []
    # Object 1: Catalog
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: Pages tree
    kids = " ".join(f"{i + 3} 0 R" for i in range(page_count))
    objs.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {page_count} >>\nendobj\n".encode("latin1")
    )
    # Leaf Page objects
    for i in range(page_count):
        objs.append(
            f"{i + 3} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n".encode(
                "latin1"
            )
        )

    body = b"".join(objs)
    header = b"%PDF-1.4\n"
    trailer = (
        f"trailer\n<< /Size {page_count + 3} /Root 1 0 R >>\nstartxref\n{len(header) + len(body)}\n%%EOF\n".encode(
            "latin1"
        )
    )
    return header + body + trailer


def create_mock_png() -> bytes:
    """Generate a valid minimal 1x1 PNG image with correct IHDR, IDAT, and IEND chunks."""
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: 1x1 pixel, 8-bit grayscale
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT chunk: compressed scanline (filter byte 0 + pixel 0)
    raw_scanline = b"\x00\x00"
    idat_data = zlib.compress(raw_scanline)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF)
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + idat_crc

    # IEND chunk
    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend = struct.pack(">I", 0) + b"IEND" + iend_crc

    return sig + ihdr + idat + iend


def create_mock_jpeg() -> bytes:
    """Generate a valid minimal 1x1 JFIF JPEG image."""
    return bytes([
        0xFF, 0xD8,  # SOI
        0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00,  # JFIF APP0
        0xFF, 0xDB, 0x00, 0x43, 0x00,  # DQT
        0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14,
        0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A,
        0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C,
        0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34, 0x32,
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,  # SOF0
        0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,  # DHT
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,  # SOS
        0xBF, 0x00,  # Scan data
        0xFF, 0xD9,  # EOI
    ])


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestFileValidation(unittest.TestCase):
    """Test suite covering file upload validation requirements."""

    def setUp(self) -> None:
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    # --- PDF Tests ---

    def test_valid_pdf_single_page(self) -> None:
        """Verify that a valid 1-page PDF passes validation."""
        pdf_path = self.base_path / "invoice_1.pdf"
        pdf_path.write_bytes(create_mock_pdf(1))

        result: ValidationResult = FileValidationService.validate_file(pdf_path)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.file_type, "pdf")
        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.errors, [])

    def test_valid_pdf_multi_page(self) -> None:
        """Verify that valid 2-page and 3-page PDFs pass validation."""
        for pages in (2, 3):
            pdf_path = self.base_path / f"financial_report_{pages}p.pdf"
            pdf_path.write_bytes(create_mock_pdf(pages))

            result = FileValidationService.validate_file(pdf_path)

            self.assertTrue(result.is_valid, f"Failed for page count {pages}")
            self.assertEqual(result.file_type, "pdf")
            self.assertEqual(result.page_count, pages)
            self.assertEqual(result.errors, [])

    def test_pdf_exceeding_max_pages(self) -> None:
        """Verify that a PDF with more than 3 pages is rejected."""
        pdf_path = self.base_path / "long_contract_4p.pdf"
        pdf_path.write_bytes(create_mock_pdf(4))

        result = FileValidationService.validate_file(pdf_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "pdf")
        self.assertEqual(result.page_count, 4)
        self.assertTrue(any("exceeds maximum allowed page count" in err for err in result.errors))

    def test_corrupted_pdf_missing_header(self) -> None:
        """Verify that a PDF without %PDF- header is rejected as corrupted."""
        pdf_path = self.base_path / "corrupt_header.pdf"
        pdf_path.write_bytes(b"This is not a PDF file at all, just plain text.")

        result = FileValidationService.validate_file(pdf_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "pdf")
        self.assertIsNone(result.page_count)
        self.assertTrue(any("Corrupted or unreadable PDF" in err for err in result.errors))

    def test_corrupted_pdf_truncated(self) -> None:
        """Verify that a truncated PDF missing %%EOF is rejected."""
        pdf_path = self.base_path / "truncated.pdf"
        full_pdf = create_mock_pdf(2)
        # Truncate halfway through
        pdf_path.write_bytes(full_pdf[: len(full_pdf) // 2])

        result = FileValidationService.validate_file(pdf_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "pdf")
        self.assertTrue(any("Corrupted or unreadable PDF" in err for err in result.errors))

    def test_empty_pdf_file(self) -> None:
        """Verify that a 0-byte PDF is rejected with an empty file error."""
        pdf_path = self.base_path / "empty.pdf"
        pdf_path.write_bytes(b"")

        result = FileValidationService.validate_file(pdf_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "pdf")
        self.assertIsNone(result.page_count)
        self.assertIn("File is empty (0 bytes).", result.errors)

    # --- Image Tests ---

    def test_valid_png(self) -> None:
        """Verify that a valid PNG passes validation with null page count."""
        png_path = self.base_path / "receipt.png"
        png_path.write_bytes(create_mock_png())

        result = FileValidationService.validate_file(png_path)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.file_type, "png")
        self.assertIsNone(result.page_count)
        self.assertEqual(result.errors, [])

    def test_valid_jpeg(self) -> None:
        """Verify that valid JPG and JPEG files pass validation."""
        for ext in (".jpg", ".jpeg"):
            jpg_path = self.base_path / f"receipt{ext}"
            jpg_path.write_bytes(create_mock_jpeg())

            result = FileValidationService.validate_file(jpg_path)

            self.assertTrue(result.is_valid, f"Failed for extension {ext}")
            self.assertEqual(result.file_type, "jpeg")
            self.assertIsNone(result.page_count)
            self.assertEqual(result.errors, [])

    def test_corrupted_png(self) -> None:
        """Verify that a corrupted PNG is rejected."""
        png_path = self.base_path / "corrupt.png"
        bad_png = bytearray(create_mock_png())
        # Flip bytes in the IDAT chunk to cause CRC mismatch
        bad_png[25] ^= 0xFF
        png_path.write_bytes(bytes(bad_png))

        result = FileValidationService.validate_file(png_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "png")
        self.assertTrue(any("Corrupted or unreadable image" in err for err in result.errors))

    def test_corrupted_jpeg(self) -> None:
        """Verify that a truncated/corrupted JPEG is rejected."""
        jpg_path = self.base_path / "corrupt.jpg"
        jpg_path.write_bytes(b"\xff\xd8\xff\xe0truncated_jpeg_data")

        result = FileValidationService.validate_file(jpg_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "jpeg")
        self.assertTrue(any("Corrupted or unreadable image" in err for err in result.errors))

    def test_empty_image_file(self) -> None:
        """Verify that an empty image file is rejected."""
        img_path = self.base_path / "empty.png"
        img_path.write_bytes(b"")

        result = FileValidationService.validate_file(img_path)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.file_type, "png")
        self.assertIn("File is empty (0 bytes).", result.errors)

    # --- Edge Cases & Inputs ---

    def test_unsupported_file_type(self) -> None:
        """Verify that unsupported file extensions are rejected."""
        for ext in (".txt", ".docx", ".exe", ".csv"):
            unsupported_path = self.base_path / f"document{ext}"
            unsupported_path.write_bytes(b"dummy content")

            result = FileValidationService.validate_file(unsupported_path)

            self.assertFalse(result.is_valid)
            self.assertIsNone(result.file_type)
            self.assertIsNone(result.page_count)
            self.assertTrue(any("Unsupported file type" in err for err in result.errors))

    def test_nonexistent_file(self) -> None:
        """Verify that a non-existent file path produces a clear error."""
        nonexistent = self.base_path / "does_not_exist.pdf"

        result = FileValidationService.validate_file(nonexistent)

        self.assertFalse(result.is_valid)
        self.assertIsNone(result.file_type)
        self.assertTrue(any("File does not exist" in err for err in result.errors))

    def test_raw_bytes_and_stream_inputs(self) -> None:
        """Verify that raw bytes and BytesIO streams can be validated directly."""
        pdf_bytes = create_mock_pdf(2)

        # Test raw bytes with filename
        res_bytes = FileValidationService.validate_file(pdf_bytes, filename="uploaded.pdf")
        self.assertTrue(res_bytes.is_valid)
        self.assertEqual(res_bytes.page_count, 2)

        # Test BytesIO stream with filename
        stream = io.BytesIO(create_mock_png())
        res_stream = validate_document_file(stream, filename="receipt.png")
        self.assertTrue(res_stream.is_valid)
        self.assertEqual(res_stream.file_type, "png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
