"""
File Validation Service

Validates uploaded financial documents (PDF, PNG, JPG/JPEG) prior to downstream processing:
- Validates file presence and non-zero length.
- Verifies supported extensions and MIME types.
- Validates structural readability and detects file corruption.
- Enforces maximum page limits on PDF documents (maximum 3 pages).
- Validates that image files can be opened and decoded properly.
- Returns a structured ValidationResult with detailed diagnostics instead of raw exceptions.
"""

from io import BytesIO
from pathlib import Path
import re
import struct
from typing import BinaryIO, Dict, Optional, Set, Union
import zlib

# Optional imports for enhanced format parsing when installed
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from PIL import Image as PIL_Image
except ImportError:
    PIL_Image = None

from app.schemas.file_validation import ValidationResult


class FileValidationError(Exception):
    """Exception raised when document file validation encounters a critical error."""

    def __init__(self, message: str, errors: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


class FileValidationService:
    """
    Production-oriented document validation service.

    Validates uploaded financial documents according to business constraints:
    1. Supported formats: PDF, PNG, JPG/JPEG.
    2. File exists and contains data (> 0 bytes).
    3. Structural integrity: File must not be corrupt or unreadable.
    4. PDF page limit: Maximum 3 pages per document.
    5. Image validity: Image headers and payload must be decodable.
    """

    MAX_PDF_PAGES: int = 3
    SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg"}

    # Canonical format mapping
    EXTENSION_MAP: Dict[str, str] = {
        ".pdf": "pdf",
        ".png": "png",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
    }

    @classmethod
    def validate_file(
        cls,
        file_input: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate an uploaded document file from a path, raw bytes, or stream.

        Args:
            file_input: File path (str/Path), raw bytes, or binary file-like stream.
            filename: Optional original filename (required if passing raw bytes or stream).

        Returns:
            ValidationResult: Structured outcome detailing validity, detected format,
                             page count (for PDFs), and diagnostic errors/warnings.
        """
        # 1. Resolve content bytes and filename
        content: bytes = b""
        resolved_name: Optional[str] = filename

        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    file_type=None,
                    page_count=None,
                    errors=[f"File does not exist: {path}"],
                )
            if not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    file_type=None,
                    page_count=None,
                    errors=[f"Specified path is not a file: {path}"],
                )
            resolved_name = resolved_name or path.name
            try:
                content = path.read_bytes()
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type=None,
                    page_count=None,
                    errors=[f"Failed to read file from disk: {exc}"],
                )

        elif isinstance(file_input, bytes):
            content = file_input

        elif hasattr(file_input, "read"):
            try:
                content = file_input.read()
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type=None,
                    page_count=None,
                    errors=[f"Failed to read stream content: {exc}"],
                )
        else:
            return ValidationResult(
                is_valid=False,
                file_type=None,
                page_count=None,
                errors=["Invalid file_input: Must be a file path, bytes, or binary stream."],
            )

        # 2. Check for empty file
        ext = Path(resolved_name).suffix.lower() if resolved_name else ""
        detected_type = cls.EXTENSION_MAP.get(ext)

        if len(content) == 0:
            return ValidationResult(
                is_valid=False,
                file_type=detected_type,
                page_count=None,
                errors=["File is empty (0 bytes)."],
            )

        # 3. Validate supported extension/type
        if not ext:
            # Attempt to infer extension from magic bytes
            ext = cls._infer_extension(content)
            detected_type = cls.EXTENSION_MAP.get(ext)

        if ext not in cls.SUPPORTED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                file_type=None,
                page_count=None,
                errors=[
                    f"Unsupported file type: '{ext or 'unknown'}'. "
                    f"Supported formats are: PDF, PNG, JPG/JPEG."
                ],
            )

        canonical_type = cls.EXTENSION_MAP[ext]

        # 4. Dispatch format-specific content validation
        if canonical_type == "pdf":
            return cls._validate_pdf(content)
        elif canonical_type in ("png", "jpeg"):
            return cls._validate_image(content, canonical_type)

        return ValidationResult(
            is_valid=False,
            file_type=None,
            page_count=None,
            errors=[f"Unhandled format: {canonical_type}"],
        )

    # ---------------------------------------------------------------------------
    # Format-Specific Validators
    # ---------------------------------------------------------------------------

    @classmethod
    def _validate_pdf(cls, content: bytes) -> ValidationResult:
        """Validate PDF readability, structure, and enforce page limit (<= 3)."""
        page_count: Optional[int] = None

        # Try pypdf if installed
        if pypdf is not None:
            try:
                reader = pypdf.PdfReader(BytesIO(content))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        return ValidationResult(
                            is_valid=False,
                            file_type="pdf",
                            page_count=None,
                            errors=["PDF is password-protected or encrypted and cannot be processed."],
                        )
                page_count = len(reader.pages)
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type="pdf",
                    page_count=None,
                    errors=[f"Corrupted or unreadable PDF: {exc}"],
                )
        else:
            # Fallback pure-Python PDF structure parser
            try:
                page_count = cls._parse_pdf_pages_fallback(content)
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type="pdf",
                    page_count=None,
                    errors=[f"Corrupted or unreadable PDF: {exc}"],
                )

        if page_count is None or page_count <= 0:
            return ValidationResult(
                is_valid=False,
                file_type="pdf",
                page_count=0,
                errors=["PDF contains no readable pages."],
            )

        if page_count > cls.MAX_PDF_PAGES:
            return ValidationResult(
                is_valid=False,
                file_type="pdf",
                page_count=page_count,
                errors=[
                    f"PDF exceeds maximum allowed page count of {cls.MAX_PDF_PAGES} pages "
                    f"(document has {page_count} pages)."
                ],
            )

        return ValidationResult(
            is_valid=True,
            file_type="pdf",
            page_count=page_count,
            errors=[],
            warnings=[],
        )

    @classmethod
    def _validate_image(cls, content: bytes, file_type: str) -> ValidationResult:
        """Validate that the image can be read, opened, and is not corrupted."""
        # Try Pillow (PIL) if installed
        if PIL_Image is not None:
            try:
                with PIL_Image.open(BytesIO(content)) as img:
                    img.verify()
                    # Check format compatibility
                    fmt = img.format.lower() if img.format else ""
                    if file_type == "png" and fmt != "png":
                        return ValidationResult(
                            is_valid=False,
                            file_type=file_type,
                            page_count=None,
                            errors=[f"Content format '{fmt}' does not match expected PNG."],
                        )
                    if file_type == "jpeg" and fmt not in ("jpeg", "jpg"):
                        return ValidationResult(
                            is_valid=False,
                            file_type=file_type,
                            page_count=None,
                            errors=[f"Content format '{fmt}' does not match expected JPEG."],
                        )
                return ValidationResult(
                    is_valid=True,
                    file_type=file_type,
                    page_count=None,
                    errors=[],
                    warnings=[],
                )
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type=file_type,
                    page_count=None,
                    errors=[f"Corrupted or unreadable image: {exc}"],
                )
        else:
            # Fallback pure-Python image structure parser
            try:
                if file_type == "png":
                    cls._validate_png_fallback(content)
                elif file_type == "jpeg":
                    cls._validate_jpeg_fallback(content)
                return ValidationResult(
                    is_valid=True,
                    file_type=file_type,
                    page_count=None,
                    errors=[],
                    warnings=[],
                )
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    file_type=file_type,
                    page_count=None,
                    errors=[f"Corrupted or unreadable image: {exc}"],
                )

    # ---------------------------------------------------------------------------
    # Pure-Python Fallback Decoders
    # ---------------------------------------------------------------------------

    @classmethod
    def _parse_pdf_pages_fallback(cls, data: bytes) -> int:
        """
        Pure-Python ISO 32000 PDF parser fallback.
        Verifies header, trailer EOF, and extracts page count from the catalog tree.
        """
        if len(data) < 32:
            raise ValueError("File is too small to be a valid PDF.")

        # PDF header check (must be within first 1024 bytes)
        if not re.search(rb"^%PDF-[0-9]+\.[0-9]+", data[:1024]):
            raise ValueError("Missing or invalid %PDF- header.")

        # PDF end-of-file marker check (must be within trailing bytes)
        if b"%%EOF" not in data[-2048:]:
            raise ValueError("Missing %%EOF marker; file appears truncated or damaged.")

        # Search for page count inside /Type /Pages dictionaries
        dicts = re.findall(rb"<<([\s\S]*?)>>", data)
        counts = []
        for d in dicts:
            if rb"/Type" in d and rb"/Pages" in d and rb"/Count" in d:
                m = re.search(rb"/Count\s+(\d+)", d)
                if m:
                    counts.append(int(m.group(1)))

        if counts:
            return max(counts)

        # Fallback: Count leaf /Type /Page objects (excluding /Pages)
        page_matches = re.findall(rb"/Type\s*/Page\b", data)
        if page_matches:
            return len(page_matches)

        raise ValueError("Could not locate any valid page objects in the PDF.")

    @classmethod
    def _validate_png_fallback(cls, data: bytes) -> None:
        """
        Pure-Python PNG chunk validator fallback.
        Verifies 8-byte PNG signature, IHDR first chunk, and chunk CRC32 checksums.
        """
        png_sig = b"\x89PNG\r\n\x1a\n"
        if len(data) < 8 or not data.startswith(png_sig):
            raise ValueError("Invalid PNG signature.")

        idx = 8
        chunks = []
        data_len = len(data)

        while idx < data_len:
            if idx + 8 > data_len:
                raise ValueError("Malformed or truncated PNG chunk header.")
            chunk_len = struct.unpack(">I", data[idx : idx + 4])[0]
            chunk_type = data[idx + 4 : idx + 8]
            idx += 8

            if idx + chunk_len + 4 > data_len:
                raise ValueError("Malformed or truncated PNG chunk data.")
            chunk_data = data[idx : idx + chunk_len]
            idx += chunk_len

            expected_crc = struct.unpack(">I", data[idx : idx + 4])[0]
            idx += 4

            actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                raise ValueError(
                    f"Corrupted PNG: CRC mismatch in chunk '{chunk_type.decode('latin1', errors='ignore')}'."
                )

            chunks.append(chunk_type)
            if chunk_type == b"IEND":
                break

        if not chunks or chunks[0] != b"IHDR":
            raise ValueError("Corrupted PNG: Missing initial IHDR header chunk.")

        if b"IEND" not in chunks:
            raise ValueError("Corrupted PNG: Missing terminating IEND chunk.")

    @classmethod
    def _validate_jpeg_fallback(cls, data: bytes) -> None:
        """
        Pure-Python JPEG validator fallback.
        Verifies SOI, segment headers, SOF (Start of Frame) marker, and EOI.
        """
        if len(data) < 4 or data[:2] != b"\xff\xd8":
            raise ValueError("Invalid JPEG: Missing SOI (Start of Image) marker.")

        if b"\xff\xd9" not in data[-128:]:
            raise ValueError("Corrupted JPEG: Missing EOI (End of Image) marker.")

        idx = 2
        data_len = len(data)
        has_sof = False

        while idx < data_len - 1:
            if data[idx] != 0xFF:
                idx += 1
                continue

            marker = data[idx + 1]
            idx += 2

            # Standalone markers with no length
            if marker in (0xD8, 0xD9, 0x00):
                continue

            # Start of Scan: followed by image entropy data until EOI
            if marker == 0xDA:
                break

            if idx + 2 > data_len:
                raise ValueError("Truncated JPEG segment header.")

            seg_len = struct.unpack(">H", data[idx : idx + 2])[0]
            if seg_len < 2 or idx + seg_len > data_len:
                raise ValueError("Malformed JPEG segment length.")

            # SOF0 (Baseline), SOF1 (Extended), SOF2 (Progressive), SOF3 (Lossless)
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                has_sof = True

            idx += seg_len

        if not has_sof:
            raise ValueError("Corrupted JPEG: Missing SOF (Start of Frame) segment.")

    @classmethod
    def _infer_extension(cls, content: bytes) -> str:
        """Infer file extension from initial magic bytes."""
        if content.startswith(b"%PDF-"):
            return ".pdf"
        elif content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        elif len(content) >= 2 and content[:2] == b"\xff\xd8":
            return ".jpg"
        return ""


# Module-level convenience function
validate_document_file = FileValidationService.validate_file
