"""
OCR Service

Handles Optical Character Recognition (OCR) for raster images and scanned document pages
using Tesseract (via pytesseract) as the primary OCR engine.
"""

from io import BytesIO
import os
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

from dotenv import load_dotenv
from PIL import Image

# Ensure root .env variables (such as TESSERACT_CMD) are loaded
ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

try:
    import pytesseract
except ImportError:
    pytesseract = None


class OCRError(Exception):
    """Exception raised when an OCR operation fails or the engine is unavailable."""

    pass


class OCRService:
    """
    Service responsible for Optical Character Recognition.

    Uses Tesseract (via pytesseract) as its primary engine.
    Supports dependency injection for mock or alternative OCR engines in test environments.
    """

    _custom_engine: Optional[Callable[[Image.Image], str]] = None

    @classmethod
    def set_ocr_engine(cls, engine_fn: Optional[Callable[[Image.Image], str]]) -> None:
        """
        Register a custom or mock OCR callable (useful for unit tests or alternative engines).

        Args:
            engine_fn: Callable taking a PIL.Image and returning extracted string text,
                       or None to restore default Tesseract behavior.
        """
        cls._custom_engine = engine_fn

    @classmethod
    def _configure_tesseract(cls) -> None:
        """Configure Tesseract binary path if specified via environment variable."""
        if pytesseract is None:
            return

        tesseract_cmd = os.getenv("TESSERACT_CMD") or os.getenv("TESSERACT_PATH")
        if tesseract_cmd:
            clean_cmd = tesseract_cmd.strip('"\'')
            pytesseract.pytesseract.tesseract_cmd = clean_cmd

    @classmethod
    def is_tesseract_available(cls) -> bool:
        """Check if Tesseract OCR is installed and accessible in the system PATH or configured path."""
        if cls._custom_engine is not None:
            return True

        if pytesseract is None:
            return False

        cls._configure_tesseract()
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @classmethod
    def extract_text_from_image(
        cls,
        image_input: Union[bytes, Image.Image, BinaryIO, Path, str],
    ) -> str:
        """
        Extract text from an image using the configured OCR engine.

        Args:
            image_input: Raw image bytes, PIL Image object, stream, or file path.

        Returns:
            str: Raw plain text extracted by the OCR engine.

        Raises:
            OCRError: If the image cannot be decoded, Tesseract is not found, or OCR execution fails.
        """
        # 1. Custom/Mock OCR engine takes precedence if registered (e.g. in test suites)
        pil_image: Optional[Image.Image] = None

        try:
            if isinstance(image_input, Image.Image):
                pil_image = image_input
            elif isinstance(image_input, (str, Path)):
                pil_image = Image.open(str(image_input))
            elif isinstance(image_input, bytes):
                pil_image = Image.open(BytesIO(image_input))
            elif hasattr(image_input, "read"):
                raw_bytes = image_input.read()
                if hasattr(image_input, "seek"):
                    image_input.seek(0)
                pil_image = Image.open(BytesIO(raw_bytes))
            else:
                raise OCRError(f"Unsupported image input type: {type(image_input)}")

            # Ensure image is in RGB or Grayscale mode for OCR
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")

        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"Failed to open image for OCR processing: {exc}") from exc

        # 2. Check for injected custom engine
        if cls._custom_engine is not None:
            try:
                return cls._custom_engine(pil_image)
            except Exception as exc:
                raise OCRError(f"Custom OCR engine failure: {exc}") from exc

        # 3. Default to pytesseract
        if pytesseract is None:
            raise OCRError(
                "pytesseract package is not installed. Please install it with 'pip install pytesseract'."
            )

        cls._configure_tesseract()

        try:
            raw_text = pytesseract.image_to_string(pil_image)
            return raw_text.strip()
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRError(
                "Tesseract OCR executable not found. Please install Tesseract-OCR on your system "
                "or specify its location via the TESSERACT_CMD environment variable."
            ) from exc
        except Exception as exc:
            raise OCRError(f"Tesseract OCR processing failed: {exc}") from exc
