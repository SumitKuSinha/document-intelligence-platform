"""
PDF Text Extraction Service

Extracts digital text and embedded raster images page-by-page from PDF files using pypdf,
with a robust PyMuPDF (fitz) page rasterization fallback for pages containing vector graphics,
flattened drawings, or non-extractable content.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional

import pypdf


@dataclass
class PDFPageData:
    """
    Extracted data from a single PDF page.

    Attributes:
        page_number: 1-indexed page number.
        text: Digital text extracted from the PDF page.
        images: List of raw byte contents of embedded images on this page.
    """

    page_number: int
    text: str
    images: List[bytes] = field(default_factory=list)


class PDFTextService:
    """
    Service for extracting digital text and embedded images from PDF documents,
    with page rasterization fallback for vector-based or non-extractable pages.
    """

    DEFAULT_RENDER_DPI: int = 200

    @classmethod
    def render_page(
        cls,
        content: bytes,
        page_number: int,
        dpi: int = DEFAULT_RENDER_DPI,
    ) -> Optional[bytes]:
        """
        Rasterize/render a single PDF page to a PNG image using PyMuPDF.

        Args:
            content: Raw binary content of the PDF file.
            page_number: 1-indexed page number.
            dpi: Resolution for rasterization (default: 200 DPI).

        Returns:
            Optional[bytes]: PNG image bytes, or None if rendering fails.
        """
        try:
            try:
                import pymupdf
            except ImportError:
                import fitz as pymupdf

            doc = pymupdf.open(stream=content, filetype="pdf")
            if page_number < 1 or page_number > len(doc):
                return None

            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        except Exception:
            return None

    @classmethod
    def extract_pages(cls, content: bytes) -> List[PDFPageData]:
        """
        Extract digital text and embedded images from all pages of a PDF document.

        Args:
            content: Raw binary content of the PDF file.

        Returns:
            List[PDFPageData]: Extracted data for each page in sequence.

        Raises:
            Exception: If the PDF stream cannot be parsed.
        """
        reader = pypdf.PdfReader(BytesIO(content))
        extracted_pages: List[PDFPageData] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1

            # 1. Extract digital text (if available)
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            # 2. Extract embedded images (for scanned/hybrid pages)
            page_images: List[bytes] = []
            try:
                for img_file in page.images:
                    if hasattr(img_file, "data") and img_file.data:
                        page_images.append(img_file.data)
            except Exception:
                # Some malformed image XObjects can raise exceptions; continue gracefully
                pass

            extracted_pages.append(
                PDFPageData(
                    page_number=page_num,
                    text=page_text,
                    images=page_images,
                )
            )

        return extracted_pages
