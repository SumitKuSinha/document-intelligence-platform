"""
PDF Text Extraction Service

Extracts digital text and extracts embedded raster images page-by-page from PDF files using pypdf.
"""

from dataclasses import dataclass, field
from io import BytesIO
from typing import List

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
    Service for extracting digital text and embedded images from PDF documents.
    """

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
