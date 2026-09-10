"""
Text Extraction Service

Orchestrates the complete text extraction and OCR pipeline:
1. Reuses FileValidationService to ensure document validity, supported format, and page limits.
2. For digital/text-based PDFs: extracts text page-by-page, preserving page numbers.
3. For scanned/image-only PDFs: identifies pages without digital text and executes OCR on page images.
4. For image documents (PNG, JPG/JPEG): routes images directly through the OCR engine.
5. Returns a structured Pydantic TextExtractionResult containing status, pages, errors, and warnings.
"""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

from app.schemas.text_extraction import ExtractedPage, TextExtractionResult
from app.services.file_validation import FileValidationService
from app.services.ocr_service import OCRError, OCRService
from app.services.pdf_text_service import PDFPageData, PDFTextService


class TextExtractionService:
    """
    Production-ready service for extracting textual content from financial documents.
    """

    @classmethod
    def extract_text(
        cls,
        file_input: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None,
    ) -> TextExtractionResult:
        """
        Validate an uploaded document and extract its text content page-by-page.

        Args:
            file_input: File path, raw bytes, or binary stream.
            filename: Original filename (recommended for stream or raw bytes).

        Returns:
            TextExtractionResult: Structured extraction outcome with page-level text,
                                 lifecycle status, errors, and warnings.
        """
        # 1. Reuse existing FileValidationService
        validation = FileValidationService.validate_file(
            file_input=file_input,
            filename=filename,
        )

        if not validation.is_valid:
            return TextExtractionResult(
                extraction_status="FAILED",
                pages=[],
                errors=validation.errors,
                warnings=validation.warnings,
            )

        # 2. Extract raw content bytes
        content: bytes = b""
        if isinstance(file_input, (str, Path)):
            content = Path(file_input).read_bytes()
        elif isinstance(file_input, bytes):
            content = file_input
        elif hasattr(file_input, "read"):
            content = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(0)

        file_type = validation.file_type
        pages: List[ExtractedPage] = []
        errors: List[str] = []
        warnings: List[str] = list(validation.warnings)

        # 3. Process PDF documents
        if file_type == "pdf":
            try:
                pdf_pages: List[PDFPageData] = PDFTextService.extract_pages(content)
            except Exception as exc:
                return TextExtractionResult(
                    extraction_status="FAILED",
                    pages=[],
                    errors=[f"Failed to read PDF document structure: {exc}"],
                    warnings=warnings,
                )

            for page_data in pdf_pages:
                raw_page_text = page_data.text.strip()

                if raw_page_text:
                    # Page has native digital text; preserve verbatim
                    pages.append(
                        ExtractedPage(
                            page_number=page_data.page_number,
                            text=page_data.text,
                        )
                    )
                elif page_data.images:
                    # Scanned / image-only page: route embedded raster images to OCR
                    ocr_results: List[str] = []
                    for img_bytes in page_data.images:
                        try:
                            extracted_ocr = OCRService.extract_text_from_image(img_bytes)
                            if extracted_ocr:
                                ocr_results.append(extracted_ocr)
                        except OCRError as ocr_err:
                            errors.append(
                                f"OCR failed for image on page {page_data.page_number}: {ocr_err}"
                            )

                    merged_ocr_text = "\n\n".join(ocr_results)
                    pages.append(
                        ExtractedPage(
                            page_number=page_data.page_number,
                            text=merged_ocr_text,
                        )
                    )
                    warnings.append(
                        f"Page {page_data.page_number} contained no digital text; extracted via OCR."
                    )
                else:
                    # Blank page or non-extractable content
                    pages.append(
                        ExtractedPage(
                            page_number=page_data.page_number,
                            text="",
                        )
                    )
                    warnings.append(
                        f"Page {page_data.page_number} contains no extractable digital text or images."
                    )

        # 4. Process raster image documents (PNG, JPG/JPEG)
        elif file_type in ("png", "jpeg"):
            try:
                ocr_text = OCRService.extract_text_from_image(content)
                pages.append(
                    ExtractedPage(
                        page_number=1,
                        text=ocr_text,
                    )
                )
            except OCRError as exc:
                errors.append(f"OCR processing failed for {file_type.upper()} image: {exc}")
            except Exception as exc:
                errors.append(f"Unexpected error during image OCR: {exc}")

        # 5. Determine overall extraction lifecycle status
        if errors and not pages:
            status = "FAILED"
        elif errors and pages:
            # Check if any page produced text
            has_any_text = any(bool(p.text.strip()) for p in pages)
            status = "PARTIAL_SUCCESS" if has_any_text else "FAILED"
        else:
            status = "SUCCESS"

        return TextExtractionResult(
            extraction_status=status,
            pages=pages,
            errors=errors,
            warnings=warnings,
        )


# Module-level convenience function
extract_document_text = TextExtractionService.extract_text
