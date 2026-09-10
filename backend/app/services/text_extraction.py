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
                            image_bytes=None,
                            mime_type=None,
                            is_scanned=False,
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
                    primary_image = page_data.images[0] if page_data.images else None
                    mime = "image/jpeg"
                    if primary_image:
                        if primary_image.startswith(b"\x89PNG"):
                            mime = "image/png"
                        elif primary_image.startswith(b"\xff\xd8\xff"):
                            mime = "image/jpeg"

                    pages.append(
                        ExtractedPage(
                            page_number=page_data.page_number,
                            text=merged_ocr_text,
                            image_bytes=primary_image,
                            mime_type=mime,
                            is_scanned=True,
                        )
                    )
                    warnings.append(
                        f"Page {page_data.page_number} contained no digital text; extracted via OCR."
                    )
                else:
                    # Page has no usable digital text and no embedded images.
                    # Attempt robust page rasterization / rendering fallback (e.g. for vector-drawn PDFs).
                    rendered_bytes = None
                    try:
                        rendered_bytes = PDFTextService.render_page(content, page_data.page_number)
                    except Exception as render_exc:
                        warnings.append(
                            f"Page {page_data.page_number} rasterization attempt failed: {render_exc}"
                        )

                    if rendered_bytes:
                        # Successfully rendered page; send through existing OCR/Vision pipeline
                        extracted_ocr = ""
                        try:
                            extracted_ocr = OCRService.extract_text_from_image(rendered_bytes)
                        except OCRError as ocr_err:
                            errors.append(
                                f"OCR failed for rendered page {page_data.page_number}: {ocr_err}"
                            )

                        pages.append(
                            ExtractedPage(
                                page_number=page_data.page_number,
                                text=extracted_ocr,
                                image_bytes=rendered_bytes,
                                mime_type="image/png",
                                is_scanned=True,
                            )
                        )
                        warnings.append(
                            f"Page {page_data.page_number} contained no digital text or embedded images; "
                            f"rasterized page image for OCR/Vision extraction."
                        )
                    else:
                        # Blank page or non-renderable content
                        pages.append(
                            ExtractedPage(
                                page_number=page_data.page_number,
                                text="",
                                image_bytes=None,
                                mime_type=None,
                                is_scanned=False,
                            )
                        )
                        warnings.append(
                            f"Page {page_data.page_number} contains no extractable digital text, images, or renderable content."
                        )

        # 4. Process raster image documents (PNG, JPG/JPEG)
        elif file_type in ("png", "jpeg"):
            try:
                ocr_text = OCRService.extract_text_from_image(content)
                mime = "image/png" if file_type == "png" else "image/jpeg"
                pages.append(
                    ExtractedPage(
                        page_number=1,
                        text=ocr_text,
                        image_bytes=content,
                        mime_type=mime,
                        is_scanned=True,
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
