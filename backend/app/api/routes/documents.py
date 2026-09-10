"""
Documents API Routes

Endpoints for uploading and processing documents.
"""

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.document import DocumentProcessResponse
from app.services.file_validation import FileValidationService

router = APIRouter()


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and validate a document",
    description="Validates an uploaded financial document (PDF, PNG, JPG/JPEG) prior to downstream processing.",
    responses={
        200: {
            "model": DocumentProcessResponse,
            "description": "Document passed file validation successfully.",
        },
        400: {
            "model": DocumentProcessResponse,
            "description": "Document failed file validation (e.g., unsupported format, empty file, exceeds 3 pages, corrupted).",
        },
    },
)
async def process_document(
    file: UploadFile = File(..., description="Uploaded financial document (PDF, PNG, JPG, JPEG)"),
) -> JSONResponse:
    """
    Process an uploaded document by executing pre-extraction file validation.

    Passes the uploaded file stream and filename to the FileValidationService
    and returns a structured validation outcome.
    """
    filename = file.filename or "unknown"

    # Pass the underlying file stream and filename to FileValidationService
    validation_result = FileValidationService.validate_file(
        file_input=file.file,
        filename=filename,
    )

    if validation_result.is_valid:
        response_data = DocumentProcessResponse(
            document_name=filename,
            processing_status="VALIDATED",
            file_validation=validation_result,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data.model_dump(),
        )

    # For invalid files, return HTTP 400 Bad Request preserving structured validation details
    response_data = DocumentProcessResponse(
        document_name=filename,
        processing_status="VALIDATION_FAILED",
        file_validation=validation_result,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response_data.model_dump(),
    )
