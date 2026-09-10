"""
Documents API Routes

Endpoints for uploading, processing, retrieving, and listing financial documents:
- POST /api/v1/documents/process
- GET /api/v1/documents
- GET /api/v1/documents/{document_name}
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.document import DocumentRepository
from app.schemas.document import (
    DocumentListResponse,
    DocumentProcessResponse,
    DocumentSummaryResponse,
)
from app.services.document_processing import DocumentProcessingService

router = APIRouter()


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and process a financial document",
    description=(
        "Executes the complete document pipeline: file validation, text/OCR extraction, "
        "Gemini financial extraction, deterministic financial validation, and database persistence."
    ),
    responses={
        200: {
            "model": DocumentProcessResponse,
            "description": "Document processed successfully or completed with diagnostic status.",
        },
        400: {
            "model": DocumentProcessResponse,
            "description": "Document failed pre-extraction file validation or document_type was invalid.",
        },
        500: {
            "description": "Unexpected internal error during processing.",
        },
    },
)
async def process_document(
    file: UploadFile = File(..., description="Uploaded document (PDF, PNG, JPG, JPEG)"),
    document_type: str = Form(
        default="invoice",
        description="Category: 'invoice', 'balance_sheet', 'profit_and_loss', or 'cash_flow_statement'",
    ),
    db: Session = Depends(get_db),
) -> JSONResponse:
    """
    Process an uploaded financial document through the full intelligence pipeline.
    """
    filename = file.filename or "unknown"
    doc_type_clean = (document_type or "invoice").strip().lower()

    if doc_type_clean not in DocumentProcessingService.ALLOWED_DOCUMENT_TYPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": (
                    f"Invalid document_type '{document_type}'. "
                    f"Supported types are: {', '.join(sorted(DocumentProcessingService.ALLOWED_DOCUMENT_TYPES))}."
                )
            },
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"Failed to read uploaded file: {exc}"},
        )

    response_data, http_status = DocumentProcessingService.process_document(
        file_bytes=file_bytes,
        filename=filename,
        document_type=doc_type_clean,
        db=db,
    )

    return JSONResponse(
        status_code=http_status,
        content=response_data.model_dump(mode="json"),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List processed financial documents",
    description="Returns a paginated collection of persisted document summaries without raw/binary payloads.",
)
def list_documents(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=100, description="Max documents to return"),
    document_type: Optional[str] = Query(None, description="Optional filter by document type"),
    processing_status: Optional[str] = Query(None, description="Optional filter by status"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """List persisted financial documents with summary metadata."""
    repo = DocumentRepository(db)
    docs = repo.list_documents(
        skip=skip,
        limit=limit,
        document_type=document_type,
        processing_status=processing_status,
    )

    summaries = [
        DocumentSummaryResponse(
            id=d.id,
            document_name=d.document_name,
            document_type=d.document_type,
            processing_status=d.processing_status,
            created_at=d.created_at,
            metadata=d.doc_metadata,
        )
        for d in docs
    ]
    return DocumentListResponse(total=len(summaries), documents=summaries)


@router.get(
    "/{document_name}",
    response_model=DocumentProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Get processed document details by filename",
    description="Retrieves the latest persisted extraction, validation, and metadata result for a document name.",
)
def get_document_by_name(
    document_name: str,
    db: Session = Depends(get_db),
) -> DocumentProcessResponse:
    """Retrieve full structured processing result for a document by name."""
    repo = DocumentRepository(db)
    doc = repo.get_by_name(document_name)

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_name}' not found.",
        )

    return DocumentProcessResponse(
        id=doc.id,
        document_name=doc.document_name,
        document_type=doc.document_type,
        processing_status=doc.processing_status,
        file_validation=doc.file_validation,
        extracted_data=doc.extracted_data,
        validations=doc.validations,
        metadata=doc.doc_metadata,
        created_at=doc.created_at,
    )
