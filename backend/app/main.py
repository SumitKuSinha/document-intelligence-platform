from fastapi import FastAPI

from app.api.routes.documents import router as documents_router

app = FastAPI(
    title="Intelligent Document Extraction, Validation & API Platform",
    version="0.1.0",
)

# Register API routes
app.include_router(
    documents_router,
    prefix="/api/v1/documents",
    tags=["documents"],
)


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy"}
