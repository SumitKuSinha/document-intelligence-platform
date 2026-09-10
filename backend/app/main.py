from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.documents import router as documents_router

app = FastAPI(
    title="Intelligent Document Extraction, Validation & API Platform",
    version="0.1.0",
)

# Enable CORS for web browsers and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# Mount frontend static assets and UI dashboard if present
BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
INDEX_FILE = FRONTEND_DIR / "templates" / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_dashboard():
    if INDEX_FILE.exists():
        return FileResponse(str(INDEX_FILE))
    return {"message": "Intelligent Document Extraction Platform API is running. Frontend not found."}
