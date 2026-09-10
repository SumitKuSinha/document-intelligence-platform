from fastapi import FastAPI

app = FastAPI(
    title="Intelligent Document Extraction, Validation & API Platform",
    version="0.1.0",
)


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy"}
