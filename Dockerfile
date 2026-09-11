# Production Dockerfile for Intelligent Document Intelligence Platform
FROM python:3.12-slim

# Set environment variables for Python, Tesseract, and application runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TESSERACT_CMD=/usr/bin/tesseract \
    PYTHONPATH=/app/backend

# Install system dependencies including Tesseract OCR engine and English language pack
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set root application directory inside the container
WORKDIR /app

# Copy requirements and install Python dependencies (cached layer)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./backend/requirements.txt

# Copy application source code:
# Preserves /app/backend and /app/frontend structure required by main.py
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Set working directory to backend so uvicorn discovers app.main:app directly
WORKDIR /app/backend

# Expose default application port
EXPOSE 8000

# Start FastAPI application using the port assigned by Render ($PORT)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
