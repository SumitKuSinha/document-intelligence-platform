# Intelligent Document Extraction, Validation & API Platform

An AI engineering case study project designed for intelligent document extraction, structured data validation, and API-driven document processing.

## Project Structure

```text
document-intelligence-platform/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── repositories/
│   │   ├── utils/
│   │   └── main.py
│   │
│   └── tests/
│
├── frontend/
│   ├── templates/
│   └── static/
│       ├── css/
│       └── js/
│
├── docs/
├── sample_outputs/
│
├── .env.example
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- FastAPI & Uvicorn

### Running the API

Navigate to the `backend` directory and run:

```bash
uvicorn app.main:app --reload
```

The health check endpoint will be available at:
`http://localhost:8000/api/v1/health`
