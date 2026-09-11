# Intelligent Document Extraction, Validation & API Platform

An enterprise-grade, production-ready document intelligence platform designed for automated financial document ingestion, optical character recognition (OCR), structured Large Language Model (LLM) information extraction, deterministic rule-based financial validation, and persistent audit trail tracking.

---

## 1. Project Overview

Financial institutions, accounting departments, and enterprise ERP systems process thousands of semi-structured and unstructured documents daily, including tax invoices, commercial receipts, consolidated balance sheets, income statements, and cash flow reports. Traditional template-based extractors fail when document layouts vary, while unconstrained LLM extractors risk hallucinations and arithmetic errors.

This platform solves these challenges through a multi-stage, defensive pipeline:
1. **File Security & Pre-Processing Validation**: Validates file integrity, magic bytes, MIME types, and size constraints before processing.
2. **Hybrid Text Extraction & OCR Engine**: Extracts page-by-page digital text natively via `pypdf`, falling back seamlessly to Tesseract OCR (`pytesseract`) for scanned documents and raster images.
3. **Semantic Document Type Reconciliation**: Inspects document header semantics to automatically correct client classification errors (e.g., balance sheets erroneously submitted as invoices) without relying on fragile filename heuristics.
4. **Structured LLM Extraction with Multi-Tier Fallback**: Leverages Google Gemini Flash models (`google-genai` SDK) conforming to strict Pydantic schemas with full audit-trail source evidence snippets and page citations. Automatically cascades through a resilient fallback ladder upon quota exhaustion (HTTP 429).
5. **Deterministic Financial Validation Engine**: Reconciles line-item arithmetic, component summations, tax reconciliations, and accounting equations with strict document-type isolation and configurable rounding tolerances (`0.01`).
6. **Persistence & Interactive Dashboard**: Stores immutable extraction and validation records in PostgreSQL (SQLAlchemy 2.0 / Alembic), served via a high-performance FastAPI REST API and an interactive real-time browser dashboard.

---

## 2. System Architecture

```mermaid
flowchart TD
    A["Uploaded File (PDF, PNG, JPG, JPEG)"] --> B["FileValidationService\n(Magic Bytes, MIME, Max 10MB)"]
    B -->|Valid| C["TextExtractionService\n(Hybrid Pipeline)"]
    B -->|Invalid| ERR1["HTTP 400 Bad Request\n(Diagnostic Failure Payload)"]
    
    C --> D{"Digital Text Available?"}
    D -->|Yes| E["pypdf Digital Extractor\n(Page-by-page text & layout)"]
    D -->|No / Scanned| F["Tesseract OCR Engine\n(pytesseract & PIL)"]
    
    E --> G["Document Type Reconciliation\n(Inspects Statement Headers)"]
    F --> G
    
    G --> H["FinancialExtractionService\n(Structured Prompt & Schemas)"]
    H --> I{"Google Gemini LLM Engine\n(Default: gemini-3.6-flash)"}
    
    I -->|HTTP 429 / Quota Limit| J["Resilient Fallback Ladder\n(3.7-flash -> 3.8-flash -> 3.5-flash -> 3-flash -> 3.5-lite -> 3.1-lite)"]
    I -->|Success| K["Structured Extraction Payload\n+ FieldEvidence Snippets & Pages"]
    J -->|Success| K
    
    K --> L["FinancialValidationService\n(Strict Document-Type Isolation)"]
    
    L --> M1["Invoice Validator\n- Line-item math (discounts/taxes)\n- Subtotal reconciliation\n- Statutory tax components\n- Grand total + round-off"]
    L --> M2["Balance Sheet Validator\n- Accounting Equation (Assets = Cap + Liab)\n- Asset components sum\n- Liability & Equity components sum\n- Footnote exclusion"]
    L --> M3["Profit & Loss Validator\n- Gross Profit = Revenue - COGS\n- Operating Income = GP - OpEx\n- Net Income reconciliation"]
    L --> M4["Cash Flow Validator\n- Net Change = Operating + Investing + Financing"]
    
    M1 --> N["Validation Report Card\n(PASS / FAILED / NOT_APPLICABLE)"]
    M2 --> N
    M3 --> N
    M4 --> N
    
    N --> O["PostgreSQL Database\n(SQLAlchemy 2.0 / Alembic Migrations)"]
    O --> P["FastAPI REST API & Swagger UI\n(/api/v1/documents, /docs)"]
    O --> Q["Interactive Frontend UI\n(Vanilla JS / Responsive Dashboard)"]
```

---

## 3. Tech Stack & Engineering Rationale

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language Runtime** | Python 3.12 | Modern type-hinting ergonomics, native performance improvements, and ecosystem compatibility with modern data libraries. |
| **Backend Framework** | FastAPI (0.141+) | High-performance asynchronous ASGI framework with automated OpenAPI 3.0 (Swagger) generation and native Pydantic validation. |
| **Data Validation** | Pydantic v2 (2.13+) | High-speed Rust-based serialization, strict schema enforcement, and flexible field alias mapping (`AliasChoices`) for handling varied financial terminology. |
| **Database & ORM** | PostgreSQL & SQLAlchemy 2.0 | Enterprise ACID compliance, robust connection pooling (`pool_pre_ping=True`), and native JSONB columns for flexible, indexed storage of extraction schemas. |
| **DB Driver** | `psycopg` 3 (3.3+) | Modern Python 3 async-capable PostgreSQL driver replacing legacy `psycopg2`. |
| **Migrations** | Alembic (1.14+) | Version-controlled, reproducible relational database schema evolution. |
| **PDF Extraction** | `pypdf` (5.0+) | Pure-Python digital text and embedded image extraction preserving page-level indexing without external C++ runtimes. |
| **OCR Engine** | Tesseract OCR & `pytesseract` | Industry-standard optical character recognition for digitized and scanned paper invoices and financial statements. |
| **LLM Engine** | Google Gemini (`google-genai` SDK) | Gemini 2.5/3 Flash series offers industry-leading structured JSON output generation, high context windows, low latency, and deterministic parameterization. |
| **Frontend UI** | HTML5, CSS3, Vanilla JS (ES6+) | Lightweight, dependency-free single-page dashboard. Zero npm build step, instant rendering, glassmorphic layout, and interactive validation inspection. |
| **Containerization** | Docker (`python:3.12-slim`) | Self-contained production image bundling Linux Tesseract OCR binaries, system language packs, and dynamic port binding for Render. |

---

## 4. Supported Document Types

The platform processes four core classes of financial documents:

1. **Tax & Commercial Invoices (`invoice`)**:
   - Vendor details, customer details, invoice numbers, dates, payment terms.
   - Itemized line items (quantities, unit prices, line discounts, tax rates, total amounts).
   - Taxable subtotals, statutory tax breakdowns (CGST, SGST, IGST, VAT), shipping charges, explicit round-off adjustments, and grand totals.
2. **Balance Sheets (`balance_sheet`)**:
   - Corporate balance sheets: Current assets, non-current assets, current liabilities, long-term liabilities, share capital, retained earnings.
   - Banking / Statutory balance sheets: Capital, reserves & surplus, deposits, borrowings, cash & balances with central banks, investments, advances, fixed assets, and off-balance sheet footnote isolation (contingent liabilities, bills for collection).
3. **Profit & Loss Statements (`profit_and_loss`)**:
   - Revenue, cost of goods sold (COGS), gross profit, operating expenses (R&D, SG&A), operating income, tax expenses, and net profit.
4. **Cash Flow Statements (`cash_flow_statement`)**:
   - Cash flows from operating activities, investing activities, and financing activities, beginning cash balances, ending cash balances, and net change in cash.

---

## 5. Pipeline Stages

### 5.1 Pre-Processing & File Security Validation
Implemented in [`FileValidationService`](backend/app/services/file_validation.py):
- **Magic Byte Verification**: Inspects the leading binary bytes of the upload to prevent extension spoofing:
  - PDF: `%PDF` (`\x25\x50\x44\x46`)
  - PNG: `\x89PNG\r\n\x1a\n` (`\x89\x50\x4e\x47\x0d\x0a\x1a\x0a`)
  - JPEG/JPG: `\xff\xd8\xff`
- **File Size Boundaries**: Enforces strict boundaries (default: Minimum 1 byte, Maximum 10 MB).
- **MIME & Extension Whitelist**: Only permits `.pdf`, `.png`, `.jpg`, and `.jpeg`.
- Spoofed, empty, or oversized files are rejected immediately with descriptive diagnostic logs before invoking OCR or LLM endpoints.

### 5.2 Hybrid Text Extraction & OCR
Implemented in [`TextExtractionService`](backend/app/services/text_extraction.py):
- **Digital PDF Extraction**: Uses `pypdf` to extract digital text page-by-page. Preserves page boundaries and layout without synthetically altering characters.
- **Scanned / Raster Fallback**: Scans pages containing images or pure image uploads (`.png`, `.jpg`, `.jpeg`) using Tesseract OCR via `pytesseract`.
- **Verbatim Text Integrity**: Extracted text preserves line breaks and tabular spacing to maximize LLM extraction accuracy.

### 5.3 Semantic Document Type Reconciliation
Implemented in [`DocumentProcessingService.reconcile_document_type`](backend/app/services/document_processing.py):
- When users upload a document without altering the default client dropdown (`document_type=invoice`), the service scans the initial 4,000 characters of extracted text for statutory statement headers:
  - `"BALANCE SHEET"`, `"CONSOLIDATED BALANCE SHEET"`, `"STATEMENT OF ASSETS AND LIABILITIES"` $\to$ `balance_sheet`
  - `"PROFIT AND LOSS"`, `"INCOME STATEMENT"`, `"STATEMENT OF OPERATIONS"` $\to$ `profit_and_loss`
  - `"CASH FLOW STATEMENT"`, `"STATEMENT OF CASH FLOWS"` $\to$ `cash_flow_statement`
- Eliminates cross-contamination without relying on arbitrary filenames.

### 5.4 Structured LLM Extraction & Resilient Fallback Ladder
Implemented in [`FinancialExtractionService`](backend/app/services/financial_extraction.py) and [`LLMClient`](backend/app/services/llm_client.py):
- Prompts Google Gemini with typed Pydantic JSON schemas, explicit extraction rules, and deterministic temperature (`0.0`).
- **Resilient Fallback Ladder**:
  To protect against rate limits and daily quota limits on Google Gemini free/tier accounts, `LLMClient` implements an automated multi-tier cascade:
  1. **Primary Model**: Configured `LLM_MODEL` (default: `gemini-3.6-flash`).
  2. **Regular Flash Pool Cascade**: Upon detecting HTTP 429 (`RESOURCE_EXHAUSTED` / `generaterequestsperday`), cascades immediately through:
     $$\text{gemini-3.6-flash} \to \text{gemini-3.7-flash} \to \text{gemini-3.8-flash} \to \text{gemini-3.5-flash} \to \text{gemini-3-flash}$$
  3. **Flash Lite Fallback Pool**: If all regular Flash models are exhausted, cascades to lightweight fallback models:
     $$\text{gemini-3.5-flash-lite} \to \text{gemini-3.1-flash-lite}$$
  4. **Transient Error Handling**: Transient HTTP 503 or connection errors retry automatically with exponential backoff and jitter before falling back.
  5. **Credential Redaction**: Redacts API keys (`GEMINI_API_KEY`, `GOOGLE_API_KEY`) and bearer tokens from all logs and error traces.

### 5.5 Evidence & Audit Trail Tracking
Every extracted data field incorporates a [`FieldEvidence`](backend/app/schemas/financial_extraction.py) audit structure:
```json
{
  "extracted_value": 6862.00,
  "source_snippet": "Total Invoice Value (In Figures) : 6,862.00",
  "page_number": 1,
  "confidence": 0.98
}
```
This enables human reviewers to cross-reference every extracted amount against the original document.

---

## 6. Deterministic Financial Validation Engine

Implemented in [`FinancialValidationService`](backend/app/services/financial_validation.py).

All arithmetic checks use a strict rounding tolerance (default `0.01`, configurable via `FINANCIAL_VALIDATION_TOLERANCE`). Rules strictly adhere to document-type boundaries:

### 6.1 Invoices
1. **Line-Item Arithmetic (`invoice_line_item_X_math`)**:
   - **Undiscounted lines**: $\text{quantity} \times \text{unit\_price} \approx \text{total\_price}$
   - **Percentage discounts**: $\text{quantity} \times \text{unit\_price} \times (1 - \frac{\text{discount\_percent}}{100}) \approx \text{total\_price}$
   - **Monetary discounts**: $(\text{quantity} \times \text{unit\_price}) - \text{discount\_amount} \approx \text{total\_price}$
   - Never invents ungrounded discounts.
2. **Subtotal Reconciliation (`invoice_subtotal_from_line_items`)**:
   $$\sum \text{line\_items.total\_price} \approx \text{subtotal}$$
3. **Statutory Tax Reconciliation (`invoice_tax_reconciliation`)**:
   - Verifies $\text{subtotal} \times \text{tax\_rate} \approx \text{tax\_amount}$ or reconciles statutory tax components ($\sum \text{tax\_components} = \text{CGST} + \text{SGST} + \text{IGST} \approx \text{tax\_amount}$).
4. **Grand Total Reconciliation (`invoice_total_reconciliation`)**:
   $$\text{subtotal} + \text{tax\_amount} - \text{discount\_amount} + \text{shipping\_amount} + \text{round\_off\_amount} \approx \text{total\_amount}$$
   Includes source-reported round-off adjustments without inventing values.

### 6.2 Balance Sheets
1. **Accounting Equation (`balance_sheet_accounting_equation`)**:
   $$\text{Total Assets} \approx \text{Total Capital and Liabilities} \quad \text{or} \quad \text{Total Assets} \approx \text{Total Liabilities} + \text{Total Equity}$$
2. **Asset Components Reconciliation (`balance_sheet_total_assets_components`)**:
   $$\sum \text{Reported On-Balance Sheet Asset Line Items} \approx \text{Total Assets}$$
3. **Liabilities & Equity Reconciliation (`balance_sheet_total_liabilities_and_equity_components`)**:
   $$\sum \text{Reported Liability \& Capital Line Items} \approx \text{Total Capital and Liabilities}$$
4. **Footnote / Off-Balance Sheet Isolation**:
   Statutory off-balance-sheet items (e.g. *Contingent liabilities*, *Bills for collection*) are excluded from balance sheet totals.
5. **Missing Components**:
   If an optional breakdown equation lacks reported inputs, it evaluates to `NOT_APPLICABLE` rather than assuming zero or marking `FAILED`.

### 6.3 Profit & Loss Statements
1. **Gross Profit**: $\text{revenue} - \text{cost\_of\_goods\_sold} \approx \text{gross\_profit}$
2. **Operating Income**: $\text{gross\_profit} - \text{operating\_expenses} \approx \text{operating\_income}$
3. **Net Income**: $\text{operating\_income} - \text{tax\_expense} - \text{other\_expenses} \approx \text{net\_income}$

### 6.4 Cash Flow Statements
1. **Net Change in Cash**:
   $$\text{operating\_cash\_flow} + \text{investing\_cash\_flow} + \text{financing\_cash\_flow} \approx \text{net\_change\_in\_cash}$$

---

## 7. Database Architecture & Persistence

Managed with PostgreSQL, SQLAlchemy 2.0, and Alembic:

### Schema: `documents` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Autoincrement | Unique document identifier. |
| `document_name` | `VARCHAR(255)` | Indexed, Not Null | Original filename of uploaded document. |
| `document_type` | `VARCHAR(100)` | Not Null | Document classification category. |
| `processing_status` | `VARCHAR(50)` | Not Null, Default `'PENDING'` | Lifecycle status (`PENDING`, `COMPLETED`, `FAILED`). |
| `file_validation` | `JSONB` | Nullable | MIME type, magic bytes, file size diagnostics. |
| `extracted_data` | `JSONB` | Nullable | Complete structured financial extraction payload. |
| `validations` | `JSONB` | Nullable | Array of validation checks, formulas, inputs, variances, and summary. |
| `metadata` | `JSONB` | Nullable | OCR confidence, page counts, model metadata, latency. |
| `created_at` | `TIMESTAMP WITH TZ` | Server Default `NOW()` | Audit record creation timestamp. |

---

## 8. REST API Reference

All endpoints are prefixed with `/api/v1`. Interactive Swagger documentation is available at `/docs`.

| Method | Endpoint | Request Body / Params | Status Codes | Description |
| :--- | :--- | :--- | :--- | :--- |
| **POST** | `/api/v1/documents/process` | `multipart/form-data`<br>- `file`: Binary file<br>- `document_type`: string (optional) | `200 OK`<br>`400 Bad Request`<br>`500 Internal Error` | Ingests, validates, extracts, verifies, and persists document. |
| **GET** | `/api/v1/documents` | Query Params:<br>- `skip`: int (default 0)<br>- `limit`: int (default 50)<br>- `document_type`: string (optional)<br>- `processing_status`: string (optional) | `200 OK` | Returns paginated list of document summaries. |
| **GET** | `/api/v1/documents/{document_name}` | Path Param:<br>- `document_name`: string | `200 OK`<br>`404 Not Found` | Retrieves full extraction and validation report for a document. |
| **GET** | `/api/v1/health` | None | `200 OK` | Platform healthcheck endpoint (`{"status": "healthy"}`). |
| **GET** | `/` | None | `200 OK` | Serves interactive web dashboard UI. |
| **GET** | `/docs` | None | `200 OK` | Interactive OpenAPI Swagger UI documentation. |
| **GET** | `/redoc` | None | `200 OK` | Alternative ReDoc API documentation. |

---

## 9. Sample API Request & Response

### Request
```bash
curl -X POST "http://localhost:8000/api/v1/documents/process" \
  -H "Accept: application/json" \
  -F "file=@sample_outputs/Invoices/batch1-1109.jpg" \
  -F "document_type=invoice"
```

### Response (`200 OK`)
```json
{
  "id": 64,
  "document_name": "batch1-1109.jpg",
  "document_type": "invoice",
  "processing_status": "COMPLETED",
  "file_validation": {
    "is_valid": true,
    "file_size_bytes": 217744,
    "detected_mime_type": "image/jpeg",
    "magic_bytes_valid": true,
    "errors": [],
    "warnings": []
  },
  "extracted_data": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-03-15",
    "subtotal": 5815.17,
    "tax_amount": 1046.72,
    "round_off_amount": 0.11,
    "total_amount": 6862.00,
    "line_items": [
      {
        "description": "Industrial Sensor Module",
        "quantity": 6.0,
        "unit_price": 7.44,
        "discount_percent": 99.0,
        "total_price": 0.45
      }
    ]
  },
  "validations": {
    "checks": [
      {
        "rule_name": "invoice_line_item_1_math",
        "formula": "quantity * unit_price * (1 - discount_percent / 100) == total_price",
        "input_values": {
          "quantity": 6.0,
          "unit_price": 7.44,
          "discount_percent": 99.0,
          "total_price": 0.45
        },
        "calculated_value": 0.45,
        "reported_value": 0.45,
        "variance": 0.0,
        "status": "PASS",
        "tolerance": 0.01,
        "message": "Line item arithmetic passed with explicit percentage discount (6.0 * 7.44 * (1 - 99.0%) = 0.45 == 0.45)."
      },
      {
        "rule_name": "invoice_total_reconciliation",
        "formula": "subtotal + tax_amount - discount_amount + shipping_amount + round_off_amount == total_amount",
        "input_values": {
          "subtotal": 5815.17,
          "tax_amount": 1046.72,
          "round_off_amount": 0.11,
          "total_amount": 6862.00
        },
        "calculated_value": 6862.00,
        "reported_value": 6862.00,
        "variance": 0.0,
        "status": "PASS",
        "tolerance": 0.01,
        "message": "Grand total reconciliation passed (5815.17 + 1046.72 + 0.11 == 6862.0)."
      }
    ],
    "summary": {
      "is_valid": true,
      "total_checks": 2,
      "passed_checks": 2,
      "failed_checks": 0,
      "not_applicable_checks": 0
    }
  },
  "created_at": "2026-09-11T19:00:00Z"
}
```

---

## 10. Environment Variables

Configure environment variables in a root `.env` file (copied from `.env.example`):

| Variable | Type | Default | Description |
| :--- | :---: | :--- | :--- |
| `DATABASE_URL` | String | *Required* | PostgreSQL connection string (`postgresql://user:password@host:port/dbname`). |
| `GEMINI_API_KEY` | String | *Required* | Google Gemini API key for structured information extraction. |
| `LLM_PROVIDER` | String | `gemini` | Primary LLM provider (`gemini` or `openai`). |
| `LLM_MODEL` | String | `gemini-3.6-flash` | Primary Gemini model attempted prior to fallback ladder cascade. |
| `LLM_TEMPERATURE` | Float | `0.0` | Sampling temperature (`0.0` for deterministic extraction). |
| `TESSERACT_CMD` | String | *Optional* | Path to Tesseract binary (Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`; Linux/Docker: `/usr/bin/tesseract` or omitted). |
| `FINANCIAL_VALIDATION_TOLERANCE` | Float | `0.01` | Maximum absolute difference allowed for mathematical equality checks. |
| `PORT` | Integer | `8000` | Port for the FastAPI server (dynamically overridden by Render). |
| `ENVIRONMENT` | String | `development` | Deployment environment (`development` / `production`). |

---

## 11. Local Setup & Execution Guide

### Prerequisites
- Python 3.12+
- PostgreSQL 14+ (or hosted database such as Render PostgreSQL / Supabase)
- Tesseract OCR (Windows installer or `sudo apt-get install tesseract-ocr`)

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SumitKuSinha/document-intelligence-platform.git
   cd document-intelligence-platform
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv backend\.venv
   .\backend\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv backend/.venv
   source backend/.venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

4. **Configure Environment Variables**:
   ```bash
   copy .env.example .env
   # Edit .env with your DATABASE_URL and GEMINI_API_KEY
   ```

5. **Execute Database Migrations**:
   ```bash
   cd backend
   alembic upgrade head
   cd ..
   ```

6. **Start the Application**:
   ```bash
   cd backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. **Access the Platform**:
   - Web Dashboard: `http://localhost:8000/`
   - API Docs (Swagger): `http://localhost:8000/docs`
   - Healthcheck: `http://localhost:8000/api/v1/health`

---

## 12. Docker Deployment

The platform provides a production-ready `Dockerfile` bundling Python 3.12, system OCR binaries, and static asset mapping.

### Build and Run Locally with Docker
```bash
# Build the Docker image
docker build -t document-intelligence-platform .

# Run container with environment variables
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:password@host:5432/dbname" \
  -e GEMINI_API_KEY="your-api-key" \
  document-intelligence-platform
```

### Deploying on Render
1. Create a **New Web Service** on Render connected to your repository.
2. Under **Build & Deploy**:
   - **Environment**: Select **Docker**.
   - **Dockerfile Path**: `Dockerfile`
   - **Docker Context**: `.`
3. Under **Environment Variables**, add:
   - `DATABASE_URL`: Connection string from your Render PostgreSQL database.
   - `GEMINI_API_KEY`: Your Google Gemini API Key.
   - `LLM_PROVIDER`: `gemini`
   - `LLM_MODEL`: `gemini-3.6-flash`
   - `LLM_TEMPERATURE`: `0.0`
   - `FINANCIAL_VALIDATION_TOLERANCE`: `0.01`
4. Render automatically manages `$PORT` and container health checks.

### Live Deployment Links
- **Live Dashboard**: `https://document-intelligence-platform.onrender.com/`
- **Swagger Documentation**: `https://document-intelligence-platform.onrender.com/docs`
- **Healthcheck**: `https://document-intelligence-platform.onrender.com/api/v1/health`

---

## 13. Automated Testing Suite

The platform includes a comprehensive, automated test suite covering all layers of the architecture:

```bash
cd backend
python -m unittest discover -s tests -p "test_*.py"
```

### Test Suite Coverage Breakdown (157 Tests Total)
| Test Module | Tests | Scope |
| :--- | :---: | :--- |
| [`test_file_validation.py`](backend/tests/test_file_validation.py) | 26 | Magic bytes verification, MIME types, file size limits, corrupted header rejection. |
| [`test_text_extraction.py`](backend/tests/test_text_extraction.py) | 24 | Multi-page digital PDF extraction, OCR raster fallback, page indexing, verbatim text. |
| [`test_financial_extraction.py`](backend/tests/test_financial_extraction.py) | 28 | Pydantic schema validation, prompt generation, alias choices, source snippet extraction. |
| [`test_gemini_fallback.py`](backend/tests/test_gemini_fallback.py) | 18 | Multi-tier fallback ladder, 429 quota cascades, Flash Lite failover, log credential redaction. |
| [`test_financial_validation.py`](backend/tests/test_financial_validation.py) | 39 | Line-item discount math, balance sheet equations, P&L rules, cash flow rules, document-type isolation. |
| [`test_api_documents.py`](backend/tests/test_api_documents.py) | 16 | FastAPI route integration, multipart uploads, error payloads, pagination, Swagger documentation. |
| [`test_database_connection.py`](backend/tests/test_database_connection.py) | 6 | SQLAlchemy connection pool, driver URL transformation (`postgresql+psycopg`), ORM CRUD operations. |
| **Total Passed** | **157 / 157** | **100% Pass Rate (~1.6s execution time)** |

---

## 14. Current Limitations & Edge Cases

1. **Severely Degraded Physical Documents**: Heavily skewed, torn, low-DPI (< 150 DPI), or stained physical paper invoices can lead to optical character misrecognitions prior to LLM processing.
2. **Multi-Currency Transactions**: Invoices containing line items priced in multiple distinct currencies on the same page are normalized to the primary invoice currency.
3. **Complex Nested Multi-Column Footnotes**: Banking balance sheets with footnotes spanning multiple subsequent pages require sequential multi-page context for complete reconciliation.

---

## 15. Production Roadmap & Future Improvements

- **Asynchronous Task Queue (Celery / Redis / AWS SQS)**: Decouple document ingestion from synchronous request-response cycles, enabling parallel bulk processing of 100+ page annual reports.
- **Object Storage Integration (AWS S3 / GCS / Cloudflare R2)**: Store raw binary files and extracted page thumbnails in object storage rather than passing byte streams in memory.
- **LayoutLM / Multimodal Vision Fine-Tuning**: Implement layout-aware bounding box embeddings for spatial table extraction across complex multi-column statements.
- **Role-Based Access Control (RBAC) & OAuth2**: Implement user authentication, tenant isolation, and audit trail signing for enterprise compliance.

---

## 16. AI & Tool Usage Declaration

In accordance with academic and case study transparency requirements, generative AI coding assistants (including Antigravity CLI and Google Gemini) were utilized during the development of this platform for architectural brainstorming, prompt engineering, regex refinement, test case synthesis, and deployment configuration. All application logic, schema designs, validation algorithms, and test suites were reviewed, verified, and audited for technical accuracy.
