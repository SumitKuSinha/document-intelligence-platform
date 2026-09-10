"""
Financial Document Extraction Service

Orchestrates AI/LLM-powered information extraction for financial documents:
- Invoices
- Balance Sheets
- Profit & Loss Statements (Income Statements)
- Cash Flow Statements

Features:
- Extracts all visible meaningful data into strongly-typed Pydantic schemas.
- Strict anti-hallucination policy: unobserved values are explicitly null.
- Gathers source evidence (verbatim snippet and page number) for extracted fields.
- Formats output for API delivery, PostgreSQL JSONB storage, and downstream mathematical validation.
- Graceful degradation on empty text, malformed responses, provider failures, and invalid types.
"""

from typing import Any, Dict, List, Optional, Union

from app.schemas.financial_extraction import (
    BalanceSheetExtractionData,
    CashFlowExtractionData,
    FieldEvidence,
    FinancialDocumentType,
    FinancialExtractionResult,
    InvoiceExtractionData,
    ProfitAndLossExtractionData,
)
from app.schemas.text_extraction import ExtractedPage
from app.services.llm_client import LLMClient, LLMClientError


class FinancialExtractionService:
    """
    Service for extracting structured financial data using Large Language Models.
    """

    SUPPORTED_TYPES = {
        FinancialDocumentType.INVOICE.value,
        FinancialDocumentType.BALANCE_SHEET.value,
        FinancialDocumentType.PROFIT_AND_LOSS.value,
        FinancialDocumentType.CASH_FLOW_STATEMENT.value,
    }

    # ---------------------------------------------------------------------------
    # Main Extraction Method
    # ---------------------------------------------------------------------------

    @classmethod
    def extract_financial_data(
        cls,
        document_type: str,
        extracted_text: str,
        pages: Optional[List[ExtractedPage]] = None,
    ) -> FinancialExtractionResult:
        """
        Extract structured financial metrics, line items, and evidence from document text.

        Args:
            document_type: Category ('invoice', 'balance_sheet', 'profit_and_loss', 'cash_flow_statement').
            extracted_text: Full raw text extracted from the document.
            pages: Optional list of pages with page numbers and text for provenance tracking.

        Returns:
            FinancialExtractionResult: Structured extraction outcome with validated fields and evidence.
        """
        doc_type_clean = (document_type or "").strip().lower()

        # 1. Validate document type
        if doc_type_clean not in cls.SUPPORTED_TYPES:
            return FinancialExtractionResult(
                document_type=doc_type_clean or "unknown",
                status="FAILED",
                data=None,
                evidence={},
                errors=[
                    f"Unsupported document type: '{document_type}'. "
                    f"Supported types are: {', '.join(sorted(cls.SUPPORTED_TYPES))}."
                ],
            )

        # 2. Validate input text presence
        if not extracted_text or not extracted_text.strip():
            return FinancialExtractionResult(
                document_type=doc_type_clean,
                status="FAILED",
                data=None,
                evidence={},
                errors=["Document text is empty or unreadable. Cannot perform extraction."],
            )

        # 3. Format document text and page context for the LLM
        formatted_context = cls._format_page_context(extracted_text, pages)

        # 4. Generate system and user prompts
        system_prompt = cls._build_system_prompt(doc_type_clean)
        user_prompt = cls._build_user_prompt(doc_type_clean, formatted_context)

        # 5. Call LLM Client
        try:
            raw_response = LLMClient.generate_json(system_prompt, user_prompt)
        except LLMClientError as exc:
            return FinancialExtractionResult(
                document_type=doc_type_clean,
                status="FAILED",
                data=None,
                evidence={},
                errors=[f"LLM extraction failure: {exc}"],
            )
        except Exception as exc:
            return FinancialExtractionResult(
                document_type=doc_type_clean,
                status="FAILED",
                data=None,
                evidence={},
                errors=[f"Unexpected error during extraction: {exc}"],
            )

        # 6. Parse and validate LLM output against document-specific schema
        return cls._parse_and_validate_response(doc_type_clean, raw_response)

    # ---------------------------------------------------------------------------
    # Context & Prompt Builders
    # ---------------------------------------------------------------------------

    @classmethod
    def _format_page_context(
        cls,
        extracted_text: str,
        pages: Optional[List[ExtractedPage]] = None,
    ) -> str:
        """Format the extracted document text with clear page delineation for evidence mapping."""
        if pages and len(pages) > 0:
            formatted_parts = []
            for p in pages:
                formatted_parts.append(
                    f"--- [PAGE {p.page_number}] ---\n{p.text.strip()}\n"
                )
            return "\n".join(formatted_parts)

        return extracted_text.strip()

    @classmethod
    def _build_system_prompt(cls, document_type: str) -> str:
        """Construct the system instructions tailored to the document category."""
        type_instructions = {
            "invoice": (
                "Document Category: INVOICE\n"
                "Target Fields:\n"
                "- invoice_number (string or null)\n"
                "- invoice_date (string/ISO format or null)\n"
                "- due_date (string/ISO format or null)\n"
                "- purchase_order_number (string or null)\n"
                "- vendor_name (string or null)\n"
                "- vendor_address (string or null)\n"
                "- vendor_tax_id (string or null)\n"
                "- customer_name (string or null)\n"
                "- customer_address (string or null)\n"
                "- subtotal (float or null)\n"
                "- tax_amount (float or null)\n"
                "- tax_rate (float or null, e.g. 0.08 for 8%)\n"
                "- discount_amount (float or null)\n"
                "- shipping_amount (float or null)\n"
                "- total_amount (float or null)\n"
                "- currency (string ISO code e.g. USD, EUR, INR or null)\n"
                "- payment_terms (string or null)\n"
                "- line_items (list of objects: description, quantity, unit_price, total_price, item_code, source_snippet, page_number)\n"
                "- additional_fields (object: any other visible key-values like bank details, notes, etc.)\n"
            ),
            "balance_sheet": (
                "Document Category: BALANCE SHEET (Statement of Financial Position)\n"
                "Target Fields:\n"
                "- company_name (string or null)\n"
                "- statement_date (string/ISO format or null)\n"
                "- reporting_period (string or null)\n"
                "- currency (string or null)\n"
                "- current_assets (float or null)\n"
                "- non_current_assets (float or null)\n"
                "- total_assets (float or null)\n"
                "- current_liabilities (float or null)\n"
                "- non_current_liabilities (float or null)\n"
                "- total_liabilities (float or null)\n"
                "- retained_earnings (float or null)\n"
                "- share_capital (float or null)\n"
                "- total_equity (float or null)\n"
                "- total_liabilities_and_equity (float or null)\n"
                "- line_items (list of objects: category, item_name, amount, source_snippet, page_number)\n"
                "- additional_fields (object: any other line items or disclosures)\n"
            ),
            "profit_and_loss": (
                "Document Category: PROFIT AND LOSS (Income Statement)\n"
                "Target Fields:\n"
                "- company_name (string or null)\n"
                "- reporting_period (string or null)\n"
                "- period_start_date (string or null)\n"
                "- period_end_date (string or null)\n"
                "- currency (string or null)\n"
                "- total_revenue (float or null)\n"
                "- cost_of_goods_sold (float or null)\n"
                "- gross_profit (float or null)\n"
                "- operating_expenses (float or null)\n"
                "- operating_income (float or null)\n"
                "- interest_expense (float or null)\n"
                "- tax_expense (float or null)\n"
                "- net_income (float or null)\n"
                "- line_items (list of objects: category, item_name, amount, source_snippet, page_number)\n"
                "- additional_fields (object: any other line items or notes)\n"
            ),
            "cash_flow_statement": (
                "Document Category: CASH FLOW STATEMENT\n"
                "Target Fields:\n"
                "- company_name (string or null)\n"
                "- reporting_period (string or null)\n"
                "- period_start_date (string or null)\n"
                "- period_end_date (string or null)\n"
                "- currency (string or null)\n"
                "- net_cash_from_operating_activities (float or null)\n"
                "- net_cash_from_investing_activities (float or null)\n"
                "- net_cash_from_financing_activities (float or null)\n"
                "- net_change_in_cash (float or null)\n"
                "- beginning_cash_balance (float or null)\n"
                "- ending_cash_balance (float or null)\n"
                "- line_items (list of objects: category, item_name, amount, source_snippet, page_number)\n"
                "- additional_fields (object: any other cash flow disclosures)\n"
            ),
        }

        return (
            "You are an expert financial document extraction engine.\n"
            "Your objective is to accurately extract ALL visible, meaningful financial information.\n\n"
            "STRICT EXTRACTION RULES:\n"
            "1. NEVER invent, guess, calculate, or hallucinate values. Extract only what is explicitly printed.\n"
            "2. If a field is not present in the document text, you MUST return null.\n"
            "3. Extract ALL visible information: primary totals, metadata, every individual line item, and "
            "place any unmodeled details inside 'additional_fields'.\n"
            "4. Numbers must be numeric floats without currency symbols or commas (e.g. 1250.00, not '$1,250.00').\n"
            "5. For every extracted top-level field, include an entry in the 'evidence' dictionary with:\n"
            "   - 'source_snippet': the verbatim text snippet from the document.\n"
            "   - 'page_number': the 1-indexed page number where it appears (from the [PAGE X] headers).\n\n"
            f"{type_instructions.get(document_type, '')}\n"
            "Return JSON matching this top-level format exactly:\n"
            "{\n"
            '  "extracted_data": { ... document fields ... },\n'
            '  "evidence": {\n'
            '    "field_name": {"source_snippet": "...", "page_number": 1}\n'
            "  }\n"
            "}"
        )

    @classmethod
    def _build_user_prompt(cls, document_type: str, formatted_context: str) -> str:
        """Construct the user prompt providing the document text."""
        return (
            f"Extract all financial data and line items for this {document_type.replace('_', ' ').upper()} document.\n\n"
            f"{formatted_context}"
        )

    # ---------------------------------------------------------------------------
    # Response Parsing & Schema Validation
    # ---------------------------------------------------------------------------

    @classmethod
    def _parse_and_validate_response(
        cls,
        document_type: str,
        raw_response: Dict[str, Any],
    ) -> FinancialExtractionResult:
        """Validate LLM output dictionary with the corresponding Pydantic schema."""
        errors: List[str] = []
        warnings: List[str] = []

        # Extract nested payload and evidence
        raw_data = raw_response.get("extracted_data") or raw_response
        raw_evidence = raw_response.get("evidence") or {}

        # Parse evidence
        parsed_evidence: Dict[str, FieldEvidence] = {}
        if isinstance(raw_evidence, dict):
            for field_key, ev_dict in raw_evidence.items():
                if isinstance(ev_dict, dict):
                    parsed_evidence[field_key] = FieldEvidence(
                        source_snippet=ev_dict.get("source_snippet"),
                        page_number=ev_dict.get("page_number"),
                    )

        # Validate with the respective Pydantic model
        validated_data: Optional[Dict[str, Any]] = None
        try:
            if document_type == "invoice":
                model_inst = InvoiceExtractionData.model_validate(raw_data)
                validated_data = model_inst.model_dump()
            elif document_type == "balance_sheet":
                model_inst = BalanceSheetExtractionData.model_validate(raw_data)
                validated_data = model_inst.model_dump()
            elif document_type == "profit_and_loss":
                model_inst = ProfitAndLossExtractionData.model_validate(raw_data)
                validated_data = model_inst.model_dump()
            elif document_type == "cash_flow_statement":
                model_inst = CashFlowExtractionData.model_validate(raw_data)
                validated_data = model_inst.model_dump()
            else:
                validated_data = raw_data
        except Exception as exc:
            errors.append(f"Validation error against schema: {exc}")
            validated_data = raw_data if isinstance(raw_data, dict) else {}

        status = "FAILED" if errors and not validated_data else "SUCCESS"
        if errors and validated_data:
            status = "PARTIAL"

        return FinancialExtractionResult(
            document_type=document_type,
            status=status,
            data=validated_data,
            evidence=parsed_evidence,
            errors=errors,
            warnings=warnings,
        )


# Module-level convenience function
extract_financial_document = FinancialExtractionService.extract_financial_data
