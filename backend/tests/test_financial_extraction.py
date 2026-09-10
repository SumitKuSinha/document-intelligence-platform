"""
Automated Tests for AI/LLM Financial Document Extraction Service

Tests cover:
- Invoice extraction with line items and evidence
- Balance sheet extraction with asset/liability/equity structures
- Profit and Loss (Income Statement) extraction
- Cash Flow Statement extraction
- Strict anti-hallucination checks: missing fields must be null (None)
- Source evidence provenance (verbatim snippet and page numbers)
- Unmodeled field preservation in `additional_fields`
- Error handling: empty document text
- Error handling: unsupported document category
- Error handling: malformed LLM JSON output
- Error handling: LLM provider API failures
"""

from pathlib import Path
import sys
import unittest

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.financial_extraction import (
    BalanceSheetExtractionData,
    CashFlowExtractionData,
    FinancialDocumentType,
    FinancialExtractionResult,
    InvoiceExtractionData,
    ProfitAndLossExtractionData,
)
from app.schemas.text_extraction import ExtractedPage
from app.services.financial_extraction import (
    FinancialExtractionService,
    extract_financial_document,
)
from app.services.llm_client import LLMClient, LLMClientError


class TestFinancialExtractionService(unittest.TestCase):
    """Test suite for AI/LLM Financial Document Extraction Service."""

    def tearDown(self):
        """Reset mock responder after each test to avoid test pollution."""
        LLMClient.set_mock_responder(None)

    # -------------------------------------------------------------------------
    # 1. Invoice Extraction
    # -------------------------------------------------------------------------
    def test_invoice_extraction_success(self):
        """Verify successful extraction of invoice data including line items and evidence."""
        mock_payload = {
            "extracted_data": {
                "invoice_number": "INV-2024-889",
                "invoice_date": "2024-03-15",
                "due_date": "2024-04-15",
                "purchase_order_number": "PO-9912",
                "vendor_name": "Acme Industrial Supplies LLC",
                "vendor_address": "123 Factory Lane, Detroit, MI",
                "vendor_tax_id": "XX-1234567",
                "customer_name": "Global Tech Corp",
                "customer_address": "456 Silicon Blvd, San Jose, CA",
                "subtotal": 1200.00,
                "tax_amount": 96.00,
                "tax_rate": 0.08,
                "discount_amount": 50.00,
                "shipping_amount": 25.00,
                "total_amount": 1271.00,
                "currency": "USD",
                "payment_terms": "Net 30",
                "line_items": [
                    {
                        "item_code": "PART-01",
                        "description": "High-grade Steel Bolts M8",
                        "quantity": 100.0,
                        "unit_price": 5.00,
                        "total_price": 500.00,
                        "source_snippet": "PART-01 High-grade Steel Bolts M8 100 $5.00 $500.00",
                        "page_number": 1,
                    },
                    {
                        "item_code": "PART-02",
                        "description": "Titanium Washers",
                        "quantity": 100.0,
                        "unit_price": 7.00,
                        "total_price": 700.00,
                        "source_snippet": "PART-02 Titanium Washers 100 $7.00 $700.00",
                        "page_number": 1,
                    },
                ],
                "additional_fields": {
                    "remittance_bank": "First National Bank",
                    "remittance_account": "****5678",
                },
            },
            "evidence": {
                "invoice_number": {
                    "source_snippet": "Invoice #: INV-2024-889",
                    "page_number": 1,
                },
                "total_amount": {
                    "source_snippet": "Total Balance Due: $1,271.00",
                    "page_number": 1,
                },
                "vendor_name": {
                    "source_snippet": "Acme Industrial Supplies LLC",
                    "page_number": 1,
                },
            },
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        sample_text = (
            "Acme Industrial Supplies LLC\nInvoice #: INV-2024-889\nDate: 2024-03-15\n"
            "Total Balance Due: $1,271.00"
        )
        pages = [ExtractedPage(page_number=1, text=sample_text)]

        result = extract_financial_document("invoice", sample_text, pages)

        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.document_type, "invoice")
        self.assertEqual(len(result.errors), 0)

        data = result.data
        self.assertIsNotNone(data)
        self.assertEqual(data["invoice_number"], "INV-2024-889")
        self.assertEqual(data["vendor_name"], "Acme Industrial Supplies LLC")
        self.assertEqual(data["total_amount"], 1271.00)
        self.assertEqual(data["subtotal"], 1200.00)
        self.assertEqual(data["currency"], "USD")
        self.assertEqual(len(data["line_items"]), 2)
        self.assertEqual(data["line_items"][0]["description"], "High-grade Steel Bolts M8")
        self.assertEqual(data["line_items"][0]["total_price"], 500.00)
        self.assertEqual(data["additional_fields"]["remittance_bank"], "First National Bank")

        # Evidence assertions
        self.assertIn("invoice_number", result.evidence)
        self.assertEqual(result.evidence["invoice_number"].source_snippet, "Invoice #: INV-2024-889")
        self.assertEqual(result.evidence["invoice_number"].page_number, 1)

    # -------------------------------------------------------------------------
    # 2. Balance Sheet Extraction
    # -------------------------------------------------------------------------
    def test_balance_sheet_extraction_success(self):
        """Verify successful extraction of balance sheet figures and equity."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Apex Holdings Inc.",
                "statement_date": "2023-12-31",
                "reporting_period": "Year Ended 2023",
                "currency": "USD",
                "current_assets": 450000.00,
                "non_current_assets": 850000.00,
                "total_assets": 1300000.00,
                "current_liabilities": 200000.00,
                "non_current_liabilities": 400000.00,
                "total_liabilities": 600000.00,
                "retained_earnings": 500000.00,
                "share_capital": 200000.00,
                "total_equity": 700000.00,
                "total_liabilities_and_equity": 1300000.00,
                "line_items": [
                    {
                        "category": "current_assets",
                        "item_name": "Cash and Cash Equivalents",
                        "amount": 150000.00,
                        "source_snippet": "Cash and Cash Equivalents: $150,000",
                        "page_number": 1,
                    },
                    {
                        "category": "non_current_assets",
                        "item_name": "Property, Plant & Equipment",
                        "amount": 700000.00,
                        "source_snippet": "PP&E: $700,000",
                        "page_number": 1,
                    },
                ],
                "additional_fields": {},
            },
            "evidence": {
                "total_assets": {
                    "source_snippet": "TOTAL ASSETS: $1,300,000",
                    "page_number": 1,
                },
                "total_equity": {
                    "source_snippet": "TOTAL STOCKHOLDERS EQUITY: $700,000",
                    "page_number": 1,
                },
            },
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        sample_text = "Apex Holdings Inc. Consolidated Balance Sheet as of December 31, 2023..."
        result = extract_financial_document("balance_sheet", sample_text)

        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.document_type, "balance_sheet")
        self.assertEqual(result.data["company_name"], "Apex Holdings Inc.")
        self.assertEqual(result.data["total_assets"], 1300000.00)
        self.assertEqual(result.data["total_liabilities"], 600000.00)
        self.assertEqual(result.data["total_equity"], 700000.00)
        self.assertEqual(result.data["total_liabilities_and_equity"], 1300000.00)
        self.assertEqual(len(result.data["line_items"]), 2)
        self.assertEqual(result.evidence["total_assets"].page_number, 1)

    # -------------------------------------------------------------------------
    # 3. Profit and Loss Extraction
    # -------------------------------------------------------------------------
    def test_profit_and_loss_extraction_success(self):
        """Verify successful extraction of income statement metrics."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Zenith Dynamics Ltd",
                "reporting_period": "Q3 2024",
                "period_start_date": "2024-07-01",
                "period_end_date": "2024-09-30",
                "currency": "EUR",
                "total_revenue": 5000000.00,
                "cost_of_goods_sold": 2000000.00,
                "gross_profit": 3000000.00,
                "operating_expenses": 1200000.00,
                "operating_income": 1800000.00,
                "interest_expense": 50000.00,
                "tax_expense": 350000.00,
                "net_income": 1400000.00,
                "line_items": [
                    {
                        "category": "revenue",
                        "item_name": "SaaS Subscription Revenue",
                        "amount": 4200000.00,
                        "source_snippet": "SaaS Subscription Revenue EUR 4,200,000",
                        "page_number": 1,
                    },
                    {
                        "category": "operating_expenses",
                        "item_name": "Research & Development",
                        "amount": 600000.00,
                        "source_snippet": "R&D EUR 600,000",
                        "page_number": 2,
                    },
                ],
                "additional_fields": {"ebitda": 2100000.00},
            },
            "evidence": {
                "net_income": {
                    "source_snippet": "Net Income: EUR 1,400,000",
                    "page_number": 2,
                }
            },
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        sample_text = "Zenith Dynamics Ltd Statement of Profit or Loss Q3 2024..."
        result = extract_financial_document("profit_and_loss", sample_text)

        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.document_type, "profit_and_loss")
        self.assertEqual(result.data["total_revenue"], 5000000.00)
        self.assertEqual(result.data["gross_profit"], 3000000.00)
        self.assertEqual(result.data["net_income"], 1400000.00)
        self.assertEqual(result.data["currency"], "EUR")
        self.assertEqual(result.data["additional_fields"]["ebitda"], 2100000.00)
        self.assertEqual(len(result.data["line_items"]), 2)

    # -------------------------------------------------------------------------
    # 4. Cash Flow Statement Extraction
    # -------------------------------------------------------------------------
    def test_cash_flow_statement_extraction_success(self):
        """Verify successful extraction of cash flow activities."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Solaria Renewable Energy",
                "reporting_period": "FY 2023",
                "period_start_date": "2023-01-01",
                "period_end_date": "2023-12-31",
                "currency": "USD",
                "net_cash_from_operating_activities": 850000.00,
                "net_cash_from_investing_activities": -400000.00,
                "net_cash_from_financing_activities": -150000.00,
                "net_change_in_cash": 300000.00,
                "beginning_cash_balance": 500000.00,
                "ending_cash_balance": 800000.00,
                "line_items": [
                    {
                        "category": "operating",
                        "item_name": "Customer Receipts",
                        "amount": 950000.00,
                        "source_snippet": "Customer Receipts: $950,000",
                        "page_number": 1,
                    },
                    {
                        "category": "investing",
                        "item_name": "Purchase of Solar Panels Equipment",
                        "amount": -400000.00,
                        "source_snippet": "Capex Equipment: ($400,000)",
                        "page_number": 1,
                    },
                ],
                "additional_fields": {},
            },
            "evidence": {
                "net_change_in_cash": {
                    "source_snippet": "Net increase in cash and equivalents: $300,000",
                    "page_number": 1,
                }
            },
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        sample_text = "Solaria Statement of Cash Flows FY 2023..."
        result = extract_financial_document("cash_flow_statement", sample_text)

        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.document_type, "cash_flow_statement")
        self.assertEqual(result.data["net_cash_from_operating_activities"], 850000.00)
        self.assertEqual(result.data["net_cash_from_investing_activities"], -400000.00)
        self.assertEqual(result.data["net_change_in_cash"], 300000.00)
        self.assertEqual(result.data["ending_cash_balance"], 800000.00)

    # -------------------------------------------------------------------------
    # 5. Strict Anti-Hallucination: Missing Values Are Strictly Null
    # -------------------------------------------------------------------------
    def test_missing_values_are_strictly_null(self):
        """Ensure missing unprinted values remain None and are never hallucinated."""
        mock_payload = {
            "extracted_data": {
                "invoice_number": "INV-ONLY-NUM",
                "total_amount": 250.00,
                # All other fields omitted or null
                "invoice_date": None,
                "due_date": None,
                "purchase_order_number": None,
                "vendor_name": None,
                "vendor_address": None,
                "vendor_tax_id": None,
                "customer_name": None,
                "customer_address": None,
                "subtotal": None,
                "tax_amount": None,
                "tax_rate": None,
                "discount_amount": None,
                "shipping_amount": None,
                "currency": None,
                "payment_terms": None,
                "line_items": [],
                "additional_fields": {},
            },
            "evidence": {
                "invoice_number": {
                    "source_snippet": "INV-ONLY-NUM",
                    "page_number": 1,
                }
            },
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        sample_text = "INV-ONLY-NUM Total: 250"
        result = extract_financial_document("invoice", sample_text)

        self.assertEqual(result.status, "SUCCESS")
        data = result.data
        self.assertEqual(data["invoice_number"], "INV-ONLY-NUM")
        self.assertEqual(data["total_amount"], 250.00)

        # Explicitly check nulls
        self.assertIsNone(data["vendor_name"])
        self.assertIsNone(data["customer_name"])
        self.assertIsNone(data["due_date"])
        self.assertIsNone(data["tax_amount"])
        self.assertIsNone(data["payment_terms"])
        self.assertIsNone(data["discount_amount"])

    # -------------------------------------------------------------------------
    # 6. Additional Fields Preservation
    # -------------------------------------------------------------------------
    def test_additional_fields_preservation(self):
        """Ensure unstructured extra fields are retained in additional_fields dictionary."""
        mock_payload = {
            "extracted_data": {
                "company_name": "Fintech Alpha",
                "statement_date": "2023-12-31",
                "additional_fields": {
                    "auditor": "PricewaterhouseCoopers",
                    "accounting_standard": "IFRS",
                    "notes_to_financial_statements": "Note 4: Revenue Recognition",
                },
            },
            "evidence": {},
        }

        LLMClient.set_mock_responder(lambda sys_p, usr_p: mock_payload)

        result = extract_financial_document(
            "balance_sheet", "Fintech Alpha Balance Sheet audited by PwC..."
        )

        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(
            result.data["additional_fields"]["auditor"], "PricewaterhouseCoopers"
        )
        self.assertEqual(
            result.data["additional_fields"]["accounting_standard"], "IFRS"
        )

    # -------------------------------------------------------------------------
    # 7. Error Handling: Empty Document Text
    # -------------------------------------------------------------------------
    def test_empty_document_text_failure(self):
        """Service should fail gracefully when document text is empty or blank."""
        result = extract_financial_document("invoice", "   \n\t  ")

        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.data)
        self.assertTrue(any("empty" in err.lower() for err in result.errors))

    # -------------------------------------------------------------------------
    # 8. Error Handling: Unsupported Document Type
    # -------------------------------------------------------------------------
    def test_unsupported_document_type_failure(self):
        """Service should reject unhandled document categories with informative error."""
        result = extract_financial_document("mortgage_statement", "Some financial text")

        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.data)
        self.assertTrue(any("unsupported" in err.lower() for err in result.errors))
        self.assertIn("invoice", result.errors[0].lower())

    # -------------------------------------------------------------------------
    # 9. Error Handling: Malformed LLM JSON Output
    # -------------------------------------------------------------------------
    def test_malformed_llm_json_failure(self):
        """Service should handle invalid JSON produced by the LLM gracefully."""

        def bad_json_responder(sys_p, usr_p):
            # Return invalid non-JSON string
            return "NOT_JSON_AT_ALL {malformed..."

        LLMClient.set_mock_responder(bad_json_responder)

        result = extract_financial_document("invoice", "Invoice # 123 Total $50")

        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.data)
        self.assertTrue(any("llm extraction failure" in err.lower() for err in result.errors))

    # -------------------------------------------------------------------------
    # 10. Error Handling: Provider API / Network Failures
    # -------------------------------------------------------------------------
    def test_llm_provider_network_failure(self):
        """Service should catch LLMClientError exceptions and format into result.errors."""

        def connection_error_responder(sys_p, usr_p):
            raise LLMClientError("Network timeout connecting to LLM endpoint.")

        LLMClient.set_mock_responder(connection_error_responder)

        result = extract_financial_document("invoice", "Invoice text")

        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.data)
        self.assertTrue(any("network timeout" in err.lower() for err in result.errors))

    # -------------------------------------------------------------------------
    # 11. Page Context Formatting
    # -------------------------------------------------------------------------
    def test_page_context_formatting(self):
        """Verify page markers are inserted to assist LLM evidence tracking."""
        pages = [
            ExtractedPage(page_number=1, text="Page 1 Content"),
            ExtractedPage(page_number=2, text="Page 2 Content"),
        ]

        formatted = FinancialExtractionService._format_page_context("Fallback text", pages)

        self.assertIn("--- [PAGE 1] ---", formatted)
        self.assertIn("Page 1 Content", formatted)
        self.assertIn("--- [PAGE 2] ---", formatted)
        self.assertIn("Page 2 Content", formatted)

    # -------------------------------------------------------------------------
    # 12. Google Gemini 2.5 Flash Configuration & Client Unit Tests
    # -------------------------------------------------------------------------
    def test_gemini_configuration_defaults(self):
        """Verify Gemini provider and Gemini 2.5 Flash model defaults."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("GEMINI_MODEL", None)
            os.environ.pop("LLM_MODEL", None)

            self.assertEqual(LLMClient.get_provider(), "gemini")
            self.assertEqual(LLMClient.get_model(), "gemini-2.5-flash")

    def test_gemini_client_missing_key_raises_error(self):
        """Verify helpful error is raised when GEMINI_API_KEY is not configured."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)

            with self.assertRaises(LLMClientError) as ctx:
                LLMClient._create_gemini_client()

            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_gemini_client_creation_with_key(self):
        """Verify google-genai Client can be initialized with dummy key without network calls."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-test-key-no-network"}):
            client = LLMClient._create_gemini_client()
            self.assertIsNotNone(client)

    def test_clean_json_text_strips_markdown_fences(self):
        """Ensure markdown json code blocks returned by LLMs are cleanly stripped."""
        raw_json_with_fence = "```json\n{\"test\": 123}\n```"
        cleaned = LLMClient._clean_json_text(raw_json_with_fence)
        self.assertEqual(cleaned, '{"test": 123}')

        raw_fence_plain = "```\n{\"test\": 456}\n```"
        cleaned_plain = LLMClient._clean_json_text(raw_fence_plain)
        self.assertEqual(cleaned_plain, '{"test": 456}')

    def test_unsupported_provider_raises_error(self):
        """Verify error is raised when an unknown LLM provider is configured."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported_provider"}):
            with self.assertRaises(LLMClientError) as ctx:
                LLMClient.generate_json("system", "user")

            self.assertIn("Unsupported LLM provider", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

