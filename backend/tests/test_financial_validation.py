"""
Automated Tests for Deterministic Financial Validation Engine

Covers:
- Numeric string parsing (commas, currencies, percentages, parentheses, trailing negs/CR)
- Invoices: line item math, line items subtotal sum, total reconciliation, tax calculation
- Balance Sheet: accounting equation, assets components, liabilities components, equity components
- Profit and Loss: gross profit, operating income, net income
- Cash Flow Statements: net change in cash, ending cash balance, negative values with parentheses
- PASS, FAILED, and NOT_APPLICABLE statuses
- Missing values handled strictly as NOT_APPLICABLE with missing_fields identified (never assumed zero)
- Rounding tolerance and configurable environment variable
"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Ensure backend root is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.financial_extraction import (
    BalanceSheetExtractionData,
    CashFlowExtractionData,
    FinancialExtractionResult,
    InvoiceExtractionData,
    InvoiceLineItem,
    ProfitAndLossExtractionData,
)
from app.schemas.financial_validation import (
    FinancialValidationResult,
    ValidationCheckResult,
    ValidationStatus,
)
from app.services.financial_validation import (
    FinancialValidationService,
    parse_financial_number,
    validate_financial_data,
)


class TestNumericParsing(unittest.TestCase):
    """Unit tests for robust financial number parsing."""

    def test_parse_plain_numbers(self):
        self.assertEqual(parse_financial_number(100), 100.0)
        self.assertEqual(parse_financial_number(125.75), 125.75)
        self.assertEqual(parse_financial_number(0), 0.0)

    def test_parse_commas(self):
        self.assertEqual(parse_financial_number("1,234.56"), 1234.56)
        self.assertEqual(parse_financial_number("1,000,000.50"), 1000000.50)
        self.assertEqual(parse_financial_number("1,250"), 1250.0)

    def test_parse_currencies(self):
        self.assertEqual(parse_financial_number("$1,200.50"), 1200.50)
        self.assertEqual(parse_financial_number("€ 500.00"), 500.00)
        self.assertEqual(parse_financial_number("£75.25"), 75.25)
        self.assertEqual(parse_financial_number("USD 10,000.00"), 10000.00)
        self.assertEqual(parse_financial_number("EUR 450"), 450.0)
        self.assertEqual(parse_financial_number("₹ 999.50"), 999.50)

    def test_parse_parentheses_negatives(self):
        self.assertEqual(parse_financial_number("(400,000)"), -400000.0)
        self.assertEqual(parse_financial_number("( $1,250.50 )"), -1250.50)
        self.assertEqual(parse_financial_number("(50)"), -50.0)

    def test_parse_percentages(self):
        self.assertEqual(parse_financial_number("8%"), 0.08)
        self.assertEqual(parse_financial_number("15.5%"), 0.155)

    def test_parse_trailing_negative_and_cr(self):
        self.assertEqual(parse_financial_number("1,234.50-"), -1234.50)
        self.assertEqual(parse_financial_number("500 CR"), -500.0)
        self.assertEqual(parse_financial_number("-125.00"), -125.00)

    def test_parse_null_and_malformed(self):
        self.assertIsNone(parse_financial_number(None))
        self.assertIsNone(parse_financial_number(""))
        self.assertIsNone(parse_financial_number("   "))
        self.assertIsNone(parse_financial_number("null"))
        self.assertIsNone(parse_financial_number("N/A"))
        self.assertIsNone(parse_financial_number("-"))
        self.assertIsNone(parse_financial_number("--"))
        self.assertIsNone(parse_financial_number("invalid_text"))


class TestInvoiceValidation(unittest.TestCase):
    """Unit tests for invoice mathematical reconciliation."""

    def test_invoice_all_checks_pass(self):
        payload = {
            "subtotal": 1200.00,
            "tax_amount": 96.00,
            "tax_rate": 0.08,
            "discount_amount": 50.00,
            "shipping_amount": 25.00,
            "total_amount": 1271.00,
            "line_items": [
                {"description": "Bolts", "quantity": 100.0, "unit_price": 5.0, "total_price": 500.0},
                {"description": "Washers", "quantity": 100.0, "unit_price": 7.0, "total_price": 700.0},
            ],
        }

        result = validate_financial_data("invoice", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 5)
        # 2 line math checks + 1 subtotal sum + 1 total reconciliation + 1 tax calculation = 5 checks

    def test_invoice_line_item_math_fail(self):
        payload = {
            "subtotal": 100.0,
            "total_amount": 100.0,
            "line_items": [
                {"description": "Widget", "quantity": 10.0, "unit_price": 5.0, "total_price": 60.0},
            ],
        }

        result = validate_financial_data("invoice", payload)

        self.assertFalse(result.is_valid)
        self.assertGreater(result.summary.failed_checks, 0)
        failed_check = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(failed_check.status, ValidationStatus.FAILED)
        self.assertEqual(failed_check.calculated_value, 50.0)
        self.assertEqual(failed_check.reported_value, 60.0)
        self.assertEqual(failed_check.variance, -10.0)

    # -------------------------------------------------------------------------
    # Line Item Arithmetic Discount Regression Tests
    # -------------------------------------------------------------------------
    def test_invoice_line_item_no_discount_existing_formula(self):
        """Verify line item with no discount uses the existing standard formula: quantity * unit_price == total_price."""
        payload = {
            "subtotal": 50.0,
            "total_amount": 50.0,
            "line_items": [
                {"description": "Standard Item", "quantity": 5.0, "unit_price": 10.0, "total_price": 50.0},
            ],
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(check.formula, "quantity * unit_price == total_price")
        self.assertEqual(check.calculated_value, 50.0)
        self.assertEqual(check.reported_value, 50.0)
        self.assertNotIn("discount_percent", check.input_values)
        self.assertNotIn("discount_amount", check.input_values)

    def test_invoice_line_item_99_percent_discount_examples(self):
        """Verify line items with 99% discount pass arithmetic validation for both user-reported examples."""
        # Example 1: Quantity = 6, Rate = 7.44, Discount = 99%, Reported amount = 0.45
        # Example 2: Quantity = 8, Rate = 7.44, Discount = 99%, Reported amount = 0.60
        payload = {
            "subtotal": 1.05,
            "total_amount": 1.05,
            "line_items": [
                {"description": "Item 1", "quantity": 6.0, "rate": 7.44, "discount": "99%", "amount": 0.45},
                {"description": "Item 2", "quantity": 8.0, "unit_price": 7.44, "discount": "99%", "total_price": 0.60},
            ],
        }
        result = validate_financial_data("invoice", payload)

        check1 = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check1.status, ValidationStatus.PASS)
        self.assertEqual(check1.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")
        self.assertEqual(check1.input_values["discount_percent"], 99.0)
        self.assertEqual(check1.reported_value, 0.45)

        check2 = next(c for c in result.checks if c.rule_name == "invoice_line_item_2_math")
        self.assertEqual(check2.status, ValidationStatus.PASS)
        self.assertEqual(check2.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")
        self.assertEqual(check2.input_values["discount_percent"], 99.0)
        self.assertEqual(check2.reported_value, 0.60)

    def test_invoice_line_item_normal_percentage_discount(self):
        """Verify line items with standard percentage discounts calculate and validate accurately."""
        payload = {
            "subtotal": 470.0,
            "total_amount": 470.0,
            "line_items": [
                # 10 * 20 * (1 - 0.15) = 200 * 0.85 = 170.0
                {"description": "Widget A", "quantity": 10.0, "unit_price": 20.0, "discount_percent": 15.0, "total_price": 170.0},
                # 4 * 100 * (1 - 0.25) = 400 * 0.75 = 300.0
                {"description": "Widget B", "quantity": 4.0, "unit_price": 100.0, "discount": "25%", "total_price": 300.0},
            ],
        }
        result = validate_financial_data("invoice", payload)

        check1 = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check1.status, ValidationStatus.PASS)
        self.assertEqual(check1.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")
        self.assertEqual(check1.input_values["discount_percent"], 15.0)
        self.assertEqual(check1.calculated_value, 170.0)

        check2 = next(c for c in result.checks if c.rule_name == "invoice_line_item_2_math")
        self.assertEqual(check2.status, ValidationStatus.PASS)
        self.assertEqual(check2.calculated_value, 300.0)

    def test_invoice_line_item_monetary_discount(self):
        """Verify line items with explicit monetary discounts calculate appropriately."""
        payload = {
            "subtotal": 175.0,
            "total_amount": 175.0,
            "line_items": [
                # 2 * 50 - 10 = 90.0
                {"description": "Item A", "quantity": 2.0, "unit_price": 50.0, "discount_amount": 10.0, "total_price": 90.0},
                # 5 * 20 - 15 = 85.0
                {"description": "Item B", "quantity": 5.0, "unit_price": 20.0, "discount": "$15.00", "total_price": 85.0},
            ],
        }
        result = validate_financial_data("invoice", payload)

        check1 = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check1.status, ValidationStatus.PASS)
        self.assertEqual(check1.formula, "quantity * unit_price - discount_amount == total_price")
        self.assertEqual(check1.input_values["discount_amount"], 10.0)
        self.assertEqual(check1.calculated_value, 90.0)

        check2 = next(c for c in result.checks if c.rule_name == "invoice_line_item_2_math")
        self.assertEqual(check2.status, ValidationStatus.PASS)
        self.assertEqual(check2.input_values["discount_amount"], 15.0)
        self.assertEqual(check2.calculated_value, 85.0)

    def test_invoice_line_item_missing_discount_preserves_existing_behavior(self):
        """Verify when discount is missing or None, validator does not invent one and uses quantity * unit_price == total_price."""
        # Passing case with missing discount field
        payload_pass = {
            "subtotal": 100.0,
            "total_amount": 100.0,
            "line_items": [
                {"description": "Item A", "quantity": 2.0, "unit_price": 50.0, "discount": None, "total_price": 100.0},
            ],
        }
        res_pass = validate_financial_data("invoice", payload_pass)
        check_pass = next(c for c in res_pass.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_pass.status, ValidationStatus.PASS)
        self.assertEqual(check_pass.formula, "quantity * unit_price == total_price")

        # Failing case: arithmetic does not match and discount is missing -> MUST FAIL without guessing a discount
        payload_fail = {
            "subtotal": 0.45,
            "total_amount": 0.45,
            "line_items": [
                {"description": "Item B", "quantity": 6.0, "unit_price": 7.44, "discount": None, "total_price": 0.45},
            ],
        }
        res_fail = validate_financial_data("invoice", payload_fail)
        check_fail = next(c for c in res_fail.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_fail.status, ValidationStatus.FAILED)
        self.assertEqual(check_fail.formula, "quantity * unit_price == total_price")
        self.assertNotIn("discount_percent", check_fail.input_values)

    def test_invoice_line_item_invalid_or_ambiguous_discount(self):
        """Verify invalid or ambiguous discounts preserve the existing formula rather than guessing."""
        # 1. Invalid non-numeric discount string ("N/A", "invalid", "abc")
        payload_invalid = {
            "subtotal": 100.0,
            "total_amount": 100.0,
            "line_items": [
                {"description": "Item A", "quantity": 2.0, "unit_price": 50.0, "discount": "N/A", "total_price": 100.0},
                {"description": "Item B", "quantity": 2.0, "unit_price": 50.0, "discount": "invalid", "total_price": 60.0},
            ],
        }
        res_invalid = validate_financial_data("invoice", payload_invalid)
        check_a = next(c for c in res_invalid.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_a.status, ValidationStatus.PASS)
        self.assertEqual(check_a.formula, "quantity * unit_price == total_price")

        check_b = next(c for c in res_invalid.checks if c.rule_name == "invoice_line_item_2_math")
        self.assertEqual(check_b.status, ValidationStatus.FAILED)
        self.assertEqual(check_b.formula, "quantity * unit_price == total_price")

        # 2. Ambiguous numeric discount that matches neither percentage nor monetary
        payload_ambiguous = {
            "subtotal": 80.0,
            "total_amount": 80.0,
            "line_items": [
                # 10 * 10 = 100. 5% off = 95, $5 off = 95. Reported 80 matches neither.
                {"description": "Item C", "quantity": 10.0, "unit_price": 10.0, "discount": 5.0, "total_price": 80.0},
            ],
        }
        res_ambiguous = validate_financial_data("invoice", payload_ambiguous)
        check_c = next(c for c in res_ambiguous.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_c.status, ValidationStatus.FAILED)
        self.assertEqual(check_c.formula, "quantity * unit_price == total_price")

    def test_invoice_line_item_discount_in_source_snippet(self):
        """Verify discount reported in line item source snippet is recognized."""
        payload = {
            "subtotal": 0.45,
            "total_amount": 0.45,
            "line_items": [
                {
                    "description": "Widget Part",
                    "quantity": 6.0,
                    "unit_price": 7.44,
                    "total_price": 0.45,
                    "source_snippet": "Widget Part  Qty: 6  Rate: 7.44  Discount: 99%  Amount: 0.45",
                },
            ],
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(check.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")

    def test_invoice_line_item_discount_in_source_snippet_real_table_format(self):
        """Verify discount formatted as anonymous percentage in OCR table row is recognized and validated."""
        payload = {
            "subtotal": 1.05,
            "total_amount": 1.05,
            "line_items": [
                {
                    "item_code": "34029011",
                    "description": "SAFED WHITE DP 140GM 60PCS Twin Pack 10/-",
                    "quantity": 6.0,
                    "unit_price": 7.44,
                    "total_price": 0.45,
                    "source_snippet": "2 SAFED WHITE DP 140GM 60PCS Twin Pack 10/- 34029011 6 PCS 8.78 7.44 PCS 99 % 0.45",
                },
                {
                    "item_code": "34029011",
                    "description": "SAFED WHITE DP 140GM 60PCS Twin Pack 10/-",
                    "quantity": 8.0,
                    "unit_price": 7.44,
                    "total_price": 0.60,
                    "source_snippet": "5 SAFED WHITE DP 140GM 60PCS Twin Pack 10/- 34029011 8 PCS 8.78 7.44 PCS 99 % 0.60",
                },
            ],
        }
        result = validate_financial_data("invoice", payload)
        check1 = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        check2 = next(c for c in result.checks if c.rule_name == "invoice_line_item_2_math")

        self.assertEqual(check1.status, ValidationStatus.PASS)
        self.assertEqual(check1.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")
        self.assertEqual(check1.input_values["discount_percent"], 99.0)
        self.assertEqual(check1.calculated_value, 0.45)
        self.assertEqual(check1.reported_value, 0.45)

        self.assertEqual(check2.status, ValidationStatus.PASS)
        self.assertEqual(check2.formula, "quantity * unit_price * (1 - discount_percent / 100) == total_price")
        self.assertEqual(check2.input_values["discount_percent"], 99.0)
        self.assertEqual(check2.calculated_value, 0.60)
        self.assertEqual(check2.reported_value, 0.60)

    def test_invoice_line_item_unrelated_percentage_does_not_invent_discount(self):
        """Verify an unrelated percentage in snippet (e.g. tax rate or moisture %) is NOT treated as a discount."""
        # Case 1: Undiscounted line passes, snippet contains "18 %" (e.g. tax rate). Validator must NOT invent an 18% discount.
        payload_pass = {
            "subtotal": 500.0,
            "total_amount": 500.0,
            "line_items": [
                {
                    "description": "Chemical Compound",
                    "quantity": 10.0,
                    "unit_price": 50.0,
                    "total_price": 500.0,
                    "source_snippet": "1 Chemical Compound 10 PCS 50.00 18 % 500.00",
                }
            ],
        }
        res_pass = validate_financial_data("invoice", payload_pass)
        check_pass = next(c for c in res_pass.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_pass.status, ValidationStatus.PASS)
        self.assertEqual(check_pass.formula, "quantity * unit_price == total_price")
        self.assertNotIn("discount_percent", check_pass.input_values)

        # Case 2: Arithmetic is erroneous and unrelated percentage exists in snippet. Validator must fail standard math without guessing discount.
        payload_fail = {
            "subtotal": 400.0,
            "total_amount": 400.0,
            "line_items": [
                {
                    "description": "Chemical Compound",
                    "quantity": 10.0,
                    "unit_price": 50.0,
                    "total_price": 400.0,
                    "source_snippet": "1 Chemical Compound 10 PCS 50.00 18 % 400.00",
                }
            ],
        }
        res_fail = validate_financial_data("invoice", payload_fail)
        check_fail = next(c for c in res_fail.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(check_fail.status, ValidationStatus.FAILED)
        self.assertEqual(check_fail.formula, "quantity * unit_price == total_price")
        self.assertNotIn("discount_percent", check_fail.input_values)

    def test_invoice_subtotal_sum_mismatch(self):
        payload = {
            "subtotal": 1000.0,  # Reported subtotal 1000
            "total_amount": 1000.0,
            "line_items": [
                {"quantity": 5.0, "unit_price": 100.0, "total_price": 500.0},
                {"quantity": 4.0, "unit_price": 100.0, "total_price": 400.0},
                # Sum = 900 != 1000
            ],
        }

        result = validate_financial_data("invoice", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "invoice_line_items_subtotal_reconciliation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 900.0)
        self.assertEqual(check.reported_value, 1000.0)
        self.assertEqual(check.variance, -100.0)

    def test_invoice_total_reconciliation_fail(self):
        payload = {
            "subtotal": 500.0,
            "tax_amount": 50.0,
            "total_amount": 600.0,  # 500 + 50 = 550 != 600
        }

        result = validate_financial_data("invoice", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 550.0)
        self.assertEqual(check.reported_value, 600.0)
        self.assertEqual(check.variance, -50.0)

    def test_invoice_tax_calculation_fail(self):
        payload = {
            "subtotal": 1000.0,
            "tax_rate": 0.10,
            "tax_amount": 120.0,  # 1000 * 0.10 = 100 != 120
            "total_amount": 1120.0,
        }

        result = validate_financial_data("invoice", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 100.0)
        self.assertEqual(check.reported_value, 120.0)

    # ---------------------------------------------------------------------------
    # Regression Tests: Round-Off and GST Tax Components
    # ---------------------------------------------------------------------------

    def test_invoice_total_reconciliation_explicit_round_off(self):
        """Test A: Explicit round-off amount is included in total reconciliation formula and passes."""
        payload = {
            "subtotal": 5815.17,
            "tax_amount": 1046.72,
            "round_off": 0.11,
            "total_amount": 6862.00,
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(
            check.formula,
            "subtotal + tax_amount - discount_amount + shipping_amount + round_off_amount == total_amount",
        )
        self.assertEqual(check.input_values.get("round_off_amount"), 0.11)
        self.assertEqual(check.calculated_value, 6862.00)
        self.assertEqual(check.reported_value, 6862.00)
        self.assertEqual(check.variance, 0.0)

    def test_invoice_total_reconciliation_no_round_off(self):
        """Test B: When round-off is not reported, existing total reconciliation formula is preserved without inventing round-off."""
        payload = {
            "subtotal": 1000.0,
            "tax_amount": 100.0,
            "total_amount": 1100.0,
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(
            check.formula,
            "subtotal + tax_amount - discount_amount + shipping_amount == total_amount",
        )
        self.assertNotIn("round_off_amount", check.input_values)
        self.assertEqual(check.calculated_value, 1100.0)
        self.assertEqual(check.reported_value, 1100.0)
        self.assertEqual(check.variance, 0.0)

    def test_invoice_tax_calculation_explicit_cgst_sgst(self):
        """Test C: Explicit CGST + SGST tax components are calculated independently, rounded, and summed."""
        payload = {
            "taxable_amount": 5815.17,
            "CGST": "9%",
            "SGST": "9%",
            "cgst_amount": 523.36,
            "sgst_amount": 523.36,
            "tax_amount": 1046.72,
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(check.formula, "cgst_amount + sgst_amount == tax_amount")
        self.assertEqual(check.calculated_value, 1046.72)
        self.assertEqual(check.reported_value, 1046.72)
        self.assertEqual(check.variance, 0.0)
        self.assertEqual(check.input_values.get("cgst_amount"), 523.36)
        self.assertEqual(check.input_values.get("sgst_amount"), 523.36)

    def test_invoice_tax_calculation_explicit_igst(self):
        """Test D: Explicit IGST component is calculated, rounded, and validated against total tax."""
        payload = {
            "subtotal": 5815.17,
            "igst_rate": "18%",
            "igst_amount": 1046.73,
            "tax_amount": 1046.73,
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(check.formula, "igst_amount == tax_amount")
        self.assertEqual(check.calculated_value, 1046.73)
        self.assertEqual(check.reported_value, 1046.73)
        self.assertEqual(check.variance, 0.0)

    def test_invoice_tax_calculation_missing_tax_components(self):
        """Test E: Missing tax components preserves existing subtotal * tax_rate behavior."""
        payload = {
            "subtotal": 1000.0,
            "tax_rate": 0.10,
            "tax_amount": 100.0,
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(check.status, ValidationStatus.PASS)
        self.assertEqual(check.formula, "subtotal * tax_rate == tax_amount")
        self.assertEqual(check.calculated_value, 100.0)
        self.assertEqual(check.reported_value, 100.0)

    def test_invoice_tax_calculation_unrelated_percentages_not_treated_as_tax(self):
        """Test F: Unrelated percentages in additional_fields are NOT treated as tax components."""
        payload = {
            "subtotal": 1000.0,
            "tax_rate": 0.10,
            "tax_amount": 120.0,
            "additional_fields": {
                "purity_percentage": "99%",
                "brokerage_rate": "5%",
            },
        }
        result = validate_financial_data("invoice", payload)
        check = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.formula, "subtotal * tax_rate == tax_amount")
        self.assertEqual(check.calculated_value, 100.0)
        self.assertEqual(check.reported_value, 120.0)

    def test_invoice_real_indian_gst_invoice_end_to_end_validation(self):
        """Verify full end-to-end mathematical validation for the real Indian GST invoice with round-off, 99% discounts, and CGST+SGST."""
        payload = {
            "subtotal": 5815.17,
            "tax_amount": 1046.72,
            "tax_rate": 0.18,
            "total_amount": 6862.00,
            "additional_fields": {
                "round_off": 0.11,
                "cgst_amount": 523.36,
                "sgst_amount": 523.36,
            },
            "line_items": [
                {
                    "item_code": "34029011",
                    "description": "SAFED 800GM (24PKT) 68/-",
                    "quantity": 48.0,
                    "unit_price": 52.08,
                    "total_price": 2499.84,
                    "source_snippet": "1 SAFED 800GM (24PKT) 68/- 34029011 48 PCS 61.45 52.08 PCS 2499.84",
                },
                {
                    "item_code": "34029011",
                    "description": "SAFED WHITE DP 140GM 60PCS Twin Pack 10/-",
                    "quantity": 6.0,
                    "unit_price": 7.44,
                    "total_price": 0.45,
                    "source_snippet": "2 SAFED WHITE DP 140GM 60PCS Twin Pack 10/- 34029011 6 PCS 8.78 7.44 PCS 99 % 0.45",
                },
                {
                    "item_code": "34029011",
                    "description": "SAFED 2 KG (12 PKT) PRINTED BUCKET+LID 230/-",
                    "quantity": 12.0,
                    "unit_price": 172.05,
                    "total_price": 2064.60,
                    "source_snippet": "3 SAFED 2 KG (12 PKT) PRINTED BUCKET+LID 230/- 34029011 12 PCS 207.69 172.05 PCS 2064.60",
                },
                {
                    "item_code": "34029011",
                    "description": "SAFED WHITE DP 140GM 60PCS Twin Pack 10/-",
                    "quantity": 120.0,
                    "unit_price": 7.44,
                    "total_price": 892.80,
                    "source_snippet": "4 SAFED WHITE DP 140GM 60PCS Twin Pack 10/- 34029011 120 PCS 8.78 7.44 PCS 892.80",
                },
                {
                    "item_code": "34029011",
                    "description": "SAFED WHITE DP 140GM 60PCS Twin Pack 10/-",
                    "quantity": 8.0,
                    "unit_price": 7.44,
                    "total_price": 0.60,
                    "source_snippet": "5 SAFED WHITE DP 140GM 60PCS Twin Pack 10/- 34029011 8 PCS 8.78 7.44 PCS 99 % 0.60",
                },
                {
                    "item_code": "34054000",
                    "description": "SPARKLE 200 GM BATI 48PCS (9.6 KG) WITH SCRUB PAD 20/-",
                    "quantity": 24.0,
                    "unit_price": 14.87,
                    "total_price": 356.88,
                    "source_snippet": "6 SPARKLE 200 GM BATI 48PCS (9.6 KG) WITH SCRUB PAD 20/- 34054000 24 PCS 17.55 14.87 PCS 356.88",
                },
            ],
        }

        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.total_checks, 9)
        self.assertEqual(result.summary.passed_checks, 9)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.not_applicable_checks, 0)
        self.assertEqual(len(result.errors), 0)

    def test_generic_tax_exclusive_standard_invoice(self):
        """Generic tax-exclusive invoice with shipping and discount."""
        payload = {
            "pricing_type": "tax_exclusive",
            "subtotal": 200.00,
            "tax_rate": 0.10,
            "tax_amount": 20.00,
            "shipping_amount": 15.00,
            "discount_amount": 10.00,
            "total_amount": 225.00,
            "line_items": [
                {"quantity": 2.0, "unit_price": 50.00, "total_price": 100.00},
                {"quantity": 1.0, "unit_price": 100.00, "total_price": 100.00},
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        total_chk = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(total_chk.status, ValidationStatus.PASS)
        tax_chk = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(tax_chk.status, ValidationStatus.PASS)

    def test_generic_tax_inclusive_invoice(self):
        """Generic tax-inclusive invoice with back-calculated tax."""
        payload = {
            "pricing_type": "tax_inclusive",
            "total_amount": 110.00,
            "subtotal": 110.00,
            "tax_rate": 0.10,
            "tax_amount": 10.00,
            "line_items": [
                {"description": "Product A", "quantity": 1.0, "unit_price": 110.00, "total_price": 110.00, "is_tax_inclusive": True}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        tax_chk = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(tax_chk.status, ValidationStatus.PASS)
        self.assertIn("tax-inclusive", tax_chk.formula)

    def test_generic_zero_tax_exempt_invoice(self):
        """Generic zero-tax / exempt invoice."""
        payload = {
            "subtotal": 350.00,
            "tax_amount": 0.00,
            "tax_rate": 0.00,
            "total_amount": 350.00,
            "line_items": [
                {"quantity": 7.0, "unit_price": 50.00, "total_price": 350.00}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        tax_chk = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(tax_chk.status, ValidationStatus.PASS)
        self.assertIn("tax-exempt", tax_chk.formula)

    def test_generic_line_discount_summary_not_double_deducted(self):
        """Line items have individual discounts, and invoice discount_amount is an informational summary."""
        payload = {
            "subtotal": 90.00,
            "tax_amount": 9.00,
            "tax_rate": 0.10,
            "discount_amount": 10.00,
            "total_amount": 99.00,
            "line_items": [
                {
                    "quantity": 1.0,
                    "unit_price": 100.00,
                    "discount_percent": 10.0,
                    "total_price": 90.00,
                }
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        total_chk = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(total_chk.status, ValidationStatus.PASS)

    def test_generic_order_level_discount_deducted_at_total(self):
        """Line items are undiscounted, and discount is applied at order level."""
        payload = {
            "subtotal": 100.00,
            "tax_amount": 10.00,
            "tax_rate": 0.10,
            "discount_amount": 20.00,
            "total_amount": 90.00,
            "line_items": [
                {
                    "quantity": 2.0,
                    "unit_price": 50.00,
                    "total_price": 100.00,
                }
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        total_chk = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(total_chk.status, ValidationStatus.PASS)

    def test_generic_invoice_with_statutory_cess_and_igst(self):
        """Statutory multi-component tax breakdown (IGST + CESS) with round-off."""
        payload = {
            "subtotal": 10000.00,
            "tax_components": [
                {"name": "igst", "rate": 0.28, "amount": 2800.00},
                {"name": "cess", "rate": 0.12, "amount": 1200.00},
            ],
            "tax_amount": 4000.00,
            "round_off_amount": -0.50,
            "total_amount": 13999.50,
            "line_items": [
                {"quantity": 10.0, "unit_price": 1000.00, "total_price": 10000.00}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        tax_chk = next(c for c in result.checks if c.rule_name == "invoice_tax_calculation")
        self.assertEqual(tax_chk.status, ValidationStatus.PASS)
        self.assertEqual(tax_chk.calculated_value, 4000.00)

    def test_generic_underspecified_invoice_returns_not_applicable_no_guessing(self):
        """Missing required core fields returns NOT_APPLICABLE without guessing or false failure."""
        payload = {
            "invoice_number": "INV-UNKNOWN",
            "total_amount": 500.0,
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertGreater(result.summary.not_applicable_checks, 0)

    def test_generic_arithmetic_error_genuine_failure(self):
        """Genuine reconciliation error correctly marked as FAILED."""
        payload = {
            "subtotal": 1000.0,
            "tax_amount": 100.0,
            "total_amount": 1500.0,
            "line_items": [
                {"quantity": 5.0, "unit_price": 200.0, "total_price": 1000.0}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertFalse(result.is_valid)
        self.assertGreater(result.summary.failed_checks, 0)
        total_chk = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(total_chk.status, ValidationStatus.FAILED)

    def test_generic_line_item_genuine_arithmetic_error(self):
        """Line item with incorrect multiplication correctly marked as FAILED."""
        payload = {
            "subtotal": 500.0,
            "total_amount": 500.0,
            "line_items": [
                {"quantity": 10.0, "unit_price": 20.0, "total_price": 500.0}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertFalse(result.is_valid)
        self.assertGreater(result.summary.failed_checks, 0)
        item_chk = next(c for c in result.checks if c.rule_name == "invoice_line_item_1_math")
        self.assertEqual(item_chk.status, ValidationStatus.FAILED)

    def test_invoice_missing_values_yields_not_applicable(self):
        # Invoice with total, but no line items and no subtotal
        payload = {
            "invoice_number": "INV-001",
            "total_amount": 500.0,
            "subtotal": None,
            "line_items": [],
        }

        result = validate_financial_data("invoice", payload)

        self.assertTrue(result.is_valid)  # No checks FAILED
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertGreater(result.summary.not_applicable_checks, 0)

        # Check total reconciliation is NOT_APPLICABLE because subtotal is missing
        total_check = next(c for c in result.checks if c.rule_name == "invoice_total_reconciliation")
        self.assertEqual(total_check.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("subtotal", total_check.missing_fields)

    def test_invoice_rounding_tolerance(self):
        payload = {
            "subtotal": 100.0,
            "tax_amount": 8.004,
            "total_amount": 108.00,  # Diff = 0.004
        }

        # With default tolerance 0.01: should pass
        res_pass = validate_financial_data("invoice", payload, tolerance=0.01)
        self.assertTrue(res_pass.is_valid)

        # With tight tolerance 0.001: should fail
        res_fail = validate_financial_data("invoice", payload, tolerance=0.001)
        self.assertFalse(res_fail.is_valid)


class TestBalanceSheetValidation(unittest.TestCase):
    """Unit tests for balance sheet mathematical reconciliation."""

    def test_balance_sheet_all_checks_pass(self):
        payload = {
            "company_name": "Apex Corp",
            "current_assets": 450000.0,
            "non_current_assets": 850000.0,
            "total_assets": 1300000.0,
            "current_liabilities": 200000.0,
            "non_current_liabilities": 400000.0,
            "total_liabilities": 600000.0,
            "retained_earnings": 500000.0,
            "share_capital": 200000.0,
            "total_equity": 700000.0,
            "total_liabilities_and_equity": 1300000.0,
        }

        result = validate_financial_data("balance_sheet", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 5)

    def test_balance_sheet_accounting_equation_fail(self):
        payload = {
            "total_assets": 1000000.0,
            "total_liabilities": 400000.0,
            "total_equity": 500000.0,  # 400k + 500k = 900k != 1000k
        }

        result = validate_financial_data("balance_sheet", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 900000.0)
        self.assertEqual(check.reported_value, 1000000.0)
        self.assertEqual(check.variance, -100000.0)

    def test_balance_sheet_assets_components_fail(self):
        payload = {
            "current_assets": 300000.0,
            "non_current_assets": 600000.0,
            "total_assets": 1000000.0,  # 300k + 600k = 900k != 1000k
            "total_liabilities": 500000.0,
            "total_equity": 500000.0,
        }

        result = validate_financial_data("balance_sheet", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "balance_sheet_total_assets_components")
        self.assertEqual(check.status, ValidationStatus.FAILED)

    def test_balance_sheet_missing_inputs_not_applicable(self):
        # Missing total_equity
        payload = {
            "total_assets": 1000000.0,
            "total_liabilities": 400000.0,
            "total_equity": None,
        }

        result = validate_financial_data("balance_sheet", payload)

        self.assertTrue(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(check.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("total_equity", check.missing_fields)

    def test_balance_sheet_assets_equals_liabilities_plus_equity_pass(self):
        """Standard Assets = Liabilities + Equity balance sheet produces PASS."""
        payload = {
            "total_assets": 500000.0,
            "total_liabilities": 300000.0,
            "total_equity": 200000.0,
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertTrue(result.is_valid)
        eq_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(eq_chk.status, ValidationStatus.PASS)
        self.assertEqual(eq_chk.calculated_value, 500000.0)
        self.assertEqual(eq_chk.reported_value, 500000.0)

    def test_balance_sheet_consolidated_banking_format_pass(self):
        """Consolidated / banking balance sheet with Capital & Liabilities and itemized assets produces PASS."""
        payload = {
            "company_name": "Consolidated Bank",
            "total_assets": 17995066442.0,
            "total_capital_and_liabilities": 17995066442.0,
            "line_items": [
                {"item_name": "Capital", "amount": 5512776.0, "category": "capital"},
                {"item_name": "Reserves and surplus", "amount": 2092589110.0, "category": "equity"},
                {"item_name": "Minority interest", "amount": 6327647.0, "category": "equity"},
                {"item_name": "Deposits", "amount": 13337208758.0, "category": "liability"},
                {"item_name": "Borrowings", "amount": 1776967487.0, "category": "liability"},
                {"item_name": "Other liabilities and provisions", "amount": 776460664.0, "category": "liability"},
                {"item_name": "Cash and balances with RBI", "amount": 973703555.0, "category": "asset"},
                {"item_name": "Balances with banks", "amount": 239021709.0, "category": "asset"},
                {"item_name": "Investments", "amount": 4388231117.0, "category": "asset"},
                {"item_name": "Advances", "amount": 11852835198.0, "category": "asset"},
                {"item_name": "Fixed assets", "amount": 50995631.0, "category": "asset"},
                {"item_name": "Other assets", "amount": 490279232.0, "category": "asset"},
                {"item_name": "Contingent liabilities", "amount": 9752806592.0, "category": "footnote"},
                {"item_name": "Bills for collection", "amount": 447481440.0, "category": "footnote"},
            ],
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        eq_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(eq_chk.status, ValidationStatus.PASS)
        asset_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_total_assets_components")
        self.assertEqual(asset_chk.status, ValidationStatus.PASS)
        self.assertEqual(asset_chk.calculated_value, 17995066442.0)
        liab_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_total_liabilities_and_equity_components")
        self.assertEqual(liab_chk.status, ValidationStatus.PASS)
        self.assertEqual(liab_chk.calculated_value, 17995066442.0)

    def test_balance_sheet_inconsistent_totals_fails(self):
        """Balance sheet with mismatched financing total and assets produces FAILED."""
        payload = {
            "total_assets": 1000000.0,
            "total_capital_and_liabilities": 950000.0,
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertFalse(result.is_valid)
        eq_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(eq_chk.status, ValidationStatus.FAILED)
        self.assertEqual(eq_chk.calculated_value, 950000.0)
        self.assertEqual(eq_chk.reported_value, 1000000.0)
        self.assertEqual(eq_chk.variance, -50000.0)

    def test_balance_sheet_inconsistent_asset_components_fails(self):
        """Balance sheet with asset components not summing to total_assets produces FAILED."""
        payload = {
            "total_assets": 1000.0,
            "total_capital_and_liabilities": 1000.0,
            "line_items": [
                {"item_name": "Cash", "amount": 300.0, "category": "asset"},
                {"item_name": "Investments", "amount": 600.0, "category": "asset"},
            ],
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertFalse(result.is_valid)
        asset_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_total_assets_components")
        self.assertEqual(asset_chk.status, ValidationStatus.FAILED)
        self.assertEqual(asset_chk.calculated_value, 900.0)
        self.assertEqual(asset_chk.reported_value, 1000.0)

    def test_balance_sheet_missing_semantic_inputs_not_applicable(self):
        """Balance sheet missing all numerical inputs produces NOT_APPLICABLE without guessing."""
        payload = {
            "company_name": "Incomplete Statements Ltd",
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertGreater(result.summary.not_applicable_checks, 0)
        eq_chk = next(c for c in result.checks if c.rule_name == "balance_sheet_accounting_equation")
        self.assertEqual(eq_chk.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("total_assets", eq_chk.missing_fields)


class TestProfitAndLossValidation(unittest.TestCase):
    """Unit tests for Profit & Loss mathematical reconciliation."""

    def test_pnl_all_checks_pass(self):
        payload = {
            "total_revenue": 5000000.0,
            "cost_of_goods_sold": 2000000.0,
            "gross_profit": 3000000.0,
            "operating_expenses": 1200000.0,
            "operating_income": 1800000.0,
            "interest_expense": 50000.0,
            "tax_expense": 350000.0,
            "net_income": 1400000.0,
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 3)

    def test_pnl_gross_profit_fail(self):
        payload = {
            "total_revenue": 5000000.0,
            "cost_of_goods_sold": 2000000.0,
            "gross_profit": 3500000.0,  # 5M - 2M = 3M != 3.5M
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_gross_profit_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 3000000.0)
        self.assertEqual(check.reported_value, 3500000.0)

    def test_pnl_net_income_fail(self):
        payload = {
            "operating_income": 1800000.0,
            "interest_expense": 50000.0,
            "tax_expense": 350000.0,
            "net_income": 1500000.0,  # 1.8M - 50k - 350k = 1.4M != 1.5M
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_net_income_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 1400000.0)
        self.assertEqual(check.reported_value, 1500000.0)

    def test_pnl_missing_inputs_not_applicable(self):
        # Revenue present, but COGS is missing
        payload = {
            "total_revenue": 1000000.0,
            "cost_of_goods_sold": None,
            "gross_profit": 800000.0,
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertTrue(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_gross_profit_calculation")
        self.assertEqual(check.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("cost_of_goods_sold", check.missing_fields)


class TestBankingProfitAndLossValidation(unittest.TestCase):
    """Unit tests for Banking & Statement-Level Profit & Loss reconciliation checks."""

    def test_pnl_banking_all_checks_pass(self):
        """Verify all 5 banking reconciliation checks pass when numbers match."""
        payload = {
            "additional_fields": {
                "interest_earned": 1221892915.0,
                "other_income": 248789748.0,
                "total_income": 1470682663.0,
                "interest_expended": 673340317.0,
                "operating_expenses": 367238629.0,
                "provisions_and_contingencies": 157141064.0,
                "total_expenditure": 1197720010.0,
                "net_profit_for_the_year": 272962653.0,
                "minority_interest": 423147.0,
                "consolidated_net_profit": 272539506.0,
                "balance_in_profit_and_loss_account_brought_forward": 528496075.0,
                "total_appropriations": 801035581.0,
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 5)

        # Check 1: Total Income
        c1 = next(c for c in result.checks if c.rule_name == "pnl_bank_total_income")
        self.assertEqual(c1.status, ValidationStatus.PASS)
        self.assertEqual(c1.calculated_value, 1470682663.0)
        self.assertEqual(c1.reported_value, 1470682663.0)
        self.assertEqual(c1.variance, 0.0)

        # Check 2: Total Expenditure
        c2 = next(c for c in result.checks if c.rule_name == "pnl_bank_total_expenditure")
        self.assertEqual(c2.status, ValidationStatus.PASS)
        self.assertEqual(c2.calculated_value, 1197720010.0)
        self.assertEqual(c2.reported_value, 1197720010.0)
        self.assertEqual(c2.variance, 0.0)

        # Check 3: Net Profit before Minority
        c3 = next(c for c in result.checks if c.rule_name == "pnl_bank_net_profit_before_minority")
        self.assertEqual(c3.status, ValidationStatus.PASS)
        self.assertEqual(c3.calculated_value, 272962653.0)
        self.assertEqual(c3.reported_value, 272962653.0)
        self.assertEqual(c3.variance, 0.0)

        # Check 4: Consolidated Net Profit
        c4 = next(c for c in result.checks if c.rule_name == "pnl_bank_consolidated_net_profit")
        self.assertEqual(c4.status, ValidationStatus.PASS)
        self.assertEqual(c4.calculated_value, 272539506.0)
        self.assertEqual(c4.reported_value, 272539506.0)
        self.assertEqual(c4.variance, 0.0)

        # Check 5: Appropriations Reconciliation
        c5 = next(c for c in result.checks if c.rule_name == "pnl_bank_appropriations_reconciliation")
        self.assertEqual(c5.status, ValidationStatus.PASS)
        self.assertEqual(c5.calculated_value, 801035581.0)
        self.assertEqual(c5.reported_value, 801035581.0)
        self.assertEqual(c5.variance, 0.0)

    def test_pnl_banking_income_reconciliation_fail(self):
        """Interest Earned + Other Income != Total Income fails validation."""
        payload = {
            "additional_fields": {
                "interest_earned": 1000.0,
                "other_income": 200.0,
                "total_income": 1500.0,  # 1000 + 200 = 1200 != 1500
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_total_income")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 1200.0)
        self.assertEqual(check.reported_value, 1500.0)
        self.assertEqual(check.variance, -300.0)

    def test_pnl_banking_expenditure_reconciliation_fail(self):
        """Interest Expended + Operating Expenses + Provisions != Total Expenditure fails."""
        payload = {
            "additional_fields": {
                "interest_expended": 600.0,
                "operating_expenses": 300.0,
                "provisions_and_contingencies": 100.0,
                "total_expenditure": 1200.0,  # 600 + 300 + 100 = 1000 != 1200
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_total_expenditure")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 1000.0)
        self.assertEqual(check.reported_value, 1200.0)
        self.assertEqual(check.variance, -200.0)

    def test_pnl_banking_net_profit_fail(self):
        """Total Income - Total Expenditure != Net Profit for Year fails."""
        payload = {
            "additional_fields": {
                "total_income": 1200.0,
                "total_expenditure": 1000.0,
                "net_profit_for_the_year": 300.0,  # 1200 - 1000 = 200 != 300
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_net_profit_before_minority")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 200.0)
        self.assertEqual(check.reported_value, 300.0)
        self.assertEqual(check.variance, -100.0)

    def test_pnl_banking_minority_interest_fail(self):
        """Net Profit for Year - Minority Interest != Consolidated Net Profit fails."""
        payload = {
            "additional_fields": {
                "net_profit_for_the_year": 250.0,
                "minority_interest": 10.0,
                "consolidated_net_profit": 230.0,  # 250 - 10 = 240 != 230
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_consolidated_net_profit")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 240.0)
        self.assertEqual(check.reported_value, 230.0)
        self.assertEqual(check.variance, 10.0)

    def test_pnl_banking_appropriations_fail(self):
        """Consolidated Profit + Brought Forward Profit != Total Appropriations fails."""
        payload = {
            "additional_fields": {
                "consolidated_net_profit": 240.0,
                "balance_in_profit_and_loss_account_brought_forward": 500.0,
                "total_appropriations": 800.0,  # 240 + 500 = 740 != 800
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_appropriations_reconciliation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 740.0)
        self.assertEqual(check.reported_value, 800.0)
        self.assertEqual(check.variance, -60.0)

    def test_pnl_banking_missing_inputs_not_applicable(self):
        """Missing required banking inputs evaluate to NOT_APPLICABLE and do not cause failure."""
        payload = {
            "additional_fields": {
                "interest_earned": 1000.0,
                # "other_income" is missing
                "total_income": 1200.0,
            }
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertTrue(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "pnl_bank_total_income")
        self.assertEqual(check.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("other_income", check.missing_fields)
        self.assertIsNone(check.calculated_value)

    def test_pnl_banking_from_line_items(self):
        """Verify banking checks work when extracted as structured line_items."""
        payload = {
            "line_items": [
                {"item_name": "I. Interest earned", "category": "income", "amount": 100000.0},
                {"item_name": "II. Other income", "category": "income", "amount": 25000.0},
                {"item_name": "TOTAL INCOME", "category": "income", "amount": 125000.0},
                {"item_name": "I. Interest expended", "category": "expenditure", "amount": 60000.0},
                {"item_name": "II. Operating expenses", "category": "expenditure", "amount": 30000.0},
                {"item_name": "III. Provisions and Contingencies", "category": "expenditure", "amount": 10000.0},
                {"item_name": "TOTAL EXPENDITURE", "category": "expenditure", "amount": 100000.0},
                {"item_name": "Net Profit for the year", "category": "profit", "amount": 25000.0},
                {"item_name": "Less: Minority Interest", "category": "profit", "amount": 500.0},
                {"item_name": "Consolidated Profit for the year", "category": "profit", "amount": 24500.0},
                {"item_name": "Balance in Profit and Loss Account brought forward", "category": "appropriations", "amount": 50000.0},
                {"item_name": "TOTAL", "category": "appropriations", "amount": 74500.0},
            ]
        }

        result = validate_financial_data("profit_and_loss", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.passed_checks, 5)
        self.assertEqual(result.summary.failed_checks, 0)

    def test_pnl_banking_before_minority_vs_minority_interest(self):
        """Verify 'before minority interest' is NOT misidentified as minority interest."""
        payload = {
            "total_income": 470915.93,
            "total_expenditure": 397475.76,
            "net_income": 70792.25,
            "line_items": [
                {"item_name": "Interest earned", "category": "income", "amount": 336367.43},
                {"item_name": "Other income", "category": "income", "amount": 134548.50},
                {"item_name": "Interest expended", "category": "expenditure", "amount": 183894.20},
                {"item_name": "Operating expenses", "category": "expenditure", "amount": 176605.07},
                {"item_name": "Provisions and contingencies", "category": "expenditure", "amount": 36976.49},
                {
                    "item_name": "Consolidated Net Profit for the year before Minority Interest",
                    "category": "profit",
                    "amount": 73440.17,
                },
                {
                    "item_name": "Less : Minority Interest",
                    "category": "profit",
                    "amount": 2647.92,
                },
                {
                    "item_name": "Consolidated Net Profit for the year attributable to the group",
                    "category": "profit",
                    "amount": 70792.25,
                },
            ],
        }

        result = validate_financial_data("profit_and_loss", payload)

        check_cons = next(c for c in result.checks if c.rule_name == "pnl_bank_consolidated_net_profit")
        self.assertEqual(check_cons.status, ValidationStatus.PASS)
        self.assertEqual(check_cons.input_values["minority_interest"], 2647.92)
        self.assertEqual(check_cons.input_values["net_profit_for_the_year"], 73440.17)
        self.assertEqual(check_cons.input_values["consolidated_net_profit"], 70792.25)
        self.assertEqual(check_cons.variance, 0.0)

    def test_pnl_banking_current_vs_brought_forward_consolidated_profit(self):
        """Verify 'brought forward consolidated profit' is not treated as current consolidated net profit."""
        payload = {
            "net_income": 70792.25,
            "line_items": [
                {
                    "item_name": "Consolidated Net Profit for the year before Minority Interest",
                    "category": "profit",
                    "amount": 73440.17,
                },
                {
                    "item_name": "Less : Minority Interest",
                    "category": "profit",
                    "amount": 2647.92,
                },
                {
                    "item_name": "Consolidated Net Profit for the year attributable to the group",
                    "category": "profit",
                    "amount": 70792.25,
                },
                {
                    "item_name": "Brought forward consolidated profit attributable to the group",
                    "category": "profit",
                    "amount": 150045.57,
                },
            ],
            "additional_fields": {
                "total_appropriations": 220837.82,
            },
        }

        result = validate_financial_data("profit_and_loss", payload)

        check_cons = next(c for c in result.checks if c.rule_name == "pnl_bank_consolidated_net_profit")
        self.assertEqual(check_cons.status, ValidationStatus.PASS)
        self.assertEqual(check_cons.input_values["consolidated_net_profit"], 70792.25)
        self.assertNotEqual(check_cons.input_values["consolidated_net_profit"], 150045.57)

        check_approp = next(c for c in result.checks if c.rule_name == "pnl_bank_appropriations_reconciliation")
        self.assertEqual(check_approp.status, ValidationStatus.PASS)
        self.assertEqual(check_approp.input_values["consolidated_net_profit"], 70792.25)
        self.assertEqual(check_approp.input_values["brought_forward_profit"], 150045.57)
        self.assertEqual(check_approp.variance, 0.0)

    def test_pnl_banking_appropriations_with_amalgamation(self):
        """Verify appropriations reconciliation supports addition on amalgamation when reported."""
        payload = {
            "net_income": 64062.04,
            "line_items": [
                {
                    "item_name": "Consolidated Net Profit for the year before minorities' interest",
                    "category": "profit",
                    "amount": 65446.50,
                },
                {
                    "item_name": "Less: Minority Interest",
                    "category": "profit",
                    "amount": 1384.46,
                },
                {
                    "item_name": "Consolidated Net Profit for the year attributable to the group",
                    "category": "profit",
                    "amount": 64062.04,
                },
                {
                    "item_name": "Brought forward consolidated profit attributable to the group",
                    "category": "profit",
                    "amount": 120369.35,
                },
                {
                    "item_name": "Addition on amalgamation",
                    "category": "profit",
                    "amount": 3570.10,
                },
                {
                    "item_name": "Total",
                    "category": "appropriations",
                    "amount": 188001.49,
                },
            ],
        }

        result = validate_financial_data("profit_and_loss", payload)

        check_approp = next(c for c in result.checks if c.rule_name == "pnl_bank_appropriations_reconciliation")
        self.assertEqual(check_approp.status, ValidationStatus.PASS)
        self.assertEqual(check_approp.input_values["consolidated_net_profit"], 64062.04)
        self.assertEqual(check_approp.input_values["brought_forward_profit"], 120369.35)
        self.assertEqual(check_approp.input_values["addition_on_amalgamation"], 3570.10)
        self.assertEqual(check_approp.calculated_value, 188001.49)
        self.assertEqual(check_approp.reported_value, 188001.49)
        self.assertEqual(check_approp.variance, 0.0)


class TestCashFlowValidation(unittest.TestCase):
    """Unit tests for Cash Flow Statement reconciliation."""

    def test_cash_flow_pass_with_parentheses_and_negatives(self):
        payload = {
            "net_cash_from_operating_activities": "850,000.00",
            "net_cash_from_investing_activities": "(400,000.00)",  # negative via parentheses
            "net_cash_from_financing_activities": "-150000",
            "net_change_in_cash": 300000.0,
            "beginning_cash_balance": 500000.0,
            "ending_cash_balance": 800000.0,
        }

        result = validate_financial_data("cash_flow_statement", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 2)

    def test_cash_flow_pass_with_exchange_fluctuation(self):
        """Verify Cash Flow statement passes when standard exchange fluctuation is present."""
        payload = {
            "net_cash_from_operating_activities": -168690920.0,
            "net_cash_from_investing_activities": -16169244.0,
            "net_cash_from_financing_activities": 243944969.0,
            "net_change_in_cash": 61224696.0,
            "beginning_cash_balance": 818176423.0,
            "ending_cash_balance": 879401119.0,
            "line_items": [
                {
                    "category": "Other",
                    "item_name": "Effect of exchange fluctuation on translation reserve",
                    "amount": 2139891.0,
                }
            ],
        }

        result = validate_financial_data("cash_flow_statement", payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.passed_checks, 2)
        nc_check = next(c for c in result.checks if c.rule_name == "cash_flow_net_change_calculation")
        self.assertEqual(nc_check.status, ValidationStatus.PASS)
        self.assertEqual(nc_check.variance, 0.0)

    def test_cash_flow_net_change_fail(self):
        payload = {
            "net_cash_from_operating_activities": 800000.0,
            "net_cash_from_investing_activities": -300000.0,
            "net_cash_from_financing_activities": -200000.0,
            "net_change_in_cash": 400000.0,  # 800 - 300 - 200 = 300 != 400
        }

        result = validate_financial_data("cash_flow_statement", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "cash_flow_net_change_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 300000.0)
        self.assertEqual(check.reported_value, 400000.0)

    def test_cash_flow_ending_cash_fail(self):
        payload = {
            "beginning_cash_balance": 500000.0,
            "net_change_in_cash": -100000.0,
            "ending_cash_balance": 450000.0,  # 500k - 100k = 400k != 450k
        }

        result = validate_financial_data("cash_flow_statement", payload)

        self.assertFalse(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "cash_flow_ending_cash_calculation")
        self.assertEqual(check.status, ValidationStatus.FAILED)
        self.assertEqual(check.calculated_value, 400000.0)
        self.assertEqual(check.reported_value, 450000.0)

    def test_cash_flow_missing_inputs_not_applicable(self):
        payload = {
            "net_cash_from_operating_activities": 500000.0,
            "net_cash_from_investing_activities": None,
            "net_change_in_cash": 200000.0,
        }

        result = validate_financial_data("cash_flow_statement", payload)

        self.assertTrue(result.is_valid)
        check = next(c for c in result.checks if c.rule_name == "cash_flow_net_change_calculation")
        self.assertEqual(check.status, ValidationStatus.NOT_APPLICABLE)
        self.assertIn("net_cash_from_investing_activities", check.missing_fields)


class TestValidationGeneralAndEdgeCases(unittest.TestCase):
    """General edge cases: unsupported doc types, Pydantic inputs, env tolerance."""

    def test_unsupported_document_type(self):
        result = validate_financial_data("mortgage_statement", {"some": 123})
        self.assertFalse(result.is_valid)
        self.assertIn("Unsupported document type", result.errors[0])

    def test_tolerance_env_variable(self):
        with patch.dict(os.environ, {"FINANCIAL_VALIDATION_TOLERANCE": "0.05"}):
            self.assertEqual(FinancialValidationService.get_tolerance(), 0.05)

    def test_pydantic_model_input_direct(self):
        invoice_model = InvoiceExtractionData(
            subtotal=100.0,
            total_amount=100.0,
            line_items=[
                InvoiceLineItem(quantity=2.0, unit_price=50.0, total_price=100.0)
            ],
        )

        result = validate_financial_data("invoice", invoice_model)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.summary.failed_checks, 0)

    def test_composite_extraction_result_input(self):
        extraction_result = FinancialExtractionResult(
            document_type="profit_and_loss",
            status="SUCCESS",
            data={
                "total_revenue": 1000.0,
                "cost_of_goods_sold": 400.0,
                "gross_profit": 600.0,
            },
        )

        result = validate_financial_data("profit_and_loss", extraction_result)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.document_type, "profit_and_loss")


class TestDocumentTypeValidationIsolation(unittest.TestCase):
    """Proves that validation rules are strictly document-type specific."""

    def test_invoice_gets_invoice_checks_only(self):
        payload = {
            "subtotal": 100.0,
            "tax_amount": 10.0,
            "total_amount": 110.0,
            "line_items": [
                {"quantity": 1.0, "unit_price": 100.0, "total_price": 100.0}
            ],
        }
        result = validate_financial_data("invoice", payload)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.checks), 0)
        for c in result.checks:
            self.assertTrue(
                c.rule_name.startswith("invoice_"),
                f"Expected invoice rule, got: {c.rule_name}",
            )
            self.assertFalse(c.rule_name.startswith("balance_sheet_"))
            self.assertFalse(c.rule_name.startswith("profit_and_loss_"))
            self.assertFalse(c.rule_name.startswith("cash_flow_"))

    def test_balance_sheet_gets_balance_sheet_checks_only(self):
        payload = {
            "total_assets": 1000.0,
            "total_capital_and_liabilities": 1000.0,
            "line_items": [
                {"item_name": "Cash", "amount": 400.0, "category": "asset"},
                {"item_name": "Investments", "amount": 600.0, "category": "asset"},
            ],
        }
        result = validate_financial_data("balance_sheet", payload)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.checks), 0)
        for c in result.checks:
            self.assertTrue(
                c.rule_name.startswith("balance_sheet_"),
                f"Expected balance sheet rule, got: {c.rule_name}",
            )
            self.assertFalse(c.rule_name.startswith("invoice_"))
            self.assertFalse(c.rule_name.startswith("profit_and_loss_"))
            self.assertFalse(c.rule_name.startswith("cash_flow_"))

    def test_profit_and_loss_gets_pnl_checks_only(self):
        payload = {
            "total_revenue": 1000.0,
            "cost_of_goods_sold": 400.0,
            "gross_profit": 600.0,
        }
        result = validate_financial_data("profit_and_loss", payload)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.checks), 0)
        for c in result.checks:
            self.assertTrue(
                c.rule_name.startswith("pnl_") or c.rule_name.startswith("profit_and_loss_"),
                f"Expected P&L rule, got: {c.rule_name}",
            )
            self.assertFalse(c.rule_name.startswith("invoice_"))
            self.assertFalse(c.rule_name.startswith("balance_sheet_"))
            self.assertFalse(c.rule_name.startswith("cash_flow_"))

    def test_cash_flow_gets_cash_flow_checks_only(self):
        payload = {
            "net_cash_from_operating_activities": 500.0,
            "net_cash_from_investing_activities": -200.0,
            "net_cash_from_financing_activities": -100.0,
            "net_change_in_cash": 200.0,
        }
        result = validate_financial_data("cash_flow_statement", payload)
        self.assertTrue(result.is_valid)
        self.assertGreater(len(result.checks), 0)
        for c in result.checks:
            self.assertTrue(
                c.rule_name.startswith("cash_flow_"),
                f"Expected cash flow rule, got: {c.rule_name}",
            )
            self.assertFalse(c.rule_name.startswith("invoice_"))
            self.assertFalse(c.rule_name.startswith("balance_sheet_"))
            self.assertFalse(c.rule_name.startswith("profit_and_loss_"))

    def test_balance_sheet_typed_model_dispatches_safely_even_if_invoice_type_passed(self):
        bs_model = BalanceSheetExtractionData(
            total_assets=1000.0,
            total_liabilities_and_equity=1000.0,
        )
        result = validate_financial_data("invoice", bs_model)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.document_type, "balance_sheet")
        for c in result.checks:
            self.assertTrue(c.rule_name.startswith("balance_sheet_"))

    def test_document_processing_content_reconciliation(self):
        from app.services.document_processing import DocumentProcessingService
        bs_text = (
            "Consolidated Balance Sheet\n"
            "As at March 31, 2021\n"
            "CAPITAL AND LIABILITIES\n"
            "Capital 5,512,776\n"
            "Total 17,995,066,442\n"
            "ASSETS\n"
            "Cash 973,703,555"
        )
        reconciled = DocumentProcessingService.reconcile_document_type("invoice", bs_text)
        self.assertEqual(reconciled, "balance_sheet")


if __name__ == "__main__":
    unittest.main()
