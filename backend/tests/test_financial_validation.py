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


if __name__ == "__main__":
    unittest.main()
