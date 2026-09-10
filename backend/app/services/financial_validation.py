"""
Financial Validation Engine Service

Performs deterministic mathematical and accounting validation on extracted financial data:
- Invoices: line item math, line items subtotal sum, total reconciliation, tax rate check.
- Balance Sheets: accounting equation (Assets = Liabilities + Equity), component reconciliations.
- Profit & Loss: Revenue - COGS = Gross Profit, Operating Income, Net Income.
- Cash Flow Statements: Operating + Investing + Financing = Net Change, Ending Cash balance.

Rules:
- No LLMs are used for validation; all logic is deterministic Python.
- Missing values are NOT treated as zero; they yield status NOT_APPLICABLE with missing inputs noted.
- Robust numeric parsing handles commas, currencies, percentages, and parentheses for negatives.
- Rounding differences are evaluated against a configurable tolerance.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from app.schemas.financial_validation import (
    FinancialValidationResult,
    FinancialValidationSummary,
    ValidationCheckResult,
    ValidationStatus,
)


def parse_financial_number(val: Any) -> Optional[float]:
    """
    Parse a numeric value from financial documents into a standard float.

    Handles:
    - floats and ints directly
    - commas as thousand separators ("1,234.50")
    - currency symbols ("$", "€", "£", "¥", "₹", "USD", "EUR")
    - parentheses denoting negative values ("(400,000)" -> -400000.0)
    - percentages ("8%" -> 0.08, or parsed as decimal)
    - trailing negative signs or 'CR' notations ("1,234.50-", "500 CR")
    - null/empty/non-numeric representations ("N/A", "-", "null") -> None

    Args:
        val: Input number or formatted string.

    Returns:
        Optional[float]: Extracted numeric value, or None if unparseable or null.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None

    cleaned = val.strip()
    if not cleaned or cleaned.lower() in ("null", "none", "n/a", "na", "-", "--", "nil"):
        return None

    is_negative = False
    is_percentage = False

    # Check for parentheses: (1,234.50) or ( $1,234.50 )
    if cleaned.startswith("(") and cleaned.endswith(")"):
        is_negative = True
        cleaned = cleaned[1:-1].strip()

    # Check for trailing negative sign or 'CR' (credit notation in accounting)
    if cleaned.endswith("-"):
        is_negative = True
        cleaned = cleaned[:-1].strip()
    elif cleaned.upper().endswith("CR"):
        is_negative = True
        cleaned = cleaned[:-2].strip()

    if cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:].strip()
    elif cleaned.startswith("+"):
        cleaned = cleaned[1:].strip()

    if cleaned.endswith("%"):
        is_percentage = True
        cleaned = cleaned[:-1].strip()

    # Remove currency symbols and alphabetic characters (e.g. USD, EUR, $, €, £, ¥, ₹)
    cleaned = re.sub(r"[A-Za-z$€£¥₹\s]", "", cleaned)

    if not cleaned:
        return None

    # Handle comma and dot separator formats
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned and "." not in cleaned:
        if re.match(r"^\d{1,3}(,\d{3})+$", cleaned):
            cleaned = cleaned.replace(",", "")
        elif re.match(r"^\d+,\d{1,2}$", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    try:
        num = float(cleaned)
        if is_percentage:
            num = num / 100.0
        if is_negative:
            num = -abs(num)
        return round(num, 6)
    except (ValueError, TypeError):
        return None


class FinancialValidationService:
    """
    Deterministic mathematical validation engine for financial documents.
    """

    SUPPORTED_TYPES = {
        "invoice",
        "balance_sheet",
        "profit_and_loss",
        "cash_flow_statement",
    }

    @classmethod
    def get_tolerance(cls) -> float:
        """Retrieve default rounding tolerance from environment variable or fallback to 0.01."""
        try:
            return float(os.getenv("FINANCIAL_VALIDATION_TOLERANCE", "0.01"))
        except ValueError:
            return 0.01

    @classmethod
    def validate(
        cls,
        document_type: str,
        data: Union[Dict[str, Any], BaseModel, Any],
        tolerance: Optional[float] = None,
    ) -> FinancialValidationResult:
        """
        Execute deterministic mathematical reconciliation checks on financial data.

        Args:
            document_type: Category ('invoice', 'balance_sheet', 'profit_and_loss', 'cash_flow_statement').
            data: Dictionary or Pydantic model containing extracted financial metrics and line items.
            tolerance: Optional numerical threshold for rounding differences.

        Returns:
            FinancialValidationResult: Comprehensive validation report with individual checks and summary.
        """
        tol = tolerance if tolerance is not None else cls.get_tolerance()
        doc_type_clean = (document_type or "").strip().lower()

        # Handle extraction result composite object if passed
        payload = data
        if hasattr(payload, "data") and hasattr(payload, "document_type"):
            doc_type_clean = getattr(payload, "document_type", doc_type_clean).strip().lower()
            payload = getattr(payload, "data")

        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif not isinstance(payload, dict):
            payload = {}

        if doc_type_clean not in cls.SUPPORTED_TYPES:
            summary = FinancialValidationSummary(
                total_checks=0,
                passed_checks=0,
                failed_checks=0,
                not_applicable_checks=0,
                is_valid=False,
            )
            return FinancialValidationResult(
                document_type=doc_type_clean or "unknown",
                is_valid=False,
                summary=summary,
                checks=[],
                errors=[
                    f"Unsupported document type for validation: '{document_type}'. "
                    f"Supported types are: {', '.join(sorted(cls.SUPPORTED_TYPES))}."
                ],
                warnings=[],
            )

        checks: List[ValidationCheckResult] = []
        errors: List[str] = []
        warnings: List[str] = []

        if doc_type_clean == "invoice":
            checks.extend(cls._validate_invoice(payload, tol))
        elif doc_type_clean == "balance_sheet":
            checks.extend(cls._validate_balance_sheet(payload, tol))
        elif doc_type_clean == "profit_and_loss":
            checks.extend(cls._validate_profit_and_loss(payload, tol))
        elif doc_type_clean == "cash_flow_statement":
            checks.extend(cls._validate_cash_flow(payload, tol))

        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.status == ValidationStatus.PASS)
        failed_checks = sum(1 for c in checks if c.status == ValidationStatus.FAILED)
        not_applicable_checks = sum(
            1 for c in checks if c.status == ValidationStatus.NOT_APPLICABLE
        )

        is_valid = failed_checks == 0

        if failed_checks > 0:
            failed_names = [c.rule_name for c in checks if c.status == ValidationStatus.FAILED]
            errors.append(f"Financial validation failed {failed_checks} check(s): {', '.join(failed_names)}.")

        summary = FinancialValidationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            not_applicable_checks=not_applicable_checks,
            is_valid=is_valid,
        )

        return FinancialValidationResult(
            document_type=doc_type_clean,
            is_valid=is_valid,
            summary=summary,
            checks=checks,
            errors=errors,
            warnings=warnings,
        )

    # ---------------------------------------------------------------------------
    # 1. Invoice Validation Checks
    # ---------------------------------------------------------------------------
    @classmethod
    def _validate_invoice(
        cls, payload: Dict[str, Any], tol: float
    ) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []

        # Check 1: Individual Line Item Math (quantity * unit_price == total_price)
        raw_items = payload.get("line_items") or []
        if not raw_items:
            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_line_item_math",
                    formula="quantity * unit_price == total_price",
                    input_values={},
                    calculated_value=None,
                    reported_value=None,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message="No line items found in invoice to validate line math.",
                    missing_fields=["line_items"],
                )
            )
        else:
            for idx, item in enumerate(raw_items):
                item_dict = item if isinstance(item, dict) else item.model_dump()
                desc = item_dict.get("description") or f"Item {idx + 1}"
                qty = parse_financial_number(item_dict.get("quantity"))
                price = parse_financial_number(item_dict.get("unit_price"))
                total_price = parse_financial_number(item_dict.get("total_price"))

                input_vals = {
                    "item_index": idx + 1,
                    "description": desc,
                    "quantity": qty,
                    "unit_price": price,
                    "total_price": total_price,
                }

                missing = []
                if qty is None:
                    missing.append("quantity")
                if price is None:
                    missing.append("unit_price")
                if total_price is None:
                    missing.append("total_price")

                if missing:
                    checks.append(
                        ValidationCheckResult(
                            rule_name=f"invoice_line_item_{idx + 1}_math",
                            formula="quantity * unit_price == total_price",
                            input_values=input_vals,
                            calculated_value=None,
                            reported_value=total_price,
                            variance=None,
                            status=ValidationStatus.NOT_APPLICABLE,
                            tolerance=tol,
                            message=f"Line item '{desc}' missing required fields: {', '.join(missing)}.",
                            missing_fields=missing,
                        )
                    )
                else:
                    calc = round(qty * price, 4)
                    var = round(calc - total_price, 4)
                    status = (
                        ValidationStatus.PASS
                        if abs(var) <= tol
                        else ValidationStatus.FAILED
                    )
                    msg = (
                        f"Line item '{desc}' passed arithmetic check."
                        if status == ValidationStatus.PASS
                        else f"Line item '{desc}' arithmetic mismatch: {qty} * {price} = {calc} != reported {total_price} (variance: {var})."
                    )
                    checks.append(
                        ValidationCheckResult(
                            rule_name=f"invoice_line_item_{idx + 1}_math",
                            formula="quantity * unit_price == total_price",
                            input_values=input_vals,
                            calculated_value=calc,
                            reported_value=total_price,
                            variance=var,
                            status=status,
                            tolerance=tol,
                            message=msg,
                            missing_fields=[],
                        )
                    )

        # Check 2: Sum of Line Items Totals Reconcile with Subtotal
        subtotal = parse_financial_number(payload.get("subtotal"))
        if subtotal is None:
            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_line_items_subtotal_reconciliation",
                    formula="sum(line_items.total_price) == subtotal",
                    input_values={"subtotal": None},
                    calculated_value=None,
                    reported_value=None,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message="Missing subtotal for line items reconciliation.",
                    missing_fields=["subtotal"],
                )
            )
        elif not raw_items:
            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_line_items_subtotal_reconciliation",
                    formula="sum(line_items.total_price) == subtotal",
                    input_values={"subtotal": subtotal},
                    calculated_value=None,
                    reported_value=subtotal,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message="No line items present to reconcile against subtotal.",
                    missing_fields=["line_items"],
                )
            )
        else:
            missing_totals = []
            line_totals = []
            for idx, item in enumerate(raw_items):
                item_dict = item if isinstance(item, dict) else item.model_dump()
                t_val = parse_financial_number(item_dict.get("total_price"))
                if t_val is None:
                    missing_totals.append(f"line_items[{idx + 1}].total_price")
                else:
                    line_totals.append(t_val)

            if missing_totals:
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_line_items_subtotal_reconciliation",
                        formula="sum(line_items.total_price) == subtotal",
                        input_values={"subtotal": subtotal, "line_item_count": len(raw_items)},
                        calculated_value=None,
                        reported_value=subtotal,
                        variance=None,
                        status=ValidationStatus.NOT_APPLICABLE,
                        tolerance=tol,
                        message=f"Cannot reconcile line items with subtotal: missing {', '.join(missing_totals)}.",
                        missing_fields=missing_totals,
                    )
                )
            else:
                calc_subtotal = round(sum(line_totals), 4)
                var = round(calc_subtotal - subtotal, 4)
                status = (
                    ValidationStatus.PASS
                    if abs(var) <= tol
                    else ValidationStatus.FAILED
                )
                msg = (
                    f"Sum of {len(line_totals)} line item(s) matches subtotal {subtotal}."
                    if status == ValidationStatus.PASS
                    else f"Sum of line items ({calc_subtotal}) does not match subtotal {subtotal} (variance: {var})."
                )
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_line_items_subtotal_reconciliation",
                        formula="sum(line_items.total_price) == subtotal",
                        input_values={
                            "subtotal": subtotal,
                            "line_item_totals": line_totals,
                            "line_item_count": len(line_totals),
                        },
                        calculated_value=calc_subtotal,
                        reported_value=subtotal,
                        variance=var,
                        status=status,
                        tolerance=tol,
                        message=msg,
                        missing_fields=[],
                    )
                )

        # Check 3: Invoice Total Reconciliation (subtotal + tax - discount + shipping == total_amount)
        tax = parse_financial_number(payload.get("tax_amount"))
        discount = parse_financial_number(payload.get("discount_amount"))
        shipping = parse_financial_number(payload.get("shipping_amount"))
        total = parse_financial_number(payload.get("total_amount"))

        inputs_total = {
            "subtotal": subtotal,
            "tax_amount": tax,
            "discount_amount": discount,
            "shipping_amount": shipping,
            "total_amount": total,
        }

        if total is None or subtotal is None:
            missing_req = []
            if subtotal is None:
                missing_req.append("subtotal")
            if total is None:
                missing_req.append("total_amount")

            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_total_reconciliation",
                    formula="subtotal + tax_amount - discount_amount + shipping_amount == total_amount",
                    input_values=inputs_total,
                    calculated_value=None,
                    reported_value=total,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Missing core inputs for total reconciliation: {', '.join(missing_req)}.",
                    missing_fields=missing_req,
                )
            )
        else:
            # If subtotal and total differ, but no adjusting fields (tax, discount, shipping) are present
            if abs(subtotal - total) > tol and tax is None and discount is None and shipping is None:
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_total_reconciliation",
                        formula="subtotal + tax_amount - discount_amount + shipping_amount == total_amount",
                        input_values=inputs_total,
                        calculated_value=None,
                        reported_value=total,
                        variance=None,
                        status=ValidationStatus.NOT_APPLICABLE,
                        tolerance=tol,
                        message="Subtotal and total amount differ, but adjusting fields (tax, discount, shipping) are not provided.",
                        missing_fields=["tax_amount"],
                    )
                )
            else:
                calc_total = round(
                    subtotal + (tax or 0.0) - (discount or 0.0) + (shipping or 0.0), 4
                )
                var = round(calc_total - total, 4)
                status = (
                    ValidationStatus.PASS
                    if abs(var) <= tol
                    else ValidationStatus.FAILED
                )
                msg = (
                    f"Invoice total reconciliation passed ({calc_total} == {total})."
                    if status == ValidationStatus.PASS
                    else f"Invoice total reconciliation failed: calculated {calc_total} != reported {total} (variance: {var})."
                )
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_total_reconciliation",
                        formula="subtotal + tax_amount - discount_amount + shipping_amount == total_amount",
                        input_values=inputs_total,
                        calculated_value=calc_total,
                        reported_value=total,
                        variance=var,
                        status=status,
                        tolerance=tol,
                        message=msg,
                        missing_fields=[],
                    )
                )

        # Check 4: Tax Calculation Check (subtotal * tax_rate == tax_amount)
        tax_rate = parse_financial_number(payload.get("tax_rate"))
        if tax_rate is not None or (tax is not None and subtotal is not None):
            missing_tax = []
            if subtotal is None:
                missing_tax.append("subtotal")
            if tax_rate is None:
                missing_tax.append("tax_rate")
            if tax is None:
                missing_tax.append("tax_amount")

            if missing_tax:
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_tax_calculation",
                        formula="subtotal * tax_rate == tax_amount",
                        input_values={"subtotal": subtotal, "tax_rate": tax_rate, "tax_amount": tax},
                        calculated_value=None,
                        reported_value=tax,
                        variance=None,
                        status=ValidationStatus.NOT_APPLICABLE,
                        tolerance=tol,
                        message=f"Cannot compute tax check: missing {', '.join(missing_tax)}.",
                        missing_fields=missing_tax,
                    )
                )
            else:
                calc_tax = round(subtotal * tax_rate, 4)
                var = round(calc_tax - tax, 4)
                status = (
                    ValidationStatus.PASS
                    if abs(var) <= tol
                    else ValidationStatus.FAILED
                )
                msg = (
                    f"Tax calculation passed ({subtotal} * {tax_rate} == {tax})."
                    if status == ValidationStatus.PASS
                    else f"Tax calculation mismatch: {subtotal} * {tax_rate} = {calc_tax} != reported tax {tax} (variance: {var})."
                )
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_tax_calculation",
                        formula="subtotal * tax_rate == tax_amount",
                        input_values={"subtotal": subtotal, "tax_rate": tax_rate, "tax_amount": tax},
                        calculated_value=calc_tax,
                        reported_value=tax,
                        variance=var,
                        status=status,
                        tolerance=tol,
                        message=msg,
                        missing_fields=[],
                    )
                )

        return checks

    # ---------------------------------------------------------------------------
    # 2. Balance Sheet Validation Checks
    # ---------------------------------------------------------------------------
    @classmethod
    def _validate_balance_sheet(
        cls, payload: Dict[str, Any], tol: float
    ) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []

        total_assets = parse_financial_number(payload.get("total_assets"))
        total_liab = parse_financial_number(payload.get("total_liabilities"))
        total_equity = parse_financial_number(payload.get("total_equity"))
        total_liab_eq = parse_financial_number(payload.get("total_liabilities_and_equity"))

        # Check 1: Fundamental Accounting Equation (Total Assets == Total Liabilities + Total Equity)
        missing_acct = []
        if total_assets is None:
            missing_acct.append("total_assets")
        if total_liab is None:
            missing_acct.append("total_liabilities")
        if total_equity is None:
            missing_acct.append("total_equity")

        if missing_acct:
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_accounting_equation",
                    formula="total_liabilities + total_equity == total_assets",
                    input_values={
                        "total_assets": total_assets,
                        "total_liabilities": total_liab,
                        "total_equity": total_equity,
                    },
                    calculated_value=None,
                    reported_value=total_assets,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Fundamental accounting equation requires: {', '.join(missing_acct)}.",
                    missing_fields=missing_acct,
                )
            )
        else:
            calc_assets = round(total_liab + total_equity, 4)
            var = round(calc_assets - total_assets, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Accounting equation balanced: Liabilities ({total_liab}) + Equity ({total_equity}) = Assets ({total_assets})."
                if status == ValidationStatus.PASS
                else f"Accounting equation out of balance: {total_liab} + {total_equity} = {calc_assets} != Total Assets {total_assets} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_accounting_equation",
                    formula="total_liabilities + total_equity == total_assets",
                    input_values={
                        "total_assets": total_assets,
                        "total_liabilities": total_liab,
                        "total_equity": total_equity,
                    },
                    calculated_value=calc_assets,
                    reported_value=total_assets,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 2: Liabilities and Equity Component Reconciliation (where total_liabilities_and_equity is reported)
        if total_liab_eq is not None or (total_liab is not None and total_equity is not None):
            missing_le = []
            if total_liab is None:
                missing_le.append("total_liabilities")
            if total_equity is None:
                missing_le.append("total_equity")
            if total_liab_eq is None:
                missing_le.append("total_liabilities_and_equity")

            if missing_le:
                checks.append(
                    ValidationCheckResult(
                        rule_name="balance_sheet_liabilities_and_equity_reconciliation",
                        formula="total_liabilities + total_equity == total_liabilities_and_equity",
                        input_values={
                            "total_liabilities": total_liab,
                            "total_equity": total_equity,
                            "total_liabilities_and_equity": total_liab_eq,
                        },
                        calculated_value=None,
                        reported_value=total_liab_eq,
                        variance=None,
                        status=ValidationStatus.NOT_APPLICABLE,
                        tolerance=tol,
                        message=f"Missing inputs for liabilities and equity reconciliation: {', '.join(missing_le)}.",
                        missing_fields=missing_le,
                    )
                )
            else:
                calc_le = round(total_liab + total_equity, 4)
                var = round(calc_le - total_liab_eq, 4)
                status = (
                    ValidationStatus.PASS
                    if abs(var) <= tol
                    else ValidationStatus.FAILED
                )
                msg = (
                    f"Total liabilities and equity reconciliation passed ({calc_le} == {total_liab_eq})."
                    if status == ValidationStatus.PASS
                    else f"Liabilities and equity reconciliation mismatch: {total_liab} + {total_equity} = {calc_le} != reported {total_liab_eq} (variance: {var})."
                )
                checks.append(
                    ValidationCheckResult(
                        rule_name="balance_sheet_liabilities_and_equity_reconciliation",
                        formula="total_liabilities + total_equity == total_liabilities_and_equity",
                        input_values={
                            "total_liabilities": total_liab,
                            "total_equity": total_equity,
                            "total_liabilities_and_equity": total_liab_eq,
                        },
                        calculated_value=calc_le,
                        reported_value=total_liab_eq,
                        variance=var,
                        status=status,
                        tolerance=tol,
                        message=msg,
                        missing_fields=[],
                    )
                )

        # Check 3: Current + Non-Current Assets == Total Assets
        curr_assets = parse_financial_number(payload.get("current_assets"))
        non_curr_assets = parse_financial_number(payload.get("non_current_assets"))

        missing_assets = []
        if curr_assets is None:
            missing_assets.append("current_assets")
        if non_curr_assets is None:
            missing_assets.append("non_current_assets")
        if total_assets is None:
            missing_assets.append("total_assets")

        if missing_assets:
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_assets_components",
                    formula="current_assets + non_current_assets == total_assets",
                    input_values={
                        "current_assets": curr_assets,
                        "non_current_assets": non_curr_assets,
                        "total_assets": total_assets,
                    },
                    calculated_value=None,
                    reported_value=total_assets,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Assets breakdown check missing required components: {', '.join(missing_assets)}.",
                    missing_fields=missing_assets,
                )
            )
        else:
            calc_ta = round(curr_assets + non_curr_assets, 4)
            var = round(calc_ta - total_assets, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Assets components sum correctly ({curr_assets} + {non_curr_assets} == {total_assets})."
                if status == ValidationStatus.PASS
                else f"Assets components mismatch: {curr_assets} + {non_curr_assets} = {calc_ta} != {total_assets} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_assets_components",
                    formula="current_assets + non_current_assets == total_assets",
                    input_values={
                        "current_assets": curr_assets,
                        "non_current_assets": non_curr_assets,
                        "total_assets": total_assets,
                    },
                    calculated_value=calc_ta,
                    reported_value=total_assets,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 4: Current + Non-Current Liabilities == Total Liabilities
        curr_liab = parse_financial_number(payload.get("current_liabilities"))
        non_curr_liab = parse_financial_number(payload.get("non_current_liabilities"))

        missing_liab = []
        if curr_liab is None:
            missing_liab.append("current_liabilities")
        if non_curr_liab is None:
            missing_liab.append("non_current_liabilities")
        if total_liab is None:
            missing_liab.append("total_liabilities")

        if missing_liab:
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_liabilities_components",
                    formula="current_liabilities + non_current_liabilities == total_liabilities",
                    input_values={
                        "current_liabilities": curr_liab,
                        "non_current_liabilities": non_curr_liab,
                        "total_liabilities": total_liab,
                    },
                    calculated_value=None,
                    reported_value=total_liab,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Liabilities breakdown check missing required components: {', '.join(missing_liab)}.",
                    missing_fields=missing_liab,
                )
            )
        else:
            calc_tl = round(curr_liab + non_curr_liab, 4)
            var = round(calc_tl - total_liab, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Liabilities components sum correctly ({curr_liab} + {non_curr_liab} == {total_liab})."
                if status == ValidationStatus.PASS
                else f"Liabilities components mismatch: {curr_liab} + {non_curr_liab} = {calc_tl} != {total_liab} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_liabilities_components",
                    formula="current_liabilities + non_current_liabilities == total_liabilities",
                    input_values={
                        "current_liabilities": curr_liab,
                        "non_current_liabilities": non_curr_liab,
                        "total_liabilities": total_liab,
                    },
                    calculated_value=calc_tl,
                    reported_value=total_liab,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 5: Retained Earnings + Share Capital == Total Equity (where both components exist)
        retained = parse_financial_number(payload.get("retained_earnings"))
        share_cap = parse_financial_number(payload.get("share_capital"))

        missing_equity = []
        if retained is None:
            missing_equity.append("retained_earnings")
        if share_cap is None:
            missing_equity.append("share_capital")
        if total_equity is None:
            missing_equity.append("total_equity")

        if missing_equity:
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_equity_components",
                    formula="retained_earnings + share_capital == total_equity",
                    input_values={
                        "retained_earnings": retained,
                        "share_capital": share_cap,
                        "total_equity": total_equity,
                    },
                    calculated_value=None,
                    reported_value=total_equity,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Equity breakdown check missing components: {', '.join(missing_equity)}.",
                    missing_fields=missing_equity,
                )
            )
        else:
            calc_eq = round(retained + share_cap, 4)
            var = round(calc_eq - total_equity, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Equity components sum correctly ({retained} + {share_cap} == {total_equity})."
                if status == ValidationStatus.PASS
                else f"Equity components mismatch: {retained} + {share_cap} = {calc_eq} != {total_equity} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_equity_components",
                    formula="retained_earnings + share_capital == total_equity",
                    input_values={
                        "retained_earnings": retained,
                        "share_capital": share_cap,
                        "total_equity": total_equity,
                    },
                    calculated_value=calc_eq,
                    reported_value=total_equity,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        return checks

    @classmethod
    def _find_line_item_amount(
        cls,
        line_items: List[Any],
        keywords: List[str],
        category_keyword: Optional[str] = None,
        exclude_keywords: Optional[List[str]] = None,
    ) -> Optional[float]:
        """Find line item amount matching keywords, optional category, and excluding disallowed keywords."""
        if not line_items:
            return None
        for item in line_items:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item
            else:
                continue
            name = str(item_dict.get("item_name") or item_dict.get("description") or "").lower()
            cat = str(item_dict.get("category") or "").lower()
            if category_keyword and category_keyword.lower() not in cat:
                continue
            if exclude_keywords and any(ex.lower() in name for ex in exclude_keywords):
                continue
            if any(kw.lower() in name for kw in keywords):
                amt = parse_financial_number(item_dict.get("amount") or item_dict.get("total_price"))
                if amt is not None:
                    return amt
        return None

    @classmethod
    def _find_additional_field_amount(
        cls,
        additional_fields: Dict[str, Any],
        exact_keys: List[str],
    ) -> Optional[float]:
        """Find amount from additional_fields dict matching key names or normalized variants."""
        if not isinstance(additional_fields, dict):
            return None
        for k in exact_keys:
            if k in additional_fields:
                val = parse_financial_number(additional_fields[k])
                if val is not None:
                    return val
        norm_map = {str(k).lower().replace("-", "_").replace(" ", "_"): v for k, v in additional_fields.items()}
        for k in exact_keys:
            k_norm = k.lower().replace("-", "_").replace(" ", "_")
            if k_norm in norm_map:
                val = parse_financial_number(norm_map[k_norm])
                if val is not None:
                    return val
        return None

    # ---------------------------------------------------------------------------
    # 3. Profit & Loss Validation Checks
    # ---------------------------------------------------------------------------
    @classmethod
    def _validate_profit_and_loss(
        cls, payload: Dict[str, Any], tol: float
    ) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []

        rev = parse_financial_number(payload.get("total_revenue"))
        cogs = parse_financial_number(payload.get("cost_of_goods_sold"))
        gp = parse_financial_number(payload.get("gross_profit"))
        opex = parse_financial_number(payload.get("operating_expenses"))
        op_inc = parse_financial_number(payload.get("operating_income"))
        interest = parse_financial_number(payload.get("interest_expense"))
        tax = parse_financial_number(payload.get("tax_expense"))
        net_inc = parse_financial_number(payload.get("net_income"))

        # Check 1: Revenue - COGS == Gross Profit
        missing_gp = []
        if rev is None:
            missing_gp.append("total_revenue")
        if cogs is None:
            missing_gp.append("cost_of_goods_sold")
        if gp is None:
            missing_gp.append("gross_profit")

        if missing_gp:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_gross_profit_calculation",
                    formula="total_revenue - cost_of_goods_sold == gross_profit",
                    input_values={"total_revenue": rev, "cost_of_goods_sold": cogs, "gross_profit": gp},
                    calculated_value=None,
                    reported_value=gp,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Gross profit check missing required inputs: {', '.join(missing_gp)}.",
                    missing_fields=missing_gp,
                )
            )
        else:
            calc_gp = round(rev - cogs, 4)
            var = round(calc_gp - gp, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Gross profit verified: {rev} - {cogs} = {gp}."
                if status == ValidationStatus.PASS
                else f"Gross profit mismatch: {rev} - {cogs} = {calc_gp} != reported {gp} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_gross_profit_calculation",
                    formula="total_revenue - cost_of_goods_sold == gross_profit",
                    input_values={"total_revenue": rev, "cost_of_goods_sold": cogs, "gross_profit": gp},
                    calculated_value=calc_gp,
                    reported_value=gp,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 2: Gross Profit - Operating Expenses == Operating Income
        missing_oi = []
        if gp is None:
            missing_oi.append("gross_profit")
        if opex is None:
            missing_oi.append("operating_expenses")
        if op_inc is None:
            missing_oi.append("operating_income")

        if missing_oi:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_operating_income_calculation",
                    formula="gross_profit - operating_expenses == operating_income",
                    input_values={"gross_profit": gp, "operating_expenses": opex, "operating_income": op_inc},
                    calculated_value=None,
                    reported_value=op_inc,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Operating income check missing required inputs: {', '.join(missing_oi)}.",
                    missing_fields=missing_oi,
                )
            )
        else:
            calc_oi = round(gp - opex, 4)
            var = round(calc_oi - op_inc, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Operating income verified: {gp} - {opex} = {op_inc}."
                if status == ValidationStatus.PASS
                else f"Operating income mismatch: {gp} - {opex} = {calc_oi} != reported {op_inc} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_operating_income_calculation",
                    formula="gross_profit - operating_expenses == operating_income",
                    input_values={"gross_profit": gp, "operating_expenses": opex, "operating_income": op_inc},
                    calculated_value=calc_oi,
                    reported_value=op_inc,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 3: Operating Income - Interest Expense - Tax Expense == Net Income
        missing_ni = []
        if op_inc is None:
            missing_ni.append("operating_income")
        if interest is None:
            missing_ni.append("interest_expense")
        if tax is None:
            missing_ni.append("tax_expense")
        if net_inc is None:
            missing_ni.append("net_income")

        if missing_ni:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_net_income_calculation",
                    formula="operating_income - interest_expense - tax_expense == net_income",
                    input_values={
                        "operating_income": op_inc,
                        "interest_expense": interest,
                        "tax_expense": tax,
                        "net_income": net_inc,
                    },
                    calculated_value=None,
                    reported_value=net_inc,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Net income check missing required inputs: {', '.join(missing_ni)}.",
                    missing_fields=missing_ni,
                )
            )
        else:
            calc_ni = round(op_inc - interest - tax, 4)
            var = round(calc_ni - net_inc, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Net income verified: {op_inc} - {interest} - {tax} = {net_inc}."
                if status == ValidationStatus.PASS
                else f"Net income mismatch: {op_inc} - {interest} - {tax} = {calc_ni} != reported {net_inc} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_net_income_calculation",
                    formula="operating_income - interest_expense - tax_expense == net_income",
                    input_values={
                        "operating_income": op_inc,
                        "interest_expense": interest,
                        "tax_expense": tax,
                        "net_income": net_inc,
                    },
                    calculated_value=calc_ni,
                    reported_value=net_inc,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # -----------------------------------------------------------------------
        # Banking & Statement-Level P&L Reconciliation Checks
        # -----------------------------------------------------------------------
        raw_items = payload.get("line_items") or []
        add_fields = payload.get("additional_fields") or {}

        # Check 4 (Banking): Interest Earned + Other Income == Total Income
        interest_earned = (
            cls._find_additional_field_amount(add_fields, ["interest_earned", "interest_income"])
            or cls._find_line_item_amount(raw_items, ["interest earned", "interest income"], category_keyword="income")
            or cls._find_line_item_amount(raw_items, ["interest earned", "interest income"])
        )
        other_income = (
            cls._find_additional_field_amount(add_fields, ["other_income", "non_interest_income"])
            or cls._find_line_item_amount(raw_items, ["other income", "non-interest income", "non interest income"], category_keyword="income")
            or cls._find_line_item_amount(raw_items, ["other income", "non-interest income", "non interest income"])
        )
        total_income = (
            parse_financial_number(payload.get("total_revenue"))
            or cls._find_additional_field_amount(add_fields, ["total_income", "total_revenue"])
            or cls._find_line_item_amount(raw_items, ["total income", "total revenue"])
        )

        missing_bank_income = []
        if interest_earned is None:
            missing_bank_income.append("interest_earned")
        if other_income is None:
            missing_bank_income.append("other_income")
        if total_income is None:
            missing_bank_income.append("total_income")

        if missing_bank_income:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_total_income",
                    formula="interest_earned + other_income == total_income",
                    input_values={
                        "interest_earned": interest_earned,
                        "other_income": other_income,
                        "total_income": total_income,
                    },
                    calculated_value=None,
                    reported_value=total_income,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Banking total income check missing required inputs: {', '.join(missing_bank_income)}.",
                    missing_fields=missing_bank_income,
                )
            )
        else:
            calc_income = round(interest_earned + other_income, 4)
            var_income = round(calc_income - total_income, 4)
            status_income = (
                ValidationStatus.PASS if abs(var_income) <= tol else ValidationStatus.FAILED
            )
            msg_income = (
                f"Banking total income verified: {interest_earned} + {other_income} = {total_income}."
                if status_income == ValidationStatus.PASS
                else f"Banking total income mismatch: {interest_earned} + {other_income} = {calc_income} != reported {total_income} (variance: {var_income})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_total_income",
                    formula="interest_earned + other_income == total_income",
                    input_values={
                        "interest_earned": interest_earned,
                        "other_income": other_income,
                        "total_income": total_income,
                    },
                    calculated_value=calc_income,
                    reported_value=total_income,
                    variance=var_income,
                    status=status_income,
                    tolerance=tol,
                    message=msg_income,
                    missing_fields=[],
                )
            )

        # Check 5 (Banking): Interest Expended + Operating Expenses + Provisions & Contingencies == Total Expenditure
        interest_expended = (
            parse_financial_number(payload.get("interest_expense"))
            or cls._find_additional_field_amount(add_fields, ["interest_expended", "interest_expense"])
            or cls._find_line_item_amount(raw_items, ["interest expended", "interest expense"], category_keyword="expenditure")
            or cls._find_line_item_amount(raw_items, ["interest expended", "interest expense"])
        )
        operating_exp = (
            parse_financial_number(payload.get("operating_expenses"))
            or cls._find_additional_field_amount(add_fields, ["operating_expenses", "operating_expense"])
            or cls._find_line_item_amount(raw_items, ["operating expenses", "operating expense"], category_keyword="expenditure")
            or cls._find_line_item_amount(raw_items, ["operating expenses", "operating expense"])
        )
        provisions = (
            cls._find_additional_field_amount(add_fields, ["provisions_and_contingencies", "provisions_contingencies", "provisions"])
            or cls._find_line_item_amount(raw_items, ["provisions and contingencies", "provisions & contingencies", "provisions"], category_keyword="expenditure")
            or cls._find_line_item_amount(raw_items, ["provisions and contingencies", "provisions & contingencies", "provisions"])
        )
        total_expenditure = (
            cls._find_additional_field_amount(add_fields, ["total_expenditure", "total_expenses", "total_expenditures"])
            or cls._find_line_item_amount(raw_items, ["total expenditure", "total expenses", "total"], category_keyword="expenditure")
            or cls._find_line_item_amount(raw_items, ["total expenditure", "total expenses"])
            or parse_financial_number(payload.get("total_expenditure"))
        )

        missing_bank_exp = []
        if interest_expended is None:
            missing_bank_exp.append("interest_expended")
        if operating_exp is None:
            missing_bank_exp.append("operating_expenses")
        if provisions is None:
            missing_bank_exp.append("provisions_and_contingencies")
        if total_expenditure is None:
            missing_bank_exp.append("total_expenditure")

        if missing_bank_exp:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_total_expenditure",
                    formula="interest_expended + operating_expenses + provisions_and_contingencies == total_expenditure",
                    input_values={
                        "interest_expended": interest_expended,
                        "operating_expenses": operating_exp,
                        "provisions_and_contingencies": provisions,
                        "total_expenditure": total_expenditure,
                    },
                    calculated_value=None,
                    reported_value=total_expenditure,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Banking total expenditure check missing required inputs: {', '.join(missing_bank_exp)}.",
                    missing_fields=missing_bank_exp,
                )
            )
        else:
            calc_exp = round(interest_expended + operating_exp + provisions, 4)
            var_exp = round(calc_exp - total_expenditure, 4)
            status_exp = (
                ValidationStatus.PASS if abs(var_exp) <= tol else ValidationStatus.FAILED
            )
            msg_exp = (
                f"Banking total expenditure verified: {interest_expended} + {operating_exp} + {provisions} = {total_expenditure}."
                if status_exp == ValidationStatus.PASS
                else f"Banking total expenditure mismatch: {interest_expended} + {operating_exp} + {provisions} = {calc_exp} != reported {total_expenditure} (variance: {var_exp})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_total_expenditure",
                    formula="interest_expended + operating_expenses + provisions_and_contingencies == total_expenditure",
                    input_values={
                        "interest_expended": interest_expended,
                        "operating_expenses": operating_exp,
                        "provisions_and_contingencies": provisions,
                        "total_expenditure": total_expenditure,
                    },
                    calculated_value=calc_exp,
                    reported_value=total_expenditure,
                    variance=var_exp,
                    status=status_exp,
                    tolerance=tol,
                    message=msg_exp,
                    missing_fields=[],
                )
            )

        # Check 6 (Banking): Total Income - Total Expenditure == Net Profit for the Year / Profit before Minority Interest
        net_profit_year = (
            cls._find_additional_field_amount(add_fields, [
                "net_profit_for_the_year",
                "profit_for_the_year",
                "net_profit_before_minority",
                "net_profit",
                "profit_before_minority_interest",
            ])
            or cls._find_line_item_amount(raw_items, [
                "net profit for the year",
                "profit for the year",
                "net profit before minority interest",
                "profit before minority interest"
            ], category_keyword="profit")
            or cls._find_line_item_amount(raw_items, ["net profit for the year", "profit for the year"])
        )

        missing_bank_np = []
        if total_income is None:
            missing_bank_np.append("total_income")
        if total_expenditure is None:
            missing_bank_np.append("total_expenditure")
        if net_profit_year is None:
            missing_bank_np.append("net_profit_for_the_year")

        if missing_bank_np:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_net_profit_before_minority",
                    formula="total_income - total_expenditure == net_profit_for_the_year",
                    input_values={
                        "total_income": total_income,
                        "total_expenditure": total_expenditure,
                        "net_profit_for_the_year": net_profit_year,
                    },
                    calculated_value=None,
                    reported_value=net_profit_year,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Banking net profit before minority check missing required inputs: {', '.join(missing_bank_np)}.",
                    missing_fields=missing_bank_np,
                )
            )
        else:
            calc_np = round(total_income - total_expenditure, 4)
            var_np = round(calc_np - net_profit_year, 4)
            status_np = (
                ValidationStatus.PASS if abs(var_np) <= tol else ValidationStatus.FAILED
            )
            msg_np = (
                f"Banking net profit before minority verified: {total_income} - {total_expenditure} = {net_profit_year}."
                if status_np == ValidationStatus.PASS
                else f"Banking net profit before minority mismatch: {total_income} - {total_expenditure} = {calc_np} != reported {net_profit_year} (variance: {var_np})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_net_profit_before_minority",
                    formula="total_income - total_expenditure == net_profit_for_the_year",
                    input_values={
                        "total_income": total_income,
                        "total_expenditure": total_expenditure,
                        "net_profit_for_the_year": net_profit_year,
                    },
                    calculated_value=calc_np,
                    reported_value=net_profit_year,
                    variance=var_np,
                    status=status_np,
                    tolerance=tol,
                    message=msg_np,
                    missing_fields=[],
                )
            )

        # Check 7 (Banking): Net Profit for the Year - Minority Interest == Consolidated Net Profit
        minority_interest = (
            cls._find_additional_field_amount(add_fields, ["minority_interest", "less_minority_interest", "minority_interests"])
            or cls._find_line_item_amount(
                raw_items,
                ["less : minority interest", "less: minority interest", "minority interest", "minority interests"],
                category_keyword="profit",
                exclude_keywords=["before", "before minority", "before minorities"],
            )
            or cls._find_line_item_amount(
                raw_items,
                ["less : minority interest", "less: minority interest", "minority interest", "minority interests"],
                exclude_keywords=["before", "before minority", "before minorities"],
            )
        )
        consolidated_profit = (
            cls._find_additional_field_amount(add_fields, [
                "consolidated_net_profit_for_the_year_attributable_to_the_group",
                "consolidated_profit_for_the_year_attributable_to_the_group",
                "consolidated_net_profit_attributable_to_group",
                "consolidated_profit_attributable_to_group",
                "consolidated_profit_for_the_year",
                "consolidated_net_profit",
                "consolidated_profit",
            ])
            or cls._find_line_item_amount(
                raw_items,
                [
                    "consolidated net profit for the year attributable to the group",
                    "consolidated profit for the year attributable to the group",
                    "consolidated net profit attributable to the group",
                    "consolidated profit attributable to the group",
                    "consolidated net profit for the year",
                    "consolidated profit for the year",
                    "consolidated net profit",
                    "consolidated profit",
                ],
                category_keyword="profit",
                exclude_keywords=["brought forward", "before", "before minority", "before minorities"],
            )
            or cls._find_line_item_amount(
                raw_items,
                [
                    "consolidated net profit for the year attributable to the group",
                    "consolidated profit for the year attributable to the group",
                    "consolidated net profit attributable to the group",
                    "consolidated profit attributable to the group",
                    "consolidated net profit for the year",
                    "consolidated profit for the year",
                    "consolidated net profit",
                    "consolidated profit",
                ],
                exclude_keywords=["brought forward", "before", "before minority", "before minorities"],
            )
            or parse_financial_number(payload.get("net_income"))
        )

        missing_bank_cons = []
        if net_profit_year is None:
            missing_bank_cons.append("net_profit_for_the_year")
        if minority_interest is None:
            missing_bank_cons.append("minority_interest")
        if consolidated_profit is None:
            missing_bank_cons.append("consolidated_net_profit")

        if missing_bank_cons:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_consolidated_net_profit",
                    formula="net_profit_for_the_year - minority_interest == consolidated_net_profit",
                    input_values={
                        "net_profit_for_the_year": net_profit_year,
                        "minority_interest": minority_interest,
                        "consolidated_net_profit": consolidated_profit,
                    },
                    calculated_value=None,
                    reported_value=consolidated_profit,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Banking consolidated net profit check missing required inputs: {', '.join(missing_bank_cons)}.",
                    missing_fields=missing_bank_cons,
                )
            )
        else:
            calc_cons = round(net_profit_year - minority_interest, 4)
            var_cons = round(calc_cons - consolidated_profit, 4)
            status_cons = (
                ValidationStatus.PASS if abs(var_cons) <= tol else ValidationStatus.FAILED
            )
            msg_cons = (
                f"Banking consolidated net profit verified: {net_profit_year} - {minority_interest} = {consolidated_profit}."
                if status_cons == ValidationStatus.PASS
                else f"Banking consolidated net profit mismatch: {net_profit_year} - {minority_interest} = {calc_cons} != reported {consolidated_profit} (variance: {var_cons})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_consolidated_net_profit",
                    formula="net_profit_for_the_year - minority_interest == consolidated_net_profit",
                    input_values={
                        "net_profit_for_the_year": net_profit_year,
                        "minority_interest": minority_interest,
                        "consolidated_net_profit": consolidated_profit,
                    },
                    calculated_value=calc_cons,
                    reported_value=consolidated_profit,
                    variance=var_cons,
                    status=status_cons,
                    tolerance=tol,
                    message=msg_cons,
                    missing_fields=[],
                )
            )

        # Check 8 (Banking): Consolidated Profit + Brought Forward Profit == Total Appropriations
        brought_forward = (
            cls._find_additional_field_amount(add_fields, [
                "brought_forward_consolidated_profit_attributable_to_the_group",
                "brought_forward_consolidated_profit",
                "balance_in_profit_and_loss_account_brought_forward",
                "balance_in_the_profit_and_loss_account_brought_forward",
                "balance_brought_forward",
                "profit_brought_forward",
                "brought_forward_profit",
                "brought_forward",
            ])
            or cls._find_line_item_amount(raw_items, [
                "brought forward consolidated profit attributable to the group",
                "brought forward consolidated profit",
                "balance in the profit and loss account brought forward",
                "balance in profit and loss account brought forward",
                "brought forward profit",
                "profit brought forward",
                "balance brought forward",
                "brought forward",
            ])
        )
        total_appropriations = (
            cls._find_additional_field_amount(add_fields, [
                "total_appropriations",
                "total_profit_and_brought_forward"
            ])
            or cls._find_line_item_amount(raw_items, ["total"], category_keyword="appropriations")
            or cls._find_line_item_amount(raw_items, ["total appropriations"])
        )
        amalgamation_addition = (
            cls._find_additional_field_amount(add_fields, [
                "addition_on_amalgamation",
                "amalgamation_addition",
                "amalgamation",
            ])
            or cls._find_line_item_amount(raw_items, ["addition on amalgamation", "amalgamation"], category_keyword="profit")
            or cls._find_line_item_amount(raw_items, ["addition on amalgamation", "amalgamation"])
        )

        missing_bank_approp = []
        if consolidated_profit is None:
            missing_bank_approp.append("consolidated_net_profit")
        if brought_forward is None:
            missing_bank_approp.append("brought_forward_profit")
        if total_appropriations is None:
            missing_bank_approp.append("total_appropriations")

        if missing_bank_approp:
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_appropriations_reconciliation",
                    formula="consolidated_net_profit + brought_forward_profit == total_appropriations",
                    input_values={
                        "consolidated_net_profit": consolidated_profit,
                        "brought_forward_profit": brought_forward,
                        "total_appropriations": total_appropriations,
                    },
                    calculated_value=None,
                    reported_value=total_appropriations,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Banking appropriations reconciliation check missing required inputs: {', '.join(missing_bank_approp)}.",
                    missing_fields=missing_bank_approp,
                )
            )
        else:
            if amalgamation_addition is not None:
                calc_approp = round(consolidated_profit + brought_forward + amalgamation_addition, 4)
                var_approp = round(calc_approp - total_appropriations, 4)
                formula_approp = "consolidated_net_profit + brought_forward_profit + addition_on_amalgamation == total_appropriations"
                input_approp = {
                    "consolidated_net_profit": consolidated_profit,
                    "brought_forward_profit": brought_forward,
                    "addition_on_amalgamation": amalgamation_addition,
                    "total_appropriations": total_appropriations,
                }
                msg_approp = (
                    f"Banking appropriations reconciliation verified: {consolidated_profit} + {brought_forward} + {amalgamation_addition} = {total_appropriations}."
                    if abs(var_approp) <= tol
                    else f"Banking appropriations reconciliation mismatch: {consolidated_profit} + {brought_forward} + {amalgamation_addition} = {calc_approp} != reported {total_appropriations} (variance: {var_approp})."
                )
            else:
                calc_approp = round(consolidated_profit + brought_forward, 4)
                var_approp = round(calc_approp - total_appropriations, 4)
                formula_approp = "consolidated_net_profit + brought_forward_profit == total_appropriations"
                input_approp = {
                    "consolidated_net_profit": consolidated_profit,
                    "brought_forward_profit": brought_forward,
                    "total_appropriations": total_appropriations,
                }
                msg_approp = (
                    f"Banking appropriations reconciliation verified: {consolidated_profit} + {brought_forward} = {total_appropriations}."
                    if abs(var_approp) <= tol
                    else f"Banking appropriations reconciliation mismatch: {consolidated_profit} + {brought_forward} = {calc_approp} != reported {total_appropriations} (variance: {var_approp})."
                )
            status_approp = (
                ValidationStatus.PASS if abs(var_approp) <= tol else ValidationStatus.FAILED
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="pnl_bank_appropriations_reconciliation",
                    formula=formula_approp,
                    input_values=input_approp,
                    calculated_value=calc_approp,
                    reported_value=total_appropriations,
                    variance=var_approp,
                    status=status_approp,
                    tolerance=tol,
                    message=msg_approp,
                    missing_fields=[],
                )
            )

        return checks

    # ---------------------------------------------------------------------------
    # 4. Cash Flow Statement Validation Checks
    # ---------------------------------------------------------------------------
    @classmethod
    def _validate_cash_flow(
        cls, payload: Dict[str, Any], tol: float
    ) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []

        operating = parse_financial_number(payload.get("net_cash_from_operating_activities"))
        investing = parse_financial_number(payload.get("net_cash_from_investing_activities"))
        financing = parse_financial_number(payload.get("net_cash_from_financing_activities"))
        net_change = parse_financial_number(payload.get("net_change_in_cash"))
        beginning = parse_financial_number(payload.get("beginning_cash_balance"))
        ending = parse_financial_number(payload.get("ending_cash_balance"))

        add_fields = payload.get("additional_fields") if isinstance(payload.get("additional_fields"), dict) else {}
        raw_items = payload.get("line_items") if isinstance(payload.get("line_items"), list) else []

        exchange_effect = (
            parse_financial_number(payload.get("effect_of_exchange_rate_changes"))
            or parse_financial_number(payload.get("exchange_rate_effect"))
            or cls._find_additional_field_amount(add_fields, [
                "effect_of_exchange_rate_changes",
                "effect_of_exchange_fluctuation",
                "exchange_fluctuation",
                "translation_reserve_exchange_fluctuation",
            ])
            or cls._find_line_item_amount(raw_items, [
                "effect of exchange fluctuation on translation reserve",
                "effect of exchange rate changes on cash",
                "effect of exchange fluctuation",
                "effect of exchange rate changes",
                "exchange fluctuation",
            ])
        )

        # Check 1: Operating + Investing + Financing (+ Exchange Fluctuation) == Net Change in Cash
        missing_nc = []
        if operating is None:
            missing_nc.append("net_cash_from_operating_activities")
        if investing is None:
            missing_nc.append("net_cash_from_investing_activities")
        if financing is None:
            missing_nc.append("net_cash_from_financing_activities")
        if net_change is None:
            missing_nc.append("net_change_in_cash")

        if missing_nc:
            checks.append(
                ValidationCheckResult(
                    rule_name="cash_flow_net_change_calculation",
                    formula="net_cash_from_operating_activities + net_cash_from_investing_activities + net_cash_from_financing_activities == net_change_in_cash",
                    input_values={
                        "net_cash_from_operating_activities": operating,
                        "net_cash_from_investing_activities": investing,
                        "net_cash_from_financing_activities": financing,
                        "net_change_in_cash": net_change,
                    },
                    calculated_value=None,
                    reported_value=net_change,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Net change in cash check missing required inputs: {', '.join(missing_nc)}.",
                    missing_fields=missing_nc,
                )
            )
        else:
            if exchange_effect is not None:
                calc_nc = round(operating + investing + financing + exchange_effect, 4)
                var = round(calc_nc - net_change, 4)
                formula_nc = "net_cash_from_operating_activities + net_cash_from_investing_activities + net_cash_from_financing_activities + effect_of_exchange_rate_changes == net_change_in_cash"
                input_nc = {
                    "net_cash_from_operating_activities": operating,
                    "net_cash_from_investing_activities": investing,
                    "net_cash_from_financing_activities": financing,
                    "effect_of_exchange_rate_changes": exchange_effect,
                    "net_change_in_cash": net_change,
                }
                msg_pass = f"Net change in cash verified: {operating} + {investing} + {financing} + {exchange_effect} = {net_change}."
                msg_fail = f"Net change in cash mismatch: {operating} + {investing} + {financing} + {exchange_effect} = {calc_nc} != reported {net_change} (variance: {var})."
            else:
                calc_nc = round(operating + investing + financing, 4)
                var = round(calc_nc - net_change, 4)
                formula_nc = "net_cash_from_operating_activities + net_cash_from_investing_activities + net_cash_from_financing_activities == net_change_in_cash"
                input_nc = {
                    "net_cash_from_operating_activities": operating,
                    "net_cash_from_investing_activities": investing,
                    "net_cash_from_financing_activities": financing,
                    "net_change_in_cash": net_change,
                }
                msg_pass = f"Net change in cash verified: {operating} + {investing} + {financing} = {net_change}."
                msg_fail = f"Net change in cash mismatch: {operating} + {investing} + {financing} = {calc_nc} != reported {net_change} (variance: {var})."

            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = msg_pass if status == ValidationStatus.PASS else msg_fail
            checks.append(
                ValidationCheckResult(
                    rule_name="cash_flow_net_change_calculation",
                    formula=formula_nc,
                    input_values=input_nc,
                    calculated_value=calc_nc,
                    reported_value=net_change,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 2: Beginning Cash + Net Change in Cash == Ending Cash Balance
        missing_end = []
        if beginning is None:
            missing_end.append("beginning_cash_balance")
        if net_change is None:
            missing_end.append("net_change_in_cash")
        if ending is None:
            missing_end.append("ending_cash_balance")

        if missing_end:
            checks.append(
                ValidationCheckResult(
                    rule_name="cash_flow_ending_cash_calculation",
                    formula="beginning_cash_balance + net_change_in_cash == ending_cash_balance",
                    input_values={
                        "beginning_cash_balance": beginning,
                        "net_change_in_cash": net_change,
                        "ending_cash_balance": ending,
                    },
                    calculated_value=None,
                    reported_value=ending,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message=f"Ending cash check missing required inputs: {', '.join(missing_end)}.",
                    missing_fields=missing_end,
                )
            )
        else:
            calc_ending = round(beginning + net_change, 4)
            var = round(calc_ending - ending, 4)
            status = (
                ValidationStatus.PASS
                if abs(var) <= tol
                else ValidationStatus.FAILED
            )
            msg = (
                f"Ending cash balance verified: {beginning} + {net_change} = {ending}."
                if status == ValidationStatus.PASS
                else f"Ending cash balance mismatch: {beginning} + {net_change} = {calc_ending} != reported {ending} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="cash_flow_ending_cash_calculation",
                    formula="beginning_cash_balance + net_change_in_cash == ending_cash_balance",
                    input_values={
                        "beginning_cash_balance": beginning,
                        "net_change_in_cash": net_change,
                        "ending_cash_balance": ending,
                    },
                    calculated_value=calc_ending,
                    reported_value=ending,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        return checks


# Convenience module-level function
validate_financial_data = FinancialValidationService.validate
