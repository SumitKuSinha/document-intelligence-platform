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
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from app.schemas.financial_validation import (
    FinancialValidationResult,
    FinancialValidationSummary,
    ValidationCheckResult,
    ValidationStatus,
)
from app.schemas.financial_extraction import (
    BalanceSheetExtractionData,
    CashFlowExtractionData,
    InvoiceExtractionData,
    ProfitAndLossExtractionData,
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

        # Type-based dispatching for typed extraction models
        if isinstance(payload, BalanceSheetExtractionData):
            doc_type_clean = "balance_sheet"
        elif isinstance(payload, ProfitAndLossExtractionData):
            doc_type_clean = "profit_and_loss"
        elif isinstance(payload, CashFlowExtractionData):
            doc_type_clean = "cash_flow_statement"
        elif isinstance(payload, InvoiceExtractionData):
            doc_type_clean = "invoice"

        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif not isinstance(payload, dict):
            payload = {}

        # If payload carries explicit document_type or statement_type metadata, honor it
        if isinstance(payload, dict):
            if payload.get("document_type") and str(payload["document_type"]).strip().lower() in cls.SUPPORTED_TYPES:
                doc_type_clean = str(payload["document_type"]).strip().lower()
            elif payload.get("statement_type") and str(payload["statement_type"]).strip().lower() in cls.SUPPORTED_TYPES:
                doc_type_clean = str(payload["statement_type"]).strip().lower()
            # If doc_type_clean defaulted to invoice but payload explicitly contains balance sheet core fields
            elif doc_type_clean == "invoice" and any(k in payload for k in ["total_assets", "total_capital_and_liabilities", "current_assets"]) and not any(k in payload for k in ["subtotal", "tax_rate", "tax_amount", "vendor_name"]):
                doc_type_clean = "balance_sheet"

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

    @classmethod
    def _extract_round_off(cls, payload: Dict[str, Any]) -> Optional[float]:
        """
        Extract explicitly reported round-off amount from payload or additional_fields.
        Returns float or None. Does not invent a round-off value.
        """
        candidate_keys = (
            "round_off_amount",
            "round_off",
            "rounding_amount",
            "rounding",
            "roundoff_amount",
            "roundoff",
        )
        for k in candidate_keys:
            if k in payload and payload[k] is not None:
                val = parse_financial_number(payload[k])
                if val is not None:
                    return val

        for k, v in payload.items():
            if v is not None and k.lower().replace(" ", "_").replace("-", "_") in candidate_keys:
                val = parse_financial_number(v)
                if val is not None:
                    return val

        add_fields = payload.get("additional_fields")
        if isinstance(add_fields, dict):
            for k in candidate_keys:
                if k in add_fields and add_fields[k] is not None:
                    val = parse_financial_number(add_fields[k])
                    if val is not None:
                        return val
            for k, v in add_fields.items():
                if v is not None and k.lower().replace(" ", "_").replace("-", "_") in candidate_keys:
                    val = parse_financial_number(v)
                    if val is not None:
                        return val

        return None

    @classmethod
    def _parse_rate(cls, val: Any) -> Optional[float]:
        """Normalize a tax or discount rate representation (e.g. 0.09, 9, '9%') to decimal fraction (e.g. 0.09)."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            num = float(val)
            if 0.0 <= num <= 1.0:
                return round(num, 6)
            elif 1.0 < num <= 100.0:
                return round(num / 100.0, 6)
            return None
        if isinstance(val, str):
            cleaned = val.strip()
            if not cleaned or cleaned.lower() in ("null", "none", "n/a", "na", "-", "--"):
                return None
            is_pct = "%" in cleaned
            parsed = parse_financial_number(cleaned)
            if parsed is not None:
                if is_pct:
                    return round(parsed, 6)
                if 0.0 <= parsed <= 1.0:
                    return round(parsed, 6)
                elif 1.0 < parsed <= 100.0:
                    return round(parsed / 100.0, 6)
        return None

    @classmethod
    def _extract_tax_components(
        cls,
        payload: Dict[str, Any],
        subtotal: Optional[float],
        total_tax_rate: Optional[float],
        raw_items: List[Any],
        tol: float,
    ) -> List[Dict[str, Any]]:
        """
        Extract explicit statutory tax components (e.g. CGST, SGST, IGST, UTGST, CESS, VAT) from payload.
        Preserves existing behavior if components are not available.
        Does not invent tax components or treat unrelated percentages as tax.
        """
        known_names = ("cgst", "sgst", "utgst", "igst", "cess", "vat")
        components_map: Dict[str, Dict[str, Any]] = {}

        add_fields = payload.get("additional_fields") if isinstance(payload.get("additional_fields"), dict) else {}

        # 1. Check structured tax_components / taxes field
        tax_comps_source = (
            payload.get("tax_components")
            or add_fields.get("tax_components")
            or payload.get("taxes")
            or add_fields.get("taxes")
        )
        if isinstance(tax_comps_source, list):
            for entry in tax_comps_source:
                if isinstance(entry, dict):
                    name_raw = str(entry.get("name") or entry.get("tax_type") or entry.get("type") or "").strip().lower()
                    if any(kn in name_raw for kn in known_names) or name_raw:
                        r = cls._parse_rate(entry.get("rate") or entry.get("tax_rate") or entry.get("percentage") or entry.get("pct"))
                        a = parse_financial_number(entry.get("amount") or entry.get("tax_amount") or entry.get("value"))
                        components_map[name_raw] = {"name": name_raw, "rate": r, "amount": a}
        elif isinstance(tax_comps_source, dict):
            for k, v in tax_comps_source.items():
                name_raw = str(k).strip().lower()
                if any(kn in name_raw for kn in known_names):
                    if isinstance(v, dict):
                        r = cls._parse_rate(v.get("rate") or v.get("tax_rate") or v.get("pct"))
                        a = parse_financial_number(v.get("amount") or v.get("tax_amount") or v.get("value"))
                        components_map[name_raw] = {"name": name_raw, "rate": r, "amount": a}
                    elif isinstance(v, (int, float, str)):
                        r = cls._parse_rate(v)
                        a = parse_financial_number(v)
                        if str(v).strip().endswith("%"):
                            components_map[name_raw] = {"name": name_raw, "rate": r, "amount": None}
                        else:
                            components_map[name_raw] = {"name": name_raw, "rate": None, "amount": a}

        # 2. Check direct fields in payload and additional_fields for known tax components
        for name in known_names:
            if name in components_map:
                continue
            rate_keys = (f"{name}_rate", f"{name}_pct", f"{name}_percentage", f"{name}_percent")
            amt_keys = (f"{name}_amount", f"{name}_val", f"{name}_value", name)

            r_val = None
            a_val = None

            # Look up rate
            for d in (payload, add_fields):
                for rk in rate_keys:
                    if rk in d and d[rk] is not None:
                        r_val = cls._parse_rate(d[rk])
                        if r_val is not None:
                            break
                    for k, v in d.items():
                        if v is not None and k.lower().replace(" ", "_").replace("-", "_") == rk:
                            r_val = cls._parse_rate(v)
                            if r_val is not None:
                                break
                    if r_val is not None:
                        break
                if r_val is not None:
                    break

            # Look up amount
            for d in (payload, add_fields):
                for ak in amt_keys:
                    if ak in d and d[ak] is not None:
                        val = d[ak]
                        if isinstance(val, str) and "%" in val:
                            if r_val is None:
                                r_val = cls._parse_rate(val)
                            continue
                        parsed = parse_financial_number(val)
                        if parsed is not None:
                            a_val = parsed
                            break
                    for k, v in d.items():
                        if v is not None and k.lower().replace(" ", "_").replace("-", "_") == ak:
                            if isinstance(v, str) and "%" in v:
                                if r_val is None:
                                    r_val = cls._parse_rate(v)
                                continue
                            parsed = parse_financial_number(v)
                            if parsed is not None:
                                a_val = parsed
                                break
                    if a_val is not None:
                        break
                if a_val is not None:
                    break

            if r_val is not None or a_val is not None:
                components_map[name] = {"name": name, "rate": r_val, "amount": a_val}

        # 3. Special GST deduction: if cgst and sgst exist, but rates are None
        if "cgst" in components_map and "sgst" in components_map:
            cgst_entry = components_map["cgst"]
            sgst_entry = components_map["sgst"]
            if cgst_entry.get("rate") is None and sgst_entry.get("rate") is None:
                if total_tax_rate is not None and total_tax_rate > 0:
                    half_rate = round(total_tax_rate / 2.0, 6)
                    cgst_entry["rate"] = half_rate
                    sgst_entry["rate"] = half_rate
                elif subtotal is not None and subtotal > 0 and cgst_entry.get("amount") is not None:
                    ratio = cgst_entry["amount"] / subtotal
                    estimated_rate = round(round(ratio * 100) / 100.0, 6)
                    if (
                        abs(round(subtotal * estimated_rate, 2) - cgst_entry["amount"]) <= tol
                        or abs(subtotal * estimated_rate - cgst_entry["amount"]) <= tol
                    ):
                        cgst_entry["rate"] = estimated_rate
                        sgst_entry["rate"] = estimated_rate

        if not components_map:
            return []

        # 4. Resolve calculated 2-decimal rounded amount for each component
        total_tax_reported = parse_financial_number(
            payload.get("tax_amount") if payload.get("tax_amount") is not None else payload.get("tax")
        )
        resolved_components = []
        for name, comp in components_map.items():
            r = comp.get("rate")
            a = comp.get("amount")

            calc_amt = None
            if subtotal is not None and r is not None:
                exact_val = subtotal * r
                std_round = round(exact_val, 2)
                floor_round = round(int(round(exact_val, 4) * 100) / 100.0, 2)
                line_sum = None
                if raw_items:
                    line_sum = round(
                        sum(
                            round(parse_financial_number(
                                (it if isinstance(it, dict) else it.model_dump()).get("total_price", 0)
                            ) * r, 2)
                            for it in raw_items
                            if parse_financial_number((it if isinstance(it, dict) else it.model_dump()).get("total_price")) is not None
                        ),
                        2
                    )

                if a is not None:
                    if (
                        abs(a - exact_val) <= tol
                        or abs(a - std_round) <= tol
                        or abs(a - floor_round) <= tol
                        or (line_sum is not None and abs(a - line_sum) <= tol)
                    ):
                        calc_amt = a
                    else:
                        calc_amt = std_round
                else:
                    calc_amt = std_round
            elif a is not None:
                calc_amt = a

            comp["calculated_amount"] = calc_amt
            resolved_components.append(comp)

        # Post-pass: if sum of standard rounds doesn't match total_tax_reported, but floor_round matches, use floor
        if total_tax_reported is not None and subtotal is not None:
            current_sum = round(sum(c["calculated_amount"] for c in resolved_components if c["calculated_amount"] is not None), 2)
            if abs(current_sum - total_tax_reported) > tol:
                floor_candidates = []
                all_have_floor = True
                for c in resolved_components:
                    r = c.get("rate")
                    if r is not None:
                        f_val = round(int(round(subtotal * r, 4) * 100) / 100.0, 2)
                        floor_candidates.append(f_val)
                    else:
                        all_have_floor = False
                if all_have_floor and abs(round(sum(floor_candidates), 2) - total_tax_reported) <= tol:
                    for idx, c in enumerate(resolved_components):
                        c["calculated_amount"] = floor_candidates[idx]

        return resolved_components

    @classmethod
    def _parse_line_item_discount(
        cls,
        item_dict: Dict[str, Any],
        qty: Optional[float] = None,
        price: Optional[float] = None,
        total_price: Optional[float] = None,
        tol: float = 0.05,
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract explicitly reported discount from an invoice line item.
        Returns ('PERCENT', pct) or ('AMOUNT', amt) or (None, None).
        """
        # 1. Explicit percentage fields
        for k in ("discount_percent", "discount_percentage", "discount_pct", "discount_rate"):
            if k in item_dict and item_dict[k] is not None:
                val = item_dict[k]
                if isinstance(val, str):
                    val = val.strip()
                    if not val or val.lower() in ("null", "none", "n/a", "na", "-", "--"):
                        return None, None
                    if val.endswith("%"):
                        val = val[:-1].strip()
                try:
                    num = float(val)
                    if k == "discount_rate" and 0.0 < num <= 1.0:
                        num = num * 100.0
                    if 0.0 <= num <= 100.0:
                        return "PERCENT", num
                    return None, None
                except (ValueError, TypeError):
                    return None, None

        # 2. Explicit monetary discount fields
        for k in ("discount_amount", "discount_value"):
            if k in item_dict and item_dict[k] is not None:
                val = item_dict[k]
                amt = parse_financial_number(val)
                if amt is not None and amt >= 0.0:
                    return "AMOUNT", amt
                return None, None

        # 3. Generic 'discount' field
        if "discount" in item_dict and item_dict["discount"] is not None:
            raw = item_dict["discount"]
            if isinstance(raw, str):
                cleaned = raw.strip()
                if not cleaned or cleaned.lower() in ("null", "none", "n/a", "na", "-", "--"):
                    return None, None
                # Explicit percentage symbol in string (e.g. "99%", "15.5%")
                if "%" in cleaned:
                    match = re.search(r"(\d+(?:\.\d+)?)\s*%", cleaned)
                    if match:
                        try:
                            pct = float(match.group(1))
                            if 0.0 <= pct <= 100.0:
                                return "PERCENT", pct
                        except ValueError:
                            pass
                    return None, None
                # Explicit currency symbols / codes -> monetary discount
                if any(sym in cleaned for sym in ("$", "€", "£", "¥", "₹", "USD", "EUR", "GBP", "INR")):
                    amt = parse_financial_number(cleaned)
                    if amt is not None and amt >= 0.0:
                        return "AMOUNT", amt
                    return None, None
                # Numeric string without units (e.g. "99", "44.19")
                try:
                    num = float(cleaned)
                    if qty is not None and price is not None and total_price is not None:
                        matches_pct = (0.0 <= num <= 100.0) and abs(round(qty * price * (1.0 - num / 100.0), 4) - total_price) <= tol
                        matches_amt = (num >= 0.0) and abs(round(qty * price - num, 4) - total_price) <= tol
                        if matches_pct and not matches_amt:
                            return "PERCENT", num
                        elif matches_amt and not matches_pct:
                            return "AMOUNT", num
                        elif matches_pct and matches_amt:
                            return "PERCENT", num
                    return None, None
                except ValueError:
                    return None, None
            elif isinstance(raw, (int, float)):
                num = float(raw)
                if qty is not None and price is not None and total_price is not None:
                    matches_pct = (0.0 <= num <= 100.0) and abs(round(qty * price * (1.0 - num / 100.0), 4) - total_price) <= tol
                    matches_amt = (num >= 0.0) and abs(round(qty * price - num, 4) - total_price) <= tol
                    if matches_pct and not matches_amt:
                        return "PERCENT", num
                    elif matches_amt and not matches_pct:
                        return "AMOUNT", num
                    elif matches_pct and matches_amt:
                        return "PERCENT", num
                return None, None

        # 4. Check source_snippet and description for explicit discount strings
        text_sources = []
        if item_dict.get("source_snippet"):
            text_sources.append(str(item_dict["source_snippet"]))
        if item_dict.get("description"):
            text_sources.append(str(item_dict["description"]))

        # 4a. Explicit discount keywords in text
        for text in text_sources:
            disc_match = re.search(r"(?:discount|disc|dis\b|less|off|rebate)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
            if not disc_match:
                disc_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:discount|disc|dis\b|less|off|rebate)", text, re.IGNORECASE)
            if disc_match:
                try:
                    pct = float(disc_match.group(1))
                    if 0.0 <= pct <= 100.0:
                        return "PERCENT", pct
                except ValueError:
                    pass

            # Monetary discount with keyword (e.g. "discount: $5.00" or "less 10.00")
            amt_match = re.search(r"(?:discount|disc|dis\b|less|off|rebate)\s*[:=]?\s*[$€£¥₹]?\s*(\d+(?:\.\d+)?)(?!\s*%)", text, re.IGNORECASE)
            if amt_match:
                try:
                    amt = float(amt_match.group(1))
                    if amt >= 0.0:
                        return "AMOUNT", amt
                except ValueError:
                    pass

        # 4b. Explicit percentage reported in snippet/description (e.g. table columns like "... 7.44 PCS 99 % 0.45")
        for text in text_sources:
            pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            for cand_str in pct_matches:
                try:
                    cand_pct = float(cand_str)
                except ValueError:
                    continue
                if 0.0 <= cand_pct <= 100.0:
                    if qty is not None and price is not None and total_price is not None:
                        calc_unrounded = round(qty * price * (1.0 - cand_pct / 100.0), 4)
                        undiscounted = round(qty * price, 2)
                        discount_amt = round(undiscounted * (cand_pct / 100.0), 2)
                        calc_rounded = round(undiscounted - discount_amt, 2)
                        if abs(calc_unrounded - total_price) <= tol or abs(calc_rounded - total_price) <= tol:
                            return "PERCENT", cand_pct

        return None, None

    @classmethod
    def _resolve_invoice_pricing_model(
        cls,
        payload: Dict[str, Any],
        subtotal: Optional[float],
        tax_amount: Optional[float],
        total_amount: Optional[float],
        line_totals_sum: Optional[float],
        tol: float,
    ) -> str:
        """
        Determine if invoice uses 'TAX_EXCLUSIVE' or 'TAX_INCLUSIVE' accounting semantics.
        Returns 'TAX_EXCLUSIVE' or 'TAX_INCLUSIVE'.
        """
        # 1. Check explicit field in payload
        pricing_type = str(payload.get("pricing_type") or "").strip().lower()
        if pricing_type in ("tax_inclusive", "inclusive", "gross"):
            return "TAX_INCLUSIVE"
        if pricing_type in ("tax_exclusive", "exclusive", "net"):
            return "TAX_EXCLUSIVE"

        # 2. Check line items for is_tax_inclusive
        raw_items = payload.get("line_items") or []
        for it in raw_items:
            it_dict = it if isinstance(it, dict) else it.model_dump()
            if it_dict.get("is_tax_inclusive") is True:
                return "TAX_INCLUSIVE"

        # 3. Check document text context in additional_fields or evidence
        add_fields = payload.get("additional_fields") if isinstance(payload.get("additional_fields"), dict) else {}
        for k, v in add_fields.items():
            k_lower = str(k).lower()
            v_lower = str(v).lower()
            if any(term in k_lower or term in v_lower for term in ("inclusive of tax", "tax inclusive", "vat inclusive", "all taxes included")):
                return "TAX_INCLUSIVE"

        # 4. Mathematical context resolution
        if line_totals_sum is not None and total_amount is not None and tax_amount is not None and tax_amount > 0:
            shipping = parse_financial_number(payload.get("shipping_amount")) or 0.0
            round_off = cls._extract_round_off(payload) or 0.0

            matches_inclusive_total = abs(line_totals_sum + shipping + round_off - total_amount) <= tol
            matches_exclusive_subtotal = subtotal is not None and abs(line_totals_sum - subtotal) <= tol

            if matches_inclusive_total and not matches_exclusive_subtotal:
                return "TAX_INCLUSIVE"
            if matches_exclusive_subtotal and not matches_inclusive_total:
                return "TAX_EXCLUSIVE"

        # Default standard B2B model is TAX_EXCLUSIVE
        return "TAX_EXCLUSIVE"

    # ---------------------------------------------------------------------------
    # 1. Invoice Validation Checks
    # ---------------------------------------------------------------------------
    @classmethod
    def _validate_invoice(
        cls, payload: Dict[str, Any], tol: float
    ) -> List[ValidationCheckResult]:
        checks: List[ValidationCheckResult] = []

        raw_items = payload.get("line_items") or []
        subtotal = parse_financial_number(
            payload.get("subtotal")
            if payload.get("subtotal") is not None
            else payload.get("taxable_amount")
            if payload.get("taxable_amount") is not None
            else payload.get("taxable_value")
        )
        tax = parse_financial_number(
            payload.get("tax_amount") if payload.get("tax_amount") is not None else payload.get("tax")
        )
        tax_rate = cls._parse_rate(payload.get("tax_rate"))
        discount = parse_financial_number(payload.get("discount_amount"))
        shipping = parse_financial_number(payload.get("shipping_amount"))
        total = parse_financial_number(
            payload.get("total_amount") if payload.get("total_amount") is not None else payload.get("total")
        )
        round_off = cls._extract_round_off(payload)

        # Pre-compute valid line totals sum if present
        line_totals = []
        missing_line_totals = []
        for idx, item in enumerate(raw_items):
            item_dict = item if isinstance(item, dict) else item.model_dump()
            t_val = parse_financial_number(item_dict.get("total_price"))
            if t_val is None:
                missing_line_totals.append(f"line_items[{idx + 1}].total_price")
            else:
                line_totals.append(t_val)

        line_totals_sum = round(sum(line_totals), 4) if (raw_items and not missing_line_totals) else None

        # Resolve pricing model
        pricing_model = cls._resolve_invoice_pricing_model(
            payload=payload,
            subtotal=subtotal,
            tax_amount=tax,
            total_amount=total,
            line_totals_sum=line_totals_sum,
            tol=tol,
        )

        # -----------------------------------------------------------------------
        # Check 1: Individual Line Item Math
        # -----------------------------------------------------------------------
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
                qty = parse_financial_number(
                    item_dict.get("quantity") if item_dict.get("quantity") is not None else item_dict.get("qty")
                )
                price = parse_financial_number(
                    item_dict.get("unit_price") if item_dict.get("unit_price") is not None else item_dict.get("rate")
                )
                total_price = parse_financial_number(
                    item_dict.get("total_price") if item_dict.get("total_price") is not None else item_dict.get("amount")
                )

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
                    disc_type, disc_val = cls._parse_line_item_discount(
                        item_dict, qty=qty, price=price, total_price=total_price, tol=tol
                    )

                    if disc_type == "PERCENT":
                        formula = "quantity * unit_price * (1 - discount_percent / 100) == total_price"
                        input_vals["discount_percent"] = disc_val
                        calc_unrounded = round(qty * price * (1.0 - disc_val / 100.0), 4)
                        undiscounted = round(qty * price, 2)
                        discount_amt = round(undiscounted * (disc_val / 100.0), 2)
                        calc_rounded = round(undiscounted - discount_amt, 2)
                        if abs(calc_rounded - total_price) < abs(calc_unrounded - total_price):
                            calc = calc_rounded
                        else:
                            calc = calc_unrounded

                        var = round(calc - total_price, 4)
                        status = (
                            ValidationStatus.PASS
                            if abs(var) <= tol
                            else ValidationStatus.FAILED
                        )
                        msg = (
                            f"Line item '{desc}' passed arithmetic check with {disc_val}% discount."
                            if status == ValidationStatus.PASS
                            else f"Line item '{desc}' arithmetic mismatch with {disc_val}% discount: {qty} * {price} * (1 - {disc_val}/100) = {calc} != reported {total_price} (variance: {var})."
                        )

                    elif disc_type == "AMOUNT":
                        formula = "quantity * unit_price - discount_amount == total_price"
                        input_vals["discount_amount"] = disc_val
                        calc_line = round(qty * price - disc_val, 4)
                        calc_unit = round(qty * (price - disc_val), 4)
                        if abs(calc_unit - total_price) < abs(calc_line - total_price) and abs(calc_unit - total_price) <= tol:
                            calc = calc_unit
                            formula = "quantity * (unit_price - discount_amount) == total_price"
                        else:
                            calc = calc_line

                        var = round(calc - total_price, 4)
                        status = (
                            ValidationStatus.PASS
                            if abs(var) <= tol
                            else ValidationStatus.FAILED
                        )
                        msg = (
                            f"Line item '{desc}' passed arithmetic check with discount amount {disc_val}."
                            if status == ValidationStatus.PASS
                            else f"Line item '{desc}' arithmetic mismatch with discount amount {disc_val}: {qty} * {price} - {disc_val} = {calc} != reported {total_price} (variance: {var})."
                        )

                    else:
                        formula = "quantity * unit_price == total_price"
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
                            formula=formula,
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

        # -----------------------------------------------------------------------
        # Check 2: Sum of Line Items Totals Reconcile
        # -----------------------------------------------------------------------
        if not raw_items:
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
        elif missing_line_totals:
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
                    message=f"Cannot reconcile line items with subtotal: missing {', '.join(missing_line_totals)}.",
                    missing_fields=missing_line_totals,
                )
            )
        elif subtotal is None and pricing_model != "TAX_INCLUSIVE":
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
        else:
            calc_subtotal = line_totals_sum
            if pricing_model == "TAX_INCLUSIVE" and (subtotal is None or abs(calc_subtotal - (total or 0.0)) <= tol):
                target_val = total
                formula_str = "sum(line_items.total_price) == total_amount (tax-inclusive)"
                var = round(calc_subtotal - target_val, 4) if target_val is not None else 0.0
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
                msg = (
                    f"Sum of {len(line_totals)} tax-inclusive line item(s) matches total {target_val}."
                    if status == ValidationStatus.PASS
                    else f"Sum of tax-inclusive line items ({calc_subtotal}) does not match total {target_val} (variance: {var})."
                )
            else:
                target_val = subtotal
                formula_str = "sum(line_items.total_price) == subtotal"
                var = round(calc_subtotal - target_val, 4) if target_val is not None else 0.0
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
                msg = (
                    f"Sum of {len(line_totals)} line item(s) matches subtotal {subtotal}."
                    if status == ValidationStatus.PASS
                    else f"Sum of line items ({calc_subtotal}) does not match subtotal {subtotal} (variance: {var})."
                )

            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_line_items_subtotal_reconciliation",
                    formula=formula_str,
                    input_values={
                        "subtotal": subtotal,
                        "line_item_totals": line_totals,
                        "line_item_count": len(line_totals),
                    },
                    calculated_value=calc_subtotal,
                    reported_value=target_val,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # -----------------------------------------------------------------------
        # Check 3: Invoice Total Reconciliation
        # -----------------------------------------------------------------------
        inputs_total = {
            "subtotal": subtotal,
            "tax_amount": tax,
            "discount_amount": discount,
            "shipping_amount": shipping,
            "total_amount": total,
        }
        if round_off is not None:
            inputs_total["round_off_amount"] = round_off

        if total is None or (subtotal is None and pricing_model != "TAX_INCLUSIVE"):
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
            round_val = round_off if round_off is not None else 0.0

            if pricing_model == "TAX_INCLUSIVE":
                if subtotal is not None and abs(subtotal + (tax or 0.0) + (shipping or 0.0) + round_val - total) <= tol:
                    calc_total = round(subtotal + (tax or 0.0) + (shipping or 0.0) + round_val, 4)
                    formula_parts = ["subtotal", "tax_amount"]
                    if shipping:
                        formula_parts.append("shipping_amount")
                    if round_off is not None:
                        formula_parts.append("round_off_amount")
                    formula_total = " + ".join(formula_parts) + " == total_amount"
                else:
                    base = subtotal if subtotal is not None else (line_totals_sum or total)
                    calc_total = round(base + (shipping or 0.0) + round_val, 4)
                    formula_total = "subtotal + shipping_amount + round_off_amount == total_amount (tax-inclusive)" if round_off is not None else "subtotal + shipping_amount == total_amount (tax-inclusive)"

                var = round(calc_total - total, 4)
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
                msg = (
                    f"Invoice total reconciliation passed ({calc_total} == {total})."
                    if status == ValidationStatus.PASS
                    else f"Invoice total reconciliation failed: calculated {calc_total} != reported {total} (variance: {var})."
                )
            else:
                has_line_discounts = any(
                    cls._parse_line_item_discount(it if isinstance(it, dict) else it.model_dump(), tol=tol)[0] is not None
                    for it in raw_items
                )
                discount_val = discount or 0.0
                order_discount = discount_val

                if has_line_discounts and discount_val > 0:
                    without_disc = round(subtotal + (tax or 0.0) + (shipping or 0.0) + round_val, 4)
                    with_disc = round(subtotal + (tax or 0.0) - discount_val + (shipping or 0.0) + round_val, 4)
                    if abs(without_disc - total) <= tol and abs(with_disc - total) > tol:
                        order_discount = 0.0

                calc_total = round(
                    subtotal + (tax or 0.0) - order_discount + (shipping or 0.0) + round_val, 4
                )
                var = round(calc_total - total, 4)
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED

                if round_off is not None:
                    formula_total = "subtotal + tax_amount - discount_amount + shipping_amount + round_off_amount == total_amount"
                else:
                    formula_total = "subtotal + tax_amount - discount_amount + shipping_amount == total_amount"

                msg = (
                    f"Invoice total reconciliation passed ({calc_total} == {total})."
                    if status == ValidationStatus.PASS
                    else f"Invoice total reconciliation failed: calculated {calc_total} != reported {total} (variance: {var})."
                )

            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_total_reconciliation",
                    formula=formula_total,
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

        # -----------------------------------------------------------------------
        # Check 4: Tax Calculation Check
        # -----------------------------------------------------------------------
        tax_components = cls._extract_tax_components(
            payload=payload,
            subtotal=subtotal,
            total_tax_rate=tax_rate,
            raw_items=raw_items,
            tol=tol,
        )

        is_zero_tax = (tax == 0.0) or (tax_rate == 0.0) or (tax is None and tax_rate == 0.0)

        if tax_components:
            missing_tax = []
            if tax is None:
                missing_tax.append("tax_amount")
            if subtotal is None and any(c.get("amount") is None for c in tax_components):
                missing_tax.append("subtotal")

            comp_names = [c["name"] for c in tax_components]
            formula_tax = " + ".join([f"{c}_amount" for c in comp_names]) + " == tax_amount"
            inputs_tax = {
                "subtotal": subtotal,
                "tax_amount": tax,
            }
            if tax_rate is not None:
                inputs_tax["tax_rate"] = tax_rate
            for c in tax_components:
                c_name = c["name"]
                if c.get("rate") is not None:
                    inputs_tax[f"{c_name}_rate"] = c["rate"]
                if c.get("amount") is not None:
                    inputs_tax[f"{c_name}_amount"] = c["amount"]
                elif c.get("calculated_amount") is not None:
                    inputs_tax[f"{c_name}_amount"] = c["calculated_amount"]

            if missing_tax:
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_tax_calculation",
                        formula=formula_tax,
                        input_values=inputs_tax,
                        calculated_value=None,
                        reported_value=tax,
                        variance=None,
                        status=ValidationStatus.NOT_APPLICABLE,
                        tolerance=tol,
                        message=f"Cannot compute tax check with components: missing {', '.join(missing_tax)}.",
                        missing_fields=missing_tax,
                    )
                )
            else:
                calc_tax = round(sum(c["calculated_amount"] for c in tax_components if c["calculated_amount"] is not None), 4)
                var = round(calc_tax - tax, 4)
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
                comp_summary = " + ".join([f"{c['name']}: {c['calculated_amount']}" for c in tax_components])
                msg = (
                    f"Tax calculation passed with components ({comp_summary} == {tax})."
                    if status == ValidationStatus.PASS
                    else f"Tax calculation mismatch with components: {comp_summary} = {calc_tax} != reported tax {tax} (variance: {var})."
                )
                checks.append(
                    ValidationCheckResult(
                        rule_name="invoice_tax_calculation",
                        formula=formula_tax,
                        input_values=inputs_tax,
                        calculated_value=calc_tax,
                        reported_value=tax,
                        variance=var,
                        status=status,
                        tolerance=tol,
                        message=msg,
                        missing_fields=[],
                    )
                )

        elif is_zero_tax:
            formula_tax = "tax_amount == 0 (tax-exempt / zero-rated)"
            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_tax_calculation",
                    formula=formula_tax,
                    input_values={"subtotal": subtotal, "tax_rate": tax_rate, "tax_amount": tax},
                    calculated_value=0.0,
                    reported_value=tax or 0.0,
                    variance=0.0,
                    status=ValidationStatus.PASS,
                    tolerance=tol,
                    message="Invoice is tax-exempt or zero-rated; tax amount is 0.0.",
                    missing_fields=[],
                )
            )

        elif pricing_model == "TAX_INCLUSIVE" and tax_rate is not None and (total is not None or subtotal is not None):
            base_for_tax = total if total is not None else ((subtotal or 0.0) + (tax or 0.0))
            calc_tax = round(base_for_tax * (tax_rate / (1.0 + tax_rate)), 4)
            var = round(calc_tax - (tax or 0.0), 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
            formula_tax = "total_amount * tax_rate / (1 + tax_rate) == tax_amount (tax-inclusive)"
            msg = (
                f"Tax-inclusive calculation passed: {base_for_tax} * {tax_rate}/(1+{tax_rate}) = {calc_tax} == {tax}."
                if status == ValidationStatus.PASS
                else f"Tax-inclusive calculation mismatch: {base_for_tax} * {tax_rate}/(1+{tax_rate}) = {calc_tax} != reported tax {tax} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="invoice_tax_calculation",
                    formula=formula_tax,
                    input_values={"total_amount": base_for_tax, "tax_rate": tax_rate, "tax_amount": tax},
                    calculated_value=calc_tax,
                    reported_value=tax,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        elif tax_rate is not None or (tax is not None and subtotal is not None):
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
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
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
        total_liab_eq = parse_financial_number(
            payload.get("total_liabilities_and_equity")
            or payload.get("total_capital_and_liabilities")
            or payload.get("total_equity_and_liabilities")
            or payload.get("total_capital_liabilities")
        )

        curr_assets = parse_financial_number(payload.get("current_assets"))
        non_curr_assets = parse_financial_number(payload.get("non_current_assets"))
        curr_liab = parse_financial_number(payload.get("current_liabilities"))
        non_curr_liab = parse_financial_number(payload.get("non_current_liabilities"))
        retained = parse_financial_number(payload.get("retained_earnings"))
        share_cap = parse_financial_number(payload.get("share_capital") or payload.get("capital"))

        raw_items = payload.get("line_items") or []
        footnote_kw = ["contingent", "bills for collection", "commitments", "guarantees", "off-balance"]

        asset_items: List[Tuple[str, float]] = []
        liability_equity_items: List[Tuple[str, float]] = []

        for it in raw_items:
            it_dict = it if isinstance(it, dict) else (it.model_dump() if hasattr(it, "model_dump") else {})
            name = str(it_dict.get("item_name") or it_dict.get("description") or "").strip()
            cat = str(it_dict.get("category") or "").strip().lower()
            amt = parse_financial_number(it_dict.get("amount") if it_dict.get("amount") is not None else it_dict.get("total_price"))
            if amt is None:
                continue
            name_lower = name.lower()
            if any(fkw in name_lower for fkw in footnote_kw):
                continue
            if name_lower in ["total", "total assets", "total capital and liabilities", "total liabilities and equity"]:
                continue

            if "asset" in cat:
                asset_items.append((name, amt))
            elif any(k in cat for k in ["liabilit", "equity", "capital"]):
                liability_equity_items.append((name, amt))
            else:
                asset_kw = ["cash", "bank", "investment", "advance", "loan", "fixed asset", "property", "equipment", "receivable", "inventory", "goodwill"]
                liab_eq_kw = ["capital", "reserves", "surplus", "minority", "deposit", "borrowing", "provision", "payable", "liabilit", "equity", "debt"]
                if any(akw in name_lower for akw in asset_kw) and not any(lkw in name_lower for lkw in liab_eq_kw):
                    asset_items.append((name, amt))
                elif any(lkw in name_lower for lkw in liab_eq_kw):
                    liability_equity_items.append((name, amt))

        # -----------------------------------------------------------------------
        # Check 1: Fundamental Accounting Equation (Total Assets == Total Financing)
        # -----------------------------------------------------------------------
        calc_financing = None
        financing_inputs = {}
        financing_formula_str = ""

        if total_liab is not None and total_equity is not None:
            calc_financing = round(total_liab + total_equity, 4)
            financing_inputs = {
                "total_assets": total_assets,
                "total_liabilities": total_liab,
                "total_equity": total_equity,
            }
            financing_formula_str = "total_liabilities + total_equity == total_assets"
        elif total_liab_eq is not None:
            calc_financing = total_liab_eq
            financing_inputs = {
                "total_assets": total_assets,
                "total_capital_and_liabilities": total_liab_eq,
            }
            financing_formula_str = "total_capital_and_liabilities == total_assets"
        elif liability_equity_items and len(liability_equity_items) >= 2:
            calc_financing = round(sum(amt for _, amt in liability_equity_items), 4)
            financing_inputs = {f"item_{i+1}_{name}": amt for i, (name, amt) in enumerate(liability_equity_items)}
            financing_inputs["total_assets"] = total_assets
            financing_formula_str = "sum(liability_and_equity_components) == total_assets"

        if total_assets is None or calc_financing is None:
            missing_eq = []
            if total_assets is None:
                missing_eq.append("total_assets")
            if total_liab is None and total_liab_eq is None and not liability_equity_items:
                missing_eq.append("total_liabilities")
            if total_equity is None and total_liab_eq is None and not liability_equity_items:
                missing_eq.append("total_equity")

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
                    message=f"Fundamental accounting equation requires: {', '.join(missing_eq)}.",
                    missing_fields=missing_eq,
                )
            )
        else:
            var = round(calc_financing - total_assets, 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
            msg = (
                f"Balance sheet accounting equation balanced: Total Liabilities & Equity ({calc_financing}) == Total Assets ({total_assets})."
                if status == ValidationStatus.PASS
                else f"Balance sheet accounting equation out of balance: calculated {calc_financing} != Total Assets {total_assets} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_accounting_equation",
                    formula=financing_formula_str,
                    input_values=financing_inputs,
                    calculated_value=calc_financing,
                    reported_value=total_assets,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # -----------------------------------------------------------------------
        # Check 2: Total Assets Components Reconciliation
        # -----------------------------------------------------------------------
        if curr_assets is not None and non_curr_assets is not None and total_assets is not None:
            calc_ta = round(curr_assets + non_curr_assets, 4)
            var = round(calc_ta - total_assets, 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
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
        elif asset_items and len(asset_items) >= 2 and total_assets is not None:
            calc_ta = round(sum(amt for _, amt in asset_items), 4)
            var = round(calc_ta - total_assets, 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
            inputs_assets = {name: amt for name, amt in asset_items}
            inputs_assets["total_assets"] = total_assets
            formula_assets = " + ".join(name for name, _ in asset_items) + " == total_assets"
            msg = (
                f"Asset components sum correctly ({calc_ta} == {total_assets})."
                if status == ValidationStatus.PASS
                else f"Asset components mismatch: calculated sum {calc_ta} != reported total assets {total_assets} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_assets_components",
                    formula=formula_assets,
                    input_values=inputs_assets,
                    calculated_value=calc_ta,
                    reported_value=total_assets,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )
        elif curr_assets is not None or non_curr_assets is not None:
            missing_assets = []
            if curr_assets is None:
                missing_assets.append("current_assets")
            if non_curr_assets is None:
                missing_assets.append("non_current_assets")
            if total_assets is None:
                missing_assets.append("total_assets")
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
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_assets_components",
                    formula="sum(asset_components) == total_assets",
                    input_values={"total_assets": total_assets},
                    calculated_value=None,
                    reported_value=total_assets,
                    variance=None,
                    status=ValidationStatus.NOT_APPLICABLE,
                    tolerance=tol,
                    message="Asset components breakdown not reported in document.",
                    missing_fields=["asset_components"],
                )
            )

        # -----------------------------------------------------------------------
        # Check 3: Total Liabilities and Equity Components Reconciliation
        # -----------------------------------------------------------------------
        target_liab_eq = total_liab_eq if total_liab_eq is not None else total_assets
        if liability_equity_items and len(liability_equity_items) >= 2 and target_liab_eq is not None:
            calc_tle = round(sum(amt for _, amt in liability_equity_items), 4)
            var = round(calc_tle - target_liab_eq, 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
            inputs_le = {name: amt for name, amt in liability_equity_items}
            inputs_le["total_capital_and_liabilities"] = target_liab_eq
            formula_le = " + ".join(name for name, _ in liability_equity_items) + " == total_capital_and_liabilities"
            msg = (
                f"Liability and equity components sum correctly ({calc_tle} == {target_liab_eq})."
                if status == ValidationStatus.PASS
                else f"Liability and equity components mismatch: calculated sum {calc_tle} != reported {target_liab_eq} (variance: {var})."
            )
            checks.append(
                ValidationCheckResult(
                    rule_name="balance_sheet_total_liabilities_and_equity_components",
                    formula=formula_le,
                    input_values=inputs_le,
                    calculated_value=calc_tle,
                    reported_value=target_liab_eq,
                    variance=var,
                    status=status,
                    tolerance=tol,
                    message=msg,
                    missing_fields=[],
                )
            )

        # Check 4: Corporate Liabilities and Equity Reconciliation (when corporate total_liabilities and total_equity are reported)
        if total_liab_eq is not None and total_liab is not None and total_equity is not None:
            calc_le = round(total_liab + total_equity, 4)
            var = round(calc_le - total_liab_eq, 4)
            status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
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

        # Check 5: Current + Non-Current Liabilities == Total Liabilities (when corporate components reported)
        if (curr_liab is not None and non_curr_liab is not None) or (not liability_equity_items and (curr_liab is not None or non_curr_liab is not None)):
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
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
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

        # Check 6: Retained Earnings + Share Capital == Total Equity (when corporate equity components reported)
        if (retained is not None and share_cap is not None) or (not liability_equity_items and (retained is not None or share_cap is not None)):
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
                status = ValidationStatus.PASS if abs(var) <= tol else ValidationStatus.FAILED
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
