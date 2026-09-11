"""
Financial Document Extraction Schemas

Defines Pydantic models for structured financial document extraction,
supporting invoices, balance sheets, profit & loss statements, and cash flow statements.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class FinancialDocumentType(str, Enum):
    """Supported financial document categories."""

    INVOICE = "invoice"
    BALANCE_SHEET = "balance_sheet"
    PROFIT_AND_LOSS = "profit_and_loss"
    CASH_FLOW_STATEMENT = "cash_flow_statement"


class FieldEvidence(BaseModel):
    """
    Source traceability evidence for an extracted field.

    Attributes:
        source_snippet: Exact substring from the document where the value was found.
        page_number: 1-indexed page number containing the evidence snippet.
    """

    source_snippet: Optional[str] = None
    page_number: Optional[int] = None


class InvoiceLineItem(BaseModel):
    """Individual line item from an invoice with explicit generic accounting semantics."""

    model_config = ConfigDict(populate_by_name=True)

    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    discount_percent: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("discount_percent", "discount_pct", "discount_rate"),
        description="Explicit percentage discount on the line item",
    )
    discount_amount: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("discount_amount", "discount_value", "discount"),
        description="Explicit monetary discount amount on the line item",
    )
    tax_rate: Optional[float] = Field(
        default=None,
        description="Line-item statutory tax rate (e.g. 0.18 for 18%)",
    )
    tax_amount: Optional[float] = Field(
        default=None,
        description="Line-item statutory tax amount if reported",
    )
    total_price: Optional[float] = Field(
        default=None,
        description="Net or reported line total (quantity * unit_price [- discount])",
    )
    gross_amount: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("gross_amount", "gross_total", "gross_worth"),
        description="Gross / post-tax line total (Gross worth) including tax if present",
    )
    is_tax_inclusive: Optional[bool] = Field(
        default=None,
        description="True if unit_price / total_price includes tax",
    )
    item_code: Optional[str] = None
    source_snippet: Optional[str] = None
    page_number: Optional[int] = None


class FinancialLineItem(BaseModel):
    """Generic financial statement line item (for balance sheet, P&L, cash flow)."""

    item_name: str = Field(..., description="Name of the financial account or line item")
    amount: Optional[float] = None
    category: Optional[str] = None
    source_snippet: Optional[str] = None
    page_number: Optional[int] = None


# ---------------------------------------------------------------------------
# Specific Extraction Data Payloads
# ---------------------------------------------------------------------------

class InvoiceExtractionData(BaseModel):
    """Structured extraction payload for invoice documents with generic accounting semantics."""

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    purchase_order_number: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_tax_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    pricing_type: Optional[str] = Field(
        default=None,
        description="Pricing model: 'tax_exclusive' (net line totals + tax = total) or 'tax_inclusive' (gross line totals include tax)",
    )
    subtotal: Optional[float] = Field(
        default=None,
        description="Pre-tax / net subtotal (sum of net line totals) or taxable base",
    )
    taxable_amount: Optional[float] = Field(
        default=None,
        description="Explicit taxable base/value if reported separately from subtotal",
    )
    tax_amount: Optional[float] = Field(
        default=None,
        description="Total tax amount",
    )
    tax_rate: Optional[float] = Field(
        default=None,
        description="Effective overall tax rate (e.g. 0.18 for 18%)",
    )
    tax_components: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Explicit breakdown of statutory tax components (e.g. CGST, SGST, IGST, VAT, CESS)",
    )
    discount_amount: Optional[float] = Field(
        default=None,
        description="Total monetary discount (order-level or total discount summary)",
    )
    discount_scope: Optional[str] = Field(
        default=None,
        description="'invoice_level' (deducted from subtotal) or 'line_item_summary' (informational summary of line discounts)",
    )
    shipping_amount: Optional[float] = None
    round_off_amount: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("round_off_amount", "round_off", "rounding"),
        description="Explicit round-off adjustment added/subtracted to reach final total",
    )
    total_amount: Optional[float] = Field(
        default=None,
        description="Final gross amount payable (Grand Total)",
    )
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    line_items: List[InvoiceLineItem] = Field(default_factory=list)
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


class BalanceSheetExtractionData(BaseModel):
    """Structured extraction payload for balance sheet documents with generic accounting semantics."""

    model_config = ConfigDict(populate_by_name=True)

    company_name: Optional[str] = None
    statement_date: Optional[str] = None
    reporting_period: Optional[str] = None
    currency: Optional[str] = None
    current_assets: Optional[float] = None
    non_current_assets: Optional[float] = None
    total_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    non_current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    retained_earnings: Optional[float] = None
    share_capital: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("share_capital", "capital", "equity_capital"),
    )
    total_equity: Optional[float] = None
    total_liabilities_and_equity: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "total_liabilities_and_equity",
            "total_capital_and_liabilities",
            "total_equity_and_liabilities",
            "total_capital_liabilities",
        ),
    )
    line_items: List[FinancialLineItem] = Field(default_factory=list)
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


class ProfitAndLossExtractionData(BaseModel):
    """Structured extraction payload for Profit & Loss / Income Statements."""

    company_name: Optional[str] = None
    reporting_period: Optional[str] = None
    period_start_date: Optional[str] = None
    period_end_date: Optional[str] = None
    currency: Optional[str] = None
    total_revenue: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    interest_expense: Optional[float] = None
    tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    line_items: List[FinancialLineItem] = Field(default_factory=list)
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


class CashFlowExtractionData(BaseModel):
    """Structured extraction payload for cash flow statements."""

    company_name: Optional[str] = None
    reporting_period: Optional[str] = None
    period_start_date: Optional[str] = None
    period_end_date: Optional[str] = None
    currency: Optional[str] = None
    net_cash_from_operating_activities: Optional[float] = None
    net_cash_from_investing_activities: Optional[float] = None
    net_cash_from_financing_activities: Optional[float] = None
    net_change_in_cash: Optional[float] = None
    beginning_cash_balance: Optional[float] = None
    ending_cash_balance: Optional[float] = None
    line_items: List[FinancialLineItem] = Field(default_factory=list)
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Composite Extraction Result
# ---------------------------------------------------------------------------

class FinancialExtractionResult(BaseModel):
    """
    Final output produced by the Financial Extraction Service.

    Suitable for:
    - Returning in API responses.
    - Persisting directly into Document.extracted_data (PostgreSQL JSONB).
    - Feeding into the downstream financial validation engine.
    """

    document_type: str = Field(..., description="Document type processed")
    status: str = Field(..., description="Outcome status: 'SUCCESS', 'PARTIAL', or 'FAILED'")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Type-specific structured financial data and line items",
    )
    evidence: Dict[str, FieldEvidence] = Field(
        default_factory=dict,
        description="Mapping of field names to source text snippets and page numbers",
    )
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a standard Python dictionary."""
        return self.model_dump()
