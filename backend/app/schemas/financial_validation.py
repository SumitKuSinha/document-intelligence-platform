"""
Financial Validation Schemas

Defines Pydantic models for deterministic mathematical validation results:
- Validation status (PASS, FAILED, NOT_APPLICABLE)
- Individual mathematical check results
- Summary counts (total, passed, failed, not applicable)
- Comprehensive validation result envelope
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    """Status of an individual validation check or overall result."""

    PASS = "PASS"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ValidationCheckResult(BaseModel):
    """
    Detailed outcome of a single deterministic mathematical check.

    Attributes:
        rule_name: Machine-readable identifier of the validation rule.
        formula: Formula or mathematical relationship being evaluated.
        input_values: Dictionary of extracted/parsed values used in the check.
        calculated_value: Deterministically computed value.
        reported_value: Stated value extracted from the document.
        variance: Numerical difference (calculated_value - reported_value).
        status: Status of the check ('PASS', 'FAILED', or 'NOT_APPLICABLE').
        tolerance: Rounding tolerance applied to the check.
        message: Human-readable explanation of the outcome.
        missing_fields: List of required field names that were missing/null.
    """

    rule_name: str = Field(..., description="Unique name of the validation check")
    formula: str = Field(..., description="Mathematical relationship or equation evaluated")
    input_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted/parsed values supplied to the check",
    )
    calculated_value: Optional[float] = Field(
        default=None,
        description="Value deterministically calculated by the rule",
    )
    reported_value: Optional[float] = Field(
        default=None,
        description="Value extracted/reported in the source document",
    )
    variance: Optional[float] = Field(
        default=None,
        description="Numerical variance (calculated_value - reported_value)",
    )
    status: ValidationStatus = Field(..., description="Outcome: PASS, FAILED, or NOT_APPLICABLE")
    tolerance: float = Field(default=0.01, description="Acceptable variance threshold")
    message: Optional[str] = Field(default=None, description="Detailed explanatory message")
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Required fields that were null/missing when status is NOT_APPLICABLE",
    )


class FinancialValidationSummary(BaseModel):
    """Summary counts of validation checks performed on a document."""

    total_checks: int = Field(0, description="Total number of checks evaluated")
    passed_checks: int = Field(0, description="Number of checks with status PASS")
    failed_checks: int = Field(0, description="Number of checks with status FAILED")
    not_applicable_checks: int = Field(0, description="Number of checks with status NOT_APPLICABLE")
    is_valid: bool = Field(
        True,
        description="True if zero checks failed, False if any check failed",
    )


class FinancialValidationResult(BaseModel):
    """
    Composite result of the Financial Validation Engine.

    Suitable for:
    - Returning in API responses.
    - Persisting alongside extraction data.
    - Displaying in frontend reconciliation dashboards.
    """

    document_type: str = Field(..., description="Financial document category validated")
    is_valid: bool = Field(
        ...,
        description="True if all applicable checks passed without failures",
    )
    summary: FinancialValidationSummary = Field(
        ...,
        description="Aggregation of validation check statuses",
    )
    checks: List[ValidationCheckResult] = Field(
        default_factory=list,
        description="List of all individual mathematical checks executed",
    )
    errors: List[str] = Field(default_factory=list, description="Processing errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings or notes")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a standard Python dictionary."""
        return self.model_dump()
