"""
File Validation Schemas

Data schemas representing structured file validation results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Structured result returned after validating an uploaded document file.

    Attributes:
        is_valid: True if the file passed all integrity, format, and constraint checks.
        file_type: Detected supported format ('pdf', 'png', 'jpeg'), or None if invalid/unsupported.
        page_count: Number of pages detected (for PDFs); None for images or unreadable files.
        errors: List of critical validation error descriptions explaining why a file was rejected.
        warnings: List of non-fatal warning messages detected during inspection.
    """

    is_valid: bool
    file_type: Optional[str] = None
    page_count: Optional[int] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary representation."""
        return self.model_dump()
