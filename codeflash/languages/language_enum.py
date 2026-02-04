"""Language enum for multi-language support.

This module is kept separate to avoid circular imports.
"""

from enum import Enum


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"

    def __str__(self) -> str:
        return self.value
