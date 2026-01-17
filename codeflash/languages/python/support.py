"""Python language support implementation."""

from __future__ import annotations

import logging

from codeflash.languages.base import Language
from codeflash.languages.registry import register_language

logger = logging.getLogger(__name__)


@register_language
class PythonSupport:
    """Python language support implementation.

    This class wraps the existing Python-specific implementations to conform
    to the LanguageSupport protocol. It delegates to existing code where possible
    to maintain backward compatibility.
    """

    # === Properties ===

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.PYTHON

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by Python."""
        return (".py", ".pyw")

    @property
    def test_framework(self) -> str:
        """Primary test framework for Python."""
        return "pytest"

    @property
    def comment_prefix(self) -> str:
        return "#"
