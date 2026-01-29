"""Code normalizers for different programming languages.

This module provides language-specific code normalizers that transform source code
into canonical forms for duplicate detection. The normalizers:
- Replace local variable names with canonical forms (var_0, var_1, etc.)
- Preserve function names, class names, parameters, and imports
- Remove or normalize comments and docstrings
- Produce consistent output for structurally identical code

Usage:
    >>> normalizer = get_normalizer("python")
    >>> normalized = normalizer.normalize(code)
    >>> fingerprint = normalizer.get_fingerprint(code)
    >>> are_same = normalizer.are_duplicates(code1, code2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.code_utils.normalizers.base import CodeNormalizer
from codeflash.code_utils.normalizers.javascript import JavaScriptNormalizer, TypeScriptNormalizer
from codeflash.code_utils.normalizers.python import PythonNormalizer

if TYPE_CHECKING:
    pass

__all__ = [
    "CodeNormalizer",
    "PythonNormalizer",
    "JavaScriptNormalizer",
    "TypeScriptNormalizer",
    "get_normalizer",
    "get_normalizer_for_extension",
]

# Registry of normalizers by language
_NORMALIZERS: dict[str, type[CodeNormalizer]] = {
    "python": PythonNormalizer,
    "javascript": JavaScriptNormalizer,
    "typescript": TypeScriptNormalizer,
}

# Singleton cache for normalizer instances
_normalizer_instances: dict[str, CodeNormalizer] = {}


def get_normalizer(language: str) -> CodeNormalizer:
    """Get a code normalizer for the specified language.

    Args:
        language: Language name ('python', 'javascript', 'typescript')

    Returns:
        CodeNormalizer instance for the language

    Raises:
        ValueError: If no normalizer exists for the language
    """
    language = language.lower()

    # Check cache first
    if language in _normalizer_instances:
        return _normalizer_instances[language]

    # Get normalizer class
    if language not in _NORMALIZERS:
        supported = ", ".join(sorted(_NORMALIZERS.keys()))
        msg = f"No normalizer available for language '{language}'. Supported: {supported}"
        raise ValueError(msg)

    # Create and cache instance
    normalizer = _NORMALIZERS[language]()
    _normalizer_instances[language] = normalizer
    return normalizer


def get_normalizer_for_extension(extension: str) -> CodeNormalizer | None:
    """Get a code normalizer based on file extension.

    Args:
        extension: File extension including dot (e.g., '.py', '.js')

    Returns:
        CodeNormalizer instance if found, None otherwise
    """
    extension = extension.lower()
    if not extension.startswith("."):
        extension = f".{extension}"

    for language, normalizer_class in _NORMALIZERS.items():
        normalizer = get_normalizer(language)
        if extension in normalizer.supported_extensions:
            return normalizer

    return None


def register_normalizer(language: str, normalizer_class: type[CodeNormalizer]) -> None:
    """Register a new normalizer for a language.

    Args:
        language: Language name
        normalizer_class: CodeNormalizer subclass
    """
    _NORMALIZERS[language.lower()] = normalizer_class
    # Clear cached instance if it exists
    _normalizer_instances.pop(language.lower(), None)
