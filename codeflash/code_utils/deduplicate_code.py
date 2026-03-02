"""Code deduplication utilities using language-specific normalizers.

This module provides functions to normalize code, generate fingerprints,
and detect duplicate code segments across different programming languages.
"""

from __future__ import annotations

import hashlib
import re

from codeflash.code_utils.normalizers import get_normalizer
from codeflash.languages import current_language


def normalize_code(code: str, language: str | None = None) -> str:
    """Normalize code by parsing, cleaning, and normalizing variable names.

    Function names, class names, and parameters are preserved.

    Args:
        code: Source code as string
        language: Language of the code. If None, uses the current session language.

    Returns:
        Normalized code as string

    """
    if language is None:
        language = current_language().value

    try:
        return get_normalizer(language).normalize(code)
    except ValueError:
        return _basic_normalize(code)
    except Exception:
        return _basic_normalize(code)


def _basic_normalize(code: str) -> str:
    """Basic normalization: remove comments and normalize whitespace."""
    # Remove single-line comments (// and #)
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    # Normalize whitespace
    return " ".join(code.split())


def get_code_fingerprint(code: str, language: str | None = None) -> str:
    """Generate a fingerprint for normalized code.

    Args:
        code: Source code
        language: Language of the code. If None, uses the current session language.

    Returns:
        SHA-256 hash of normalized code

    """
    if language is None:
        language = current_language().value

    try:
        normalizer = get_normalizer(language)
        return normalizer.get_fingerprint(code)
    except ValueError:
        # Unknown language - use basic normalization
        normalized = _basic_normalize(code)
        return hashlib.sha256(normalized.encode()).hexdigest()


def are_codes_duplicate(code1: str, code2: str, language: str | None = None) -> bool:
    """Check if two code segments are duplicates after normalization.

    Args:
        code1: First code segment
        code2: Second code segment
        language: Language of the code. If None, uses the current session language.

    Returns:
        True if codes are structurally identical (ignoring local variable names)

    """
    if language is None:
        language = current_language().value

    try:
        normalizer = get_normalizer(language)
        return normalizer.are_duplicates(code1, code2)
    except ValueError:
        # Unknown language - use basic comparison
        return _basic_normalize(code1) == _basic_normalize(code2)
    except Exception:
        return False


# Re-export for backward compatibility
__all__ = ["are_codes_duplicate", "get_code_fingerprint", "normalize_code"]
