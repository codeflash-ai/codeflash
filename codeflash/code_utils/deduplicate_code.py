"""Code deduplication utilities using language-specific normalizers.

This module provides functions to normalize code, generate fingerprints,
and detect duplicate code segments across different programming languages.
"""

from __future__ import annotations

import hashlib
import re

from codeflash.code_utils.normalizers import get_normalizer
from codeflash.languages import current_language, is_python


def normalize_code(
    code: str,
    remove_docstrings: bool = True,
    return_ast_dump: bool = False,
    language: str | None = None,
) -> str:
    """Normalize code by parsing, cleaning, and normalizing variable names.

    Function names, class names, and parameters are preserved.

    Args:
        code: Source code as string
        remove_docstrings: Whether to remove docstrings (Python only)
        return_ast_dump: Return AST dump instead of unparsed code (Python only)
        language: Language of the code. If None, uses the current session language.

    Returns:
        Normalized code as string

    """
    if language is None:
        language = current_language().value

    try:
        normalizer = get_normalizer(language)

        # Python has additional options
        if is_python():
            if return_ast_dump:
                return normalizer.normalize_for_hash(code)
            return normalizer.normalize(code, remove_docstrings=remove_docstrings)

        # For other languages, use standard normalization
        return normalizer.normalize(code)
    except ValueError:
        # Unknown language - fall back to basic normalization
        return _basic_normalize(code)
    except Exception:
        # Parsing error - try other languages or fall back
        if is_python():
            # Try JavaScript as fallback
            try:
                js_normalizer = get_normalizer("javascript")
                js_result = js_normalizer.normalize(code)
                if js_result != _basic_normalize(code):
                    return js_result
            except Exception:
                pass
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
