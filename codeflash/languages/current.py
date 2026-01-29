"""Singleton for the current language being used in the codeflash session.

This module provides a centralized way to access and set the current language
throughout the codeflash codebase, eliminating scattered language checks and
string comparisons.

Usage:
    from codeflash.languages import current_language, set_current_language, is_python

    # Set the language at the start of a session
    set_current_language(Language.PYTHON)
    # or
    set_current_language("javascript")

    # Check the current language anywhere in the codebase
    if is_python():
        # Python-specific code
        ...

    # Get the current language
    lang = current_language()

    # Get language support for the current language
    support = current_language_support()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.languages.base import Language

if TYPE_CHECKING:
    from codeflash.languages.base import LanguageSupport

# Module-level singleton for the current language
_current_language: Language | None = None


def current_language() -> Language:
    """Get the current language being used in this codeflash session.

    Returns:
        The current Language enum value.

    """
    return _current_language


def set_current_language(language: Language | str) -> None:
    """Set the current language for this codeflash session.

    This should be called once at the start of an optimization run,
    typically after reading the project configuration.

    Args:
        language: Either a Language enum value or a string like "python", "javascript", "typescript".

    """
    global _current_language

    if _current_language is not None:
        return
    _current_language = Language(language) if isinstance(language, str) else language


def reset_current_language() -> None:
    """Reset the current language to the default (Python).

    Useful for testing or when starting a new session.
    """
    global _current_language
    _current_language = Language.PYTHON


def is_python() -> bool:
    """Check if the current language is Python.

    Returns:
        True if the current language is Python.

    """
    return _current_language == Language.PYTHON


def is_javascript() -> bool:
    """Check if the current language is JavaScript or TypeScript.

    This returns True for both JavaScript and TypeScript since they are
    typically treated the same way in the optimization pipeline.

    Returns:
        True if the current language is JavaScript or TypeScript.

    """
    return _current_language in (Language.JAVASCRIPT, Language.TYPESCRIPT)


def is_typescript() -> bool:
    """Check if the current language is TypeScript specifically.

    Returns:
        True if the current language is TypeScript.

    """
    return _current_language == Language.TYPESCRIPT


def current_language_support() -> LanguageSupport:
    """Get the LanguageSupport instance for the current language.

    Returns:
        The LanguageSupport instance for the current language.

    """
    from codeflash.languages.registry import get_language_support

    return get_language_support(_current_language)
