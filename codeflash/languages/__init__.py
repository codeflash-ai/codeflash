"""Multi-language support for Codeflash.

This package provides the abstraction layer that allows Codeflash to support
multiple programming languages while keeping the core optimization pipeline
language-agnostic.

Usage:
    from codeflash.languages import get_language_support, Language

    # Get language support for a file
    lang = get_language_support(Path("example.py"))

    # Discover functions
    functions = lang.discover_functions(file_path)

    # Replace a function
    new_source = lang.replace_function(file_path, function, new_code)
"""

from codeflash.languages.base import (
    CodeContext,
    FunctionInfo,
    HelperFunction,
    Language,
    LanguageSupport,
    ParentInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.current import (
    current_language,
    current_language_support,
    is_javascript,
    is_python,
    is_typescript,
    reset_current_language,
    set_current_language,
)
from codeflash.languages.javascript import JavaScriptSupport, TypeScriptSupport  # noqa: F401

# Import language support modules to trigger auto-registration
# This ensures all supported languages are available when this package is imported
from codeflash.languages.python import PythonSupport  # noqa: F401
from codeflash.languages.registry import (
    detect_project_language,
    get_language_support,
    get_supported_extensions,
    get_supported_languages,
    register_language,
)

__all__ = [
    # Base types
    "CodeContext",
    "FunctionInfo",
    "HelperFunction",
    "Language",
    "LanguageSupport",
    "ParentInfo",
    "TestInfo",
    "TestResult",
    # Current language singleton
    "current_language",
    "current_language_support",
    "is_javascript",
    "is_python",
    "is_typescript",
    "reset_current_language",
    "set_current_language",
    # Registry functions
    "detect_project_language",
    "get_language_support",
    "get_supported_extensions",
    "get_supported_languages",
    "register_language",
]
