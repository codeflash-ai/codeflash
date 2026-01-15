"""
Multi-language support for Codeflash.

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
from codeflash.languages.registry import (
    detect_project_language,
    get_language_support,
    get_supported_extensions,
    get_supported_languages,
    register_language,
)

__all__ = [
    # Base types
    "Language",
    "LanguageSupport",
    "FunctionInfo",
    "ParentInfo",
    "CodeContext",
    "HelperFunction",
    "TestResult",
    "TestInfo",
    # Registry functions
    "get_language_support",
    "detect_project_language",
    "register_language",
    "get_supported_languages",
    "get_supported_extensions",
]
