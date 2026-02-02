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
    HelperFunction,
    Language,
    LanguageSupport,
    ParentInfo,
    TestInfo,
    TestResult,
)


# Lazy import for FunctionInfo to avoid circular imports
def __getattr__(name: str):
    if name == "FunctionInfo":
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize

        return FunctionToOptimize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
from codeflash.languages.test_framework import (
    current_test_framework,
    get_js_test_framework_or_default,
    is_jest,
    is_mocha,
    is_pytest,
    is_unittest,
    is_vitest,
    reset_test_framework,
    set_current_test_framework,
)

__all__ = [
    "CodeContext",
    "FunctionInfo",
    "HelperFunction",
    "Language",
    "LanguageSupport",
    "ParentInfo",
    "TestInfo",
    "TestResult",
    "current_language",
    "current_language_support",
    "current_test_framework",
    "detect_project_language",
    "get_js_test_framework_or_default",
    "get_language_support",
    "get_supported_extensions",
    "get_supported_languages",
    "is_javascript",
    "is_jest",
    "is_mocha",
    "is_pytest",
    "is_python",
    "is_typescript",
    "is_unittest",
    "is_vitest",
    "register_language",
    "reset_current_language",
    "reset_test_framework",
    "set_current_language",
    "set_current_test_framework",
]
