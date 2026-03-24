"""Multi-language support for Codeflash.

This package provides the abstraction layer that allows Codeflash to support
multiple programming languages while keeping the core optimization pipeline
language-agnostic.

Usage:
    from codeflash.languages import get_language_support, Language

    # Get language support for a file
    lang = get_language_support(Path("example.py"))

    # Discover functions
    functions = lang.discover_functions(source, file_path)

    # Replace a function
    new_source = lang.replace_function(file_path, function, new_code)
"""

import importlib

from codeflash.languages.base import (
    CodeContext,
    DependencyResolver,
    HelperFunction,
    IndexResult,
    Language,
    LanguageSupport,
    ParentInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.current import (
    current_language,
    current_language_support,
    is_java,
    is_javascript,
    is_python,
    is_typescript,
    reset_current_language,
    set_current_language,
)

# Language support modules are imported lazily to avoid circular imports
# They get registered when first accessed via get_language_support()
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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FunctionInfo": ("codeflash_core.models", "FunctionToOptimize"),
    "JavaScriptSupport": ("codeflash.languages.javascript.support", "JavaScriptSupport"),
    "TypeScriptSupport": ("codeflash.languages.javascript.support", "TypeScriptSupport"),
    "PythonSupport": ("codeflash.languages.python.support", "PythonSupport"),
    "JavaSupport": ("codeflash.languages.java.support", "JavaSupport"),
}

_LAZY_CACHE: dict[str, object] = {}


# Lazy imports to avoid circular imports
def __getattr__(name: str):
    # Fast path: return cached value if already resolved
    try:
        return _LAZY_CACHE[name]
    except KeyError:
        pass

    importer = _LAZY_IMPORTS.get(name)
    if importer is not None:
        module_path, attr_name = importer
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache both in module globals (so future attribute access avoids __getattr__)
        # and in the local cache to speed subsequent lookups in this function.
        globals()[name] = value
        _LAZY_CACHE[name] = value
        return value

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "CodeContext",
    "DependencyResolver",
    "FunctionInfo",
    "HelperFunction",
    "IndexResult",
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
    "is_java",
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
