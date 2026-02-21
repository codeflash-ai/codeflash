"""Language registry for multi-language support.

This module provides functions for registering, detecting, and retrieving
language support implementations. It maintains a registry of all available
language implementations and provides utilities for language detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.languages.language_enum import Language

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeflash.languages.base import LanguageSupport

logger = logging.getLogger(__name__)


# Registry mapping file extensions to language support classes
_EXTENSION_REGISTRY: dict[str, type[LanguageSupport]] = {}

# Registry mapping Language enum to language support classes
_LANGUAGE_REGISTRY: dict[Language, type[LanguageSupport]] = {}

# Cache of instantiated language support objects
_SUPPORT_CACHE: dict[Language, LanguageSupport] = {}

# Flag to track if language modules have been imported
_languages_registered = False


def _ensure_languages_registered() -> None:
    """Ensure all language support modules are imported and registered.

    This lazily imports the language support modules to avoid circular imports
    at module load time. The imports trigger the @register_language decorators
    which populate the registries.
    """
    global _languages_registered
    if _languages_registered:
        return

    # Import support modules to trigger registration
    # These imports are deferred to avoid circular imports
    import contextlib
    import importlib

    with contextlib.suppress(ImportError):
        importlib.import_module("codeflash.languages.python.support")

    with contextlib.suppress(ImportError):
        importlib.import_module("codeflash.languages.javascript.support")

    with contextlib.suppress(ImportError):
        importlib.import_module("codeflash.languages.java.support")

    _languages_registered = True


class UnsupportedLanguageError(Exception):
    """Raised when attempting to use an unsupported language."""

    def __init__(self, identifier: str | Path, supported: Iterable[str] | None = None) -> None:
        self.identifier = identifier
        self.supported = list(supported) if supported else []
        msg = f"Unsupported language: {identifier}"
        if self.supported:
            msg += f". Supported: {', '.join(self.supported)}"
        super().__init__(msg)


def register_language(cls: type[LanguageSupport]) -> type[LanguageSupport]:
    """Decorator to register a language support implementation.

    This decorator registers a language support class in both the extension
    registry (for file-based lookup) and the language registry (for direct lookup).

    Args:
        cls: The language support class to register.

    Returns:
        The same class (unmodified).

    Example:
        @register_language
        class PythonSupport(LanguageSupport):
            @property
            def language(self) -> Language:
                return Language.PYTHON

            @property
            def file_extensions(self) -> tuple[str, ...]:
                return (".py", ".pyw")

            # ... other methods

    """
    # Create a temporary instance to get language and extensions
    # Note: This requires the class to be instantiable without arguments
    try:
        instance = cls()
        language = instance.language
        extensions = instance.file_extensions
    except Exception as e:
        msg = (
            f"Failed to instantiate {cls.__name__} for registration. "
            f"Language support classes must be instantiable without arguments. "
            f"Error: {e}"
        )
        raise ValueError(msg) from e

    # Register by extension
    for ext in extensions:
        ext_lower = ext.lower()
        if ext_lower in _EXTENSION_REGISTRY:
            existing = _EXTENSION_REGISTRY[ext_lower]
            logger.warning(
                "Extension '%s' already registered to %s, overwriting with %s", ext, existing.__name__, cls.__name__
            )
        _EXTENSION_REGISTRY[ext_lower] = cls

    # Register by language
    if language in _LANGUAGE_REGISTRY:
        existing = _LANGUAGE_REGISTRY[language]
        logger.warning(
            "Language '%s' already registered to %s, overwriting with %s", language, existing.__name__, cls.__name__
        )
    _LANGUAGE_REGISTRY[language] = cls

    logger.debug("Registered %s for language '%s' with extensions %s", cls.__name__, language, extensions)

    return cls


def get_language_support(identifier: Path | Language | str) -> LanguageSupport:
    """Get language support for a file, language, or extension.

    This function accepts multiple identifier types:
    - Path: Uses file extension to determine language
    - Language enum: Direct lookup
    - str: Interpreted as extension or language name

    Args:
        identifier: File path, Language enum, or extension/language string.

    Returns:
        LanguageSupport instance for the identified language.

    Raises:
        UnsupportedLanguageError: If the language is not supported.

    Note:
        This function lazily imports language support modules on first call
        to avoid circular import issues at module load time.

    Example:
        # By file path
        lang = get_language_support(Path("example.py"))

        # By Language enum
        lang = get_language_support(Language.PYTHON)

        # By extension
        lang = get_language_support(".py")

        # By language name
        lang = get_language_support("python")

    """
    _ensure_languages_registered()
    language: Language | None = None

    if isinstance(identifier, Language):
        language = identifier

    elif isinstance(identifier, Path):
        ext = identifier.suffix.lower()
        if ext not in _EXTENSION_REGISTRY:
            raise UnsupportedLanguageError(identifier, get_supported_extensions())
        cls = _EXTENSION_REGISTRY[ext]
        language = cls().language

    elif isinstance(identifier, str):
        # Try as extension first
        ext = identifier.lower() if identifier.startswith(".") else f".{identifier.lower()}"
        if ext in _EXTENSION_REGISTRY:
            cls = _EXTENSION_REGISTRY[ext]
            language = cls().language
        else:
            # Try as language name
            try:
                language = Language(identifier.lower())
            except ValueError:
                raise UnsupportedLanguageError(identifier, get_supported_languages()) from None

    if language is None:
        raise UnsupportedLanguageError(str(identifier), get_supported_languages())

    # Return cached instance or create new one
    if language not in _SUPPORT_CACHE:
        if language not in _LANGUAGE_REGISTRY:
            raise UnsupportedLanguageError(str(language), get_supported_languages())
        _SUPPORT_CACHE[language] = _LANGUAGE_REGISTRY[language]()

    return _SUPPORT_CACHE[language]


# Cache for test framework to language support mapping
_FRAMEWORK_CACHE: dict[str, LanguageSupport] = {}


def get_language_support_by_common_formatters(formatter_cmd: str | list[str]) -> LanguageSupport | None:
    _ensure_languages_registered()
    language: Language | None = None
    if isinstance(formatter_cmd, str):
        formatter_cmd = [formatter_cmd]

    if len(formatter_cmd) == 1:
        formatter_cmd = formatter_cmd[0].split(" ")

    # Try as extension first
    ext = None

    py_formatters = ["black", "isort", "ruff", "autopep8", "yapf", "pyfmt"]
    js_ts_formatters = ["prettier", "eslint", "biome", "rome", "deno", "standard", "tslint"]

    if any(cmd in py_formatters for cmd in formatter_cmd):
        ext = ".py"
    elif any(cmd in js_ts_formatters for cmd in formatter_cmd):
        ext = ".js"

    if ext is None:
        # can't determine language
        return None

    cls = _EXTENSION_REGISTRY[ext]
    language = cls().language

    # Return cached instance or create new one
    if language not in _SUPPORT_CACHE:
        if language not in _LANGUAGE_REGISTRY:
            raise UnsupportedLanguageError(str(language), get_supported_languages())
        _SUPPORT_CACHE[language] = _LANGUAGE_REGISTRY[language]()

    return _SUPPORT_CACHE[language]


def get_language_support_by_framework(test_framework: str) -> LanguageSupport | None:
    """Get language support for a test framework.

    This function looks up the language support implementation that uses
    the specified test framework.

    Args:
        test_framework: Name of the test framework (e.g., "jest", "pytest").

    Returns:
        LanguageSupport instance for the test framework, or None if not found.

    Example:
        # Get Jest language support
        lang = get_language_support_by_framework("jest")
        if lang:
            result = lang.run_behavioral_tests(...)

    """
    # Check cache first
    if test_framework in _FRAMEWORK_CACHE:
        return _FRAMEWORK_CACHE[test_framework]

    # Map of frameworks that should use the same language support
    # All Java test frameworks (junit4, junit5, testng) use the Java language support
    framework_aliases = {
        "junit4": "junit5",  # JUnit 4 uses Java support (which reports junit5 as primary)
        "testng": "junit5",  # TestNG also uses Java support
    }

    # Use the canonical framework name for lookup
    lookup_framework = framework_aliases.get(test_framework, test_framework)

    # Search all registered languages for one with matching test framework
    for language in _LANGUAGE_REGISTRY:
        support = get_language_support(language)
        if hasattr(support, "test_framework") and support.test_framework == lookup_framework:
            _FRAMEWORK_CACHE[test_framework] = support
            return support

    return None


def detect_project_language(project_root: Path, module_root: Path) -> Language:
    """Detect the primary language of a project by analyzing file extensions.

    Counts files by extension in the module root and returns the most
    common supported language.

    Args:
        project_root: Root directory of the project.
        module_root: Root directory of the module to analyze.

    Returns:
        The detected Language.

    Raises:
        UnsupportedLanguageError: If no supported language is detected.

    """
    _ensure_languages_registered()
    extension_counts: dict[str, int] = {}

    # Count files by extension
    for file in module_root.rglob("*"):
        if file.is_file():
            ext = file.suffix.lower()
            if ext:
                extension_counts[ext] = extension_counts.get(ext, 0) + 1

    # Find the most common supported extension
    for ext, count in sorted(extension_counts.items(), key=lambda x: -x[1]):
        if ext in _EXTENSION_REGISTRY:
            cls = _EXTENSION_REGISTRY[ext]
            logger.info("Detected language: %s (found %d '%s' files)", cls().language, count, ext)
            return cls().language

    msg = f"No supported language detected in {module_root}"
    raise UnsupportedLanguageError(msg, get_supported_languages())


def get_supported_languages() -> list[str]:
    """Get list of supported language names.

    Returns:
        List of language name strings.

    """
    _ensure_languages_registered()
    return [lang.value for lang in _LANGUAGE_REGISTRY]


def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions.

    Returns:
        List of extension strings (with leading dots).

    """
    _ensure_languages_registered()
    return list(_EXTENSION_REGISTRY.keys())


def is_language_supported(identifier: Path | Language | str) -> bool:
    """Check if a language/extension is supported.

    Args:
        identifier: File path, Language enum, or extension/language string.

    Returns:
        True if supported, False otherwise.

    """
    try:
        get_language_support(identifier)
        return True
    except UnsupportedLanguageError:
        return False


def clear_registry() -> None:
    """Clear all registered languages.

    Primarily useful for testing.
    """
    global _languages_registered
    _EXTENSION_REGISTRY.clear()
    _LANGUAGE_REGISTRY.clear()
    _SUPPORT_CACHE.clear()
    _FRAMEWORK_CACHE.clear()
    _languages_registered = False


def clear_cache() -> None:
    """Clear the language support instance cache.

    Useful if you need fresh instances of language support objects.
    """
    _SUPPORT_CACHE.clear()
    _FRAMEWORK_CACHE.clear()
