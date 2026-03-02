"""Language-agnostic schemas for AI service communication.

This module defines standardized payload schemas that work across all supported
languages (Python, JavaScript, TypeScript, and future languages).

Design principles:
1. General fields that apply to any language
2. Language-specific fields grouped in a nested object
3. Backward compatible with existing backend
4. Extensible for future languages without breaking changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModuleSystem(str, Enum):
    """Module system used by the code."""

    COMMONJS = "commonjs"  # JavaScript/Node.js require/exports
    ESM = "esm"  # ES Modules import/export
    PYTHON = "python"  # Python import system
    UNKNOWN = "unknown"


class TestFramework(str, Enum):
    """Supported test frameworks."""

    # Python
    PYTEST = "pytest"
    UNITTEST = "unittest"

    # JavaScript/TypeScript
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"


@dataclass
class LanguageInfo:
    """Language-specific information.

    General fields that describe the programming language and its environment.
    This is designed to be extensible for future languages.
    """

    # Core language identifier
    name: str  # "python", "javascript", "typescript", "rust", etc.

    # Language version (format varies by language)
    # - Python: "3.11.0"
    # - JavaScript/TypeScript: "ES2022", "ES2023"
    # - Rust: "1.70.0"
    version: str | None = None

    # Module system (primarily for JS/TS, but could apply to others)
    module_system: ModuleSystem = ModuleSystem.UNKNOWN

    # File extension (for generated files)
    # - Python: ".py"
    # - JavaScript: ".js", ".mjs", ".cjs"
    # - TypeScript: ".ts", ".mts", ".cts"
    file_extension: str = ""

    # Type system info (for typed languages)
    has_type_annotations: bool = False
    type_checker: str | None = None  # "mypy", "typescript", "pyright", etc.


@dataclass
class TestInfo:
    """Test-related information."""

    # Test framework being used
    framework: TestFramework

    # Timeout for test execution (seconds)
    timeout: int = 60

    # Test file path patterns (for discovery)
    test_patterns: list[str] = field(default_factory=list)

    # Path to test files relative to project root
    tests_root: str = "tests"


@dataclass
class OptimizeRequest:
    """Request payload for code optimization.

    This schema is designed to be language-agnostic while supporting
    language-specific fields through the `language_info` object.
    """

    # === Core required fields ===
    source_code: str  # Code to optimize
    trace_id: str  # Unique identifier for this optimization run

    # === Language information ===
    language_info: LanguageInfo

    # === Optional context ===
    dependency_code: str = ""  # Read-only context code
    module_path: str = ""  # Path to the module being optimized

    # === Function metadata ===
    is_async: bool = False  # Whether function is async/await
    is_numerical_code: bool | None = None  # Whether code does numerical computation

    # === Generation parameters ===
    n_candidates: int = 5  # Number of optimization candidates

    # === Metadata ===
    codeflash_version: str = ""
    experiment_metadata: dict[str, Any] | None = None
    repo_owner: str | None = None
    repo_name: str | None = None
    current_username: str | None = None

    # === React-specific ===
    is_react_component: bool = False
    react_context: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload dict, maintaining backward compatibility."""
        payload = {
            "source_code": self.source_code,
            "trace_id": self.trace_id,
            "language": self.language_info.name,
            "dependency_code": self.dependency_code,
            "is_async": self.is_async,
            "n_candidates": self.n_candidates,
            "codeflash_version": self.codeflash_version,
            "experiment_metadata": self.experiment_metadata,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "current_username": self.current_username,
            "is_numerical_code": self.is_numerical_code,
        }

        # Add language-specific fields
        if self.language_info.version:
            payload["language_version"] = self.language_info.version

        # Backward compat: always include python_version
        import platform

        payload["python_version"] = platform.python_version()

        # Module system for JS/TS
        if self.language_info.module_system != ModuleSystem.UNKNOWN:
            payload["module_system"] = self.language_info.module_system.value

        # React-specific fields
        if self.is_react_component:
            payload["is_react_component"] = True
            if self.react_context:
                payload["react_context"] = self.react_context

        return payload


@dataclass
class TestGenRequest:
    """Request payload for test generation.

    This schema is designed to be language-agnostic while supporting
    language-specific fields through the `language_info` and `test_info` objects.
    """

    # === Core required fields ===
    source_code: str  # Code being tested
    function_name: str  # Name of function to generate tests for
    trace_id: str  # Unique identifier

    # === Language information ===
    language_info: LanguageInfo

    # === Test information ===
    test_info: TestInfo

    # === Path information ===
    module_path: str = ""  # Path to source module
    test_module_path: str = ""  # Path for generated test

    # === Function metadata ===
    helper_function_names: list[str] = field(default_factory=list)
    is_async: bool = False
    is_numerical_code: bool | None = None

    # === Generation parameters ===
    test_index: int = 0  # Index when generating multiple tests

    # === Metadata ===
    codeflash_version: str = ""

    # === React-specific ===
    is_react_component: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload dict, maintaining backward compatibility."""
        payload = {
            "source_code_being_tested": self.source_code,
            "function_to_optimize": {"function_name": self.function_name, "is_async": self.is_async},
            "helper_function_names": self.helper_function_names,
            "module_path": self.module_path,
            "test_module_path": self.test_module_path,
            "test_framework": self.test_info.framework.value,
            "test_timeout": self.test_info.timeout,
            "trace_id": self.trace_id,
            "test_index": self.test_index,
            "language": self.language_info.name,
            "codeflash_version": self.codeflash_version,
            "is_async": self.is_async,
            "is_numerical_code": self.is_numerical_code,
        }

        # Add language version
        if self.language_info.version:
            payload["language_version"] = self.language_info.version

        # Backward compat: always include python_version
        import platform

        payload["python_version"] = platform.python_version()

        # Module system for JS/TS
        if self.language_info.module_system != ModuleSystem.UNKNOWN:
            payload["module_system"] = self.language_info.module_system.value

        # React-specific fields
        if self.is_react_component:
            payload["is_react_component"] = True

        return payload


# === Helper functions to create language info ===


def python_language_info(version: str | None = None) -> LanguageInfo:
    """Create LanguageInfo for Python."""
    import platform

    return LanguageInfo(
        name="python",
        version=version or platform.python_version(),
        module_system=ModuleSystem.PYTHON,
        file_extension=".py",
        has_type_annotations=True,
        type_checker="mypy",
    )


def javascript_language_info(
    module_system: ModuleSystem = ModuleSystem.COMMONJS, version: str = "ES2022"
) -> LanguageInfo:
    """Create LanguageInfo for JavaScript."""
    ext = ".mjs" if module_system == ModuleSystem.ESM else ".js"
    return LanguageInfo(
        name="javascript", version=version, module_system=module_system, file_extension=ext, has_type_annotations=False
    )


def typescript_language_info(module_system: ModuleSystem = ModuleSystem.ESM, version: str = "ES2022") -> LanguageInfo:
    """Create LanguageInfo for TypeScript."""
    return LanguageInfo(
        name="typescript",
        version=version,
        module_system=module_system,
        file_extension=".ts",
        has_type_annotations=True,
        type_checker="typescript",
    )
