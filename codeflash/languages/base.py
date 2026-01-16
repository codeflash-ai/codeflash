"""Base types and protocol for multi-language support in Codeflash.

This module defines the core abstractions that all language implementations must follow.
The LanguageSupport protocol defines the interface that each language must implement,
while the dataclasses define language-agnostic representations of code constructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GOLANG = "golang"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ParentInfo:
    """Parent scope information for nested functions/methods.

    Represents the parent class or function that contains a nested function.
    Used to construct the qualified name of a function.

    Attributes:
        name: The name of the parent scope (class name or function name).
        type: The type of parent ("ClassDef", "FunctionDef", "AsyncFunctionDef", etc.).

    """

    name: str
    type: str  # "ClassDef", "FunctionDef", "AsyncFunctionDef", etc.

    def __str__(self) -> str:
        return f"{self.type}:{self.name}"


@dataclass(frozen=True)
class FunctionInfo:
    """Language-agnostic representation of a function to optimize.

    This class captures all the information needed to identify, locate, and
    work with a function across different programming languages.

    Attributes:
        name: The simple function name (e.g., "add").
        file_path: Absolute path to the file containing the function.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed, inclusive).
        parents: List of parent scopes (for nested functions/methods).
        is_async: Whether this is an async function.
        is_method: Whether this is a method (belongs to a class).
        language: The programming language.
        start_col: Starting column (0-indexed), optional for more precise location.
        end_col: Ending column (0-indexed), optional.

    """

    name: str
    file_path: Path
    start_line: int
    end_line: int
    parents: tuple[ParentInfo, ...] = ()
    is_async: bool = False
    is_method: bool = False
    language: Language = Language.PYTHON
    start_col: int | None = None
    end_col: int | None = None

    @property
    def qualified_name(self) -> str:
        """Full qualified name including parent scopes.

        For a method `add` in class `Calculator`, returns "Calculator.add".
        For nested functions, includes all parent scopes.
        """
        if not self.parents:
            return self.name
        parent_path = ".".join(parent.name for parent in self.parents)
        return f"{parent_path}.{self.name}"

    @property
    def class_name(self) -> str | None:
        """Get the immediate parent class name, if any."""
        for parent in reversed(self.parents):
            if parent.type == "ClassDef":
                return parent.name
        return None

    @property
    def top_level_parent_name(self) -> str:
        """Get the top-level parent name, or function name if no parents."""
        return self.parents[0].name if self.parents else self.name

    def __str__(self) -> str:
        return f"FunctionInfo({self.qualified_name} at {self.file_path}:{self.start_line}-{self.end_line})"


@dataclass
class HelperFunction:
    """A helper function that is a dependency of the target function.

    Helper functions are functions called by the target function that are
    within the same module/project (not external libraries).

    Attributes:
        name: The simple function name.
        qualified_name: Full qualified name including parent scopes.
        file_path: Path to the file containing the helper.
        source_code: The source code of the helper function.
        start_line: Starting line number.
        end_line: Ending line number.

    """

    name: str
    qualified_name: str
    file_path: Path
    source_code: str
    start_line: int
    end_line: int


@dataclass
class CodeContext:
    """Code context extracted for optimization.

    Contains the target function code and all relevant dependencies
    needed for the AI to understand and optimize the function.

    Attributes:
        target_code: Source code of the function to optimize.
        target_file: Path to the file containing the target function.
        helper_functions: List of helper functions called by the target.
        read_only_context: Additional context code (read-only dependencies).
        imports: List of import statements needed.
        language: The programming language.

    """

    target_code: str
    target_file: Path
    helper_functions: list[HelperFunction] = field(default_factory=list)
    read_only_context: str = ""
    imports: list[str] = field(default_factory=list)
    language: Language = Language.PYTHON


@dataclass
class TestInfo:
    """Information about a test that exercises a function.

    Attributes:
        test_name: Name of the test function.
        test_file: Path to the test file.
        test_class: Name of the test class, if any.

    """

    test_name: str
    test_file: Path
    test_class: str | None = None

    @property
    def full_test_path(self) -> str:
        """Get full test path in pytest format (file::class::function)."""
        if self.test_class:
            return f"{self.test_file}::{self.test_class}::{self.test_name}"
        return f"{self.test_file}::{self.test_name}"


@dataclass
class TestResult:
    """Language-agnostic test result.

    Captures the outcome of running a single test, including timing
    and behavioral data for equivalence checking.

    Attributes:
        test_name: Name of the test function.
        test_file: Path to the test file.
        passed: Whether the test passed.
        runtime_ns: Execution time in nanoseconds.
        return_value: The return value captured from the test.
        stdout: Standard output captured during test execution.
        stderr: Standard error captured during test execution.
        error_message: Error message if the test failed.

    """

    test_name: str
    test_file: Path
    passed: bool
    runtime_ns: int | None = None
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    error_message: str | None = None


@dataclass
class FunctionFilterCriteria:
    """Criteria for filtering which functions to discover.

    Attributes:
        include_patterns: Glob patterns for functions to include.
        exclude_patterns: Glob patterns for functions to exclude.
        require_return: Only include functions with return statements.
        include_async: Include async functions.
        include_methods: Include class methods.
        min_lines: Minimum number of lines in the function.
        max_lines: Maximum number of lines in the function.

    """

    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    require_return: bool = True
    include_async: bool = True
    include_methods: bool = True
    min_lines: int | None = None
    max_lines: int | None = None


@runtime_checkable
class LanguageSupport(Protocol):
    """Protocol defining what a language implementation must provide.

    All language-specific implementations (Python, JavaScript, etc.) must
    implement this protocol. The protocol defines the interface for:
    - Function discovery
    - Code context extraction
    - Code transformation (replacement)
    - Test execution
    - Test discovery
    - Instrumentation for tracing

    Example:
        class PythonSupport(LanguageSupport):
            @property
            def language(self) -> Language:
                return Language.PYTHON

            def discover_functions(self, file_path: Path, ...) -> list[FunctionInfo]:
                # Python-specific implementation using LibCST
                ...

    """

    # === Properties ===

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        ...

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by this language.

        Returns:
            Tuple of extensions with leading dots (e.g., (".py",) for Python).

        """
        ...

    @property
    def test_framework(self) -> str:
        """Primary test framework name.

        Returns:
            Test framework identifier (e.g., "pytest", "jest").

        """
        ...

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionInfo]:
        """Find all optimizable functions in a file.

        Args:
            file_path: Path to the source file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionInfo objects for discovered functions.

        """
        ...

    def discover_tests(self, test_root: Path, source_functions: Sequence[FunctionInfo]) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.

        """
        ...

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionInfo, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies.

        Args:
            function: The function to extract context for.
            project_root: Root of the project.
            module_root: Root of the module containing the function.

        Returns:
            CodeContext with target code and dependencies.

        """
        ...

    def find_helper_functions(self, function: FunctionInfo, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.

        """
        ...

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionInfo, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Args:
            source: Original source code.
            function: FunctionInfo identifying the function to replace.
            new_source: New function source code.

        Returns:
            Modified source code with function replaced.

        """
        ...

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format code using language-specific formatter.

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.

        """
        ...

    # === Test Execution ===

    def run_tests(
        self, test_files: Sequence[Path], cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[list[TestResult], Path]:
        """Run tests and return results.

        Args:
            test_files: Paths to test files to run.
            cwd: Working directory for test execution.
            env: Environment variables.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (list of TestResults, path to JUnit XML).

        """
        ...

    def parse_test_results(self, junit_xml_path: Path, stdout: str) -> list[TestResult]:
        """Parse test results from JUnit XML and stdout.

        Args:
            junit_xml_path: Path to JUnit XML results file.
            stdout: Standard output from test execution.

        Returns:
            List of TestResult objects.

        """
        ...

    # === Instrumentation ===

    def instrument_for_tracing(self, source: str, functions: Sequence[FunctionInfo]) -> str:
        """Add tracing instrumentation to capture inputs/outputs.

        Args:
            source: Source code to instrument.
            functions: Functions to add tracing to.

        Returns:
            Instrumented source code.

        """
        ...

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionInfo) -> str:
        """Add timing instrumentation to test code.

        Args:
            test_source: Test source code to instrument.
            target_function: Function being benchmarked.

        Returns:
            Instrumented test source code.

        """
        ...

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """Check if source code is syntactically valid.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.

        """
        ...

    def normalize_code(self, source: str) -> str:
        """Normalize code for deduplication.

        Removes comments, normalizes whitespace, etc. to allow
        comparison of semantically equivalent code.

        Args:
            source: Source code to normalize.

        Returns:
            Normalized source code.

        """
        ...


def convert_parents_to_tuple(parents: list | tuple) -> tuple[ParentInfo, ...]:
    """Convert a list of parent objects to a tuple of ParentInfo.

    This helper handles conversion from the existing FunctionParent
    dataclass to the new ParentInfo dataclass.

    Args:
        parents: List or tuple of parent objects with name and type attributes.

    Returns:
        Tuple of ParentInfo objects.

    """
    return tuple(ParentInfo(name=p.name, type=p.type) for p in parents)
