"""Base types and protocol for multi-language support in Codeflash.

This module defines the core abstractions that all language implementations must follow.
The LanguageSupport protocol defines the interface that each language must implement,
while FunctionToOptimize is the canonical representation of functions across all languages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize

from codeflash.languages.language_enum import Language
from codeflash.models.function_types import FunctionParent

# Backward compatibility aliases - ParentInfo is now FunctionParent
ParentInfo = FunctionParent


# Lazy import for FunctionInfo to avoid circular imports
# This allows `from codeflash.languages.base import FunctionInfo` to work at runtime
def __getattr__(name: str) -> Any:
    if name == "FunctionInfo":
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize

        return FunctionToOptimize
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
        file_path = self.test_file.as_posix()
        if self.test_class:
            return f"{file_path}::{self.test_class}::{self.test_name}"
        return f"{file_path}::{self.test_name}"


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


@dataclass
class ReferenceInfo:
    """Information about a reference (call site) to a function.

    This class captures information about where a function is called
    from, including the file, line number, context, and caller function.

    Attributes:
        file_path: Path to the file containing the reference.
        line: Line number (1-indexed).
        column: Column number (0-indexed).
        end_line: End line number (1-indexed).
        end_column: End column number (0-indexed).
        context: The line of code containing the reference.
        reference_type: Type of reference ("call", "callback", "memoized", "import", "reexport").
        import_name: Name used to import the function (may differ from original).
        caller_function: Name of the function containing this reference (or None for module-level).

    """

    file_path: Path
    line: int
    column: int
    end_line: int
    end_column: int
    context: str
    reference_type: str
    import_name: str | None
    caller_function: str | None = None


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
    def default_file_extension(self) -> str:
        """Default file extension for this language."""
        ...

    @property
    def test_framework(self) -> str:
        """Primary test framework name.

        Returns:
            Test framework identifier (e.g., "pytest", "jest").

        """
        ...

    @property
    def comment_prefix(self) -> str:
        """Like # or //."""
        ...

    @property
    def dir_excludes(self) -> frozenset[str]:
        """Directory name patterns to skip during file discovery.

        Supports glob wildcards: "name" for exact, "prefix*" for startswith, "*suffix" for endswith.
        """
        ...

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        """Find all optimizable functions in a file.

        Args:
            file_path: Path to the source file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionToOptimize objects for discovered functions.

        """
        ...

    def discover_tests(
        self, test_root: Path, source_functions: Sequence[FunctionToOptimize]
    ) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.

        """
        ...

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies.

        Args:
            function: The function to extract context for.
            project_root: Root of the project.
            module_root: Root of the module containing the function.

        Returns:
            CodeContext with target code and dependencies.

        """
        ...

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.

        """
        ...

    def find_references(
        self, function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 500
    ) -> list[ReferenceInfo]:
        """Find all references (call sites) to a function across the codebase.

        This method finds all places where a function is called, including:
        - Direct calls
        - Callbacks (passed to other functions)
        - Memoized versions
        - Re-exports

        Args:
            function: The function to find references for.
            project_root: Root of the project to search.
            tests_root: Root of tests directory (references in tests are excluded).
            max_files: Maximum number of files to search.

        Returns:
            List of ReferenceInfo objects describing each reference location.

        """
        ...

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Args:
            source: Original source code.
            function: FunctionToOptimize identifying the function to replace.
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

    def instrument_for_behavior(self, source: str, functions: Sequence[FunctionToOptimize]) -> str:
        """Add behavior instrumentation to capture inputs/outputs.

        Args:
            source: Source code to instrument.
            functions: Functions to add behavior capture.

        Returns:
            Instrumented source code.

        """
        ...

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionToOptimize) -> str:
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

    # === Test Editing ===

    def add_runtime_comments(
        self, test_source: str, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
    ) -> str:
        """Add runtime performance comments to test source code.

        Adds comments showing the original vs optimized runtime for each
        function call (e.g., "// 1.5ms -> 0.3ms (80% faster)").

        Args:
            test_source: Test source code to annotate.
            original_runtimes: Map of invocation IDs to original runtimes (ns).
            optimized_runtimes: Map of invocation IDs to optimized runtimes (ns).

        Returns:
            Test source code with runtime comments added.

        """
        ...

    def remove_test_functions(self, test_source: str, functions_to_remove: list[str]) -> str:
        """Remove specific test functions from test source code.

        Args:
            test_source: Test source code.
            functions_to_remove: List of function names to remove.

        Returns:
            Test source code with specified functions removed.

        """
        ...

    # === Test Result Comparison ===

    def compare_test_results(
        self, original_results_path: Path, candidate_results_path: Path, project_root: Path | None = None
    ) -> tuple[bool, list]:
        """Compare test results between original and candidate code.

        Args:
            original_results_path: Path to original test results (e.g., SQLite DB).
            candidate_results_path: Path to candidate test results.
            project_root: Project root directory (for finding node_modules, etc.).

        Returns:
            Tuple of (are_equivalent, list of TestDiff objects).

        """
        ...

    # === Configuration ===

    def get_test_file_suffix(self) -> str:
        """Get the test file suffix for this language.

        Returns:
            Test file suffix (e.g., ".test.js", "_test.py").

        """
        ...

    def find_test_root(self, project_root: Path) -> Path | None:
        """Find the test root directory for a project.

        Args:
            project_root: Root directory of the project.

        Returns:
            Path to test root, or None if not found.

        """
        ...

    def get_runtime_files(self) -> list[Path]:
        """Get paths to runtime files that need to be copied to user's project.

        Returns:
            List of paths to runtime files (e.g., codeflash-jest-helper.js).

        """
        ...

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        """Ensure the runtime environment is set up for the project.

        This method handles language-specific runtime setup, such as installing
        npm packages for JavaScript or pip packages for Python.

        Args:
            project_root: The project root directory.

        Returns:
            True if runtime environment is ready, False otherwise.

        """
        # Default implementation: just copy runtime files
        return False

    def instrument_existing_test(
        self,
        test_path: Path,
        call_positions: Sequence[Any],
        function_to_optimize: Any,
        tests_project_root: Path,
        mode: str,
    ) -> tuple[bool, str | None]:
        """Inject profiling code into an existing test file.

        Wraps function calls with capture/benchmark instrumentation for
        behavioral verification and performance benchmarking.

        Args:
            test_path: Path to the test file.
            call_positions: List of code positions where the function is called.
            function_to_optimize: The function being optimized.
            tests_project_root: Root directory of tests.
            mode: Testing mode - "behavior" or "performance".

        Returns:
            Tuple of (success, instrumented_code).

        """
        ...

    def instrument_source_for_line_profiler(
        self, func_info: FunctionToOptimize, line_profiler_output_file: Path
    ) -> bool:
        """Instrument source code before line profiling."""
        ...

    def parse_line_profile_results(self, line_profiler_output_file: Path) -> dict:
        """Parse line profiler output."""
        ...

    # === Test Execution ===

    def run_behavioral_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        enable_coverage: bool = False,
        candidate_index: int = 0,
    ) -> tuple[Path, Any, Path | None, Path | None]:
        """Run behavioral tests for this language.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            enable_coverage: Whether to collect coverage information.
            candidate_index: Index of the candidate being tested.

        Returns:
            Tuple of (result_file_path, subprocess_result, coverage_path, config_path).

        """
        ...

    def run_benchmarking_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        min_loops: int = 5,
        max_loops: int = 100_000,
        target_duration_seconds: float = 10.0,
    ) -> tuple[Path, Any]:
        """Run benchmarking tests for this language.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for the test run.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            min_loops: Minimum number of loops for benchmarking.
            max_loops: Maximum number of loops for benchmarking.
            target_duration_seconds: Target duration for benchmarking in seconds.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
        ...


def convert_parents_to_tuple(parents: list | tuple) -> tuple[FunctionParent, ...]:
    """Convert a list of parent objects to a tuple of FunctionParent.

    Args:
        parents: List or tuple of parent objects with name and type attributes.

    Returns:
        Tuple of FunctionParent objects.

    """
    return tuple(FunctionParent(name=p.name, type=p.type) for p in parents)
