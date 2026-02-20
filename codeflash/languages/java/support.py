"""Main JavaSupport class implementing the LanguageSupport protocol.

This module provides the main JavaSupport class that implements all
required methods for Java language support in codeflash.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import Language, LanguageSupport
from codeflash.languages.java.build_tools import find_test_root
from codeflash.languages.java.comparator import compare_test_results as _compare_test_results
from codeflash.languages.java.concurrency_analyzer import analyze_function_concurrency
from codeflash.languages.java.config import detect_java_project
from codeflash.languages.java.context import extract_code_context, find_helper_functions
from codeflash.languages.java.discovery import discover_functions, discover_functions_from_source
from codeflash.languages.java.formatter import format_java_code, normalize_java_code
from codeflash.languages.java.instrumentation import (
    instrument_existing_test,
    instrument_for_behavior,
    instrument_for_benchmarking,
)
from codeflash.languages.java.parser import get_java_analyzer
from codeflash.languages.java.replacement import add_runtime_comments, remove_test_functions, replace_function
from codeflash.languages.java.test_discovery import discover_tests
from codeflash.languages.java.test_runner import (
    parse_test_results,
    run_behavioral_tests,
    run_benchmarking_tests,
    run_tests,
)
from codeflash.languages.registry import register_language

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import CodeContext, FunctionFilterCriteria, HelperFunction, TestInfo, TestResult
    from codeflash.languages.java.concurrency_analyzer import ConcurrencyInfo

logger = logging.getLogger(__name__)


@register_language
class JavaSupport(LanguageSupport):
    """Java language support implementation.

    Implements the LanguageSupport protocol for Java, providing:
    - Function discovery using tree-sitter
    - Test discovery for JUnit 5
    - Test execution via Maven Surefire
    - Code context extraction
    - Code replacement and formatting
    - Behavior capture instrumentation
    - Benchmarking instrumentation
    """

    def __init__(self) -> None:
        """Initialize Java support."""
        self._analyzer = get_java_analyzer()

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.JAVA

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by Java."""
        return (".java",)

    @property
    def test_framework(self) -> str:
        """Primary test framework name."""
        return "junit5"

    @property
    def comment_prefix(self) -> str:
        """Comment prefix for Java."""
        return "//"

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        """Find all optimizable functions in a Java file."""
        return discover_functions(file_path, filter_criteria, self._analyzer)

    def discover_functions_from_source(
        self, source: str, file_path: Path | None = None, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        """Find all optimizable functions in Java source code."""
        return discover_functions_from_source(source, file_path, filter_criteria, self._analyzer)

    def discover_tests(
        self, test_root: Path, source_functions: Sequence[FunctionToOptimize]
    ) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests."""
        return discover_tests(test_root, source_functions, self._analyzer)

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies."""
        return extract_code_context(function, project_root, module_root, analyzer=self._analyzer)

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function."""
        return find_helper_functions(function, project_root, analyzer=self._analyzer)

    def analyze_concurrency(self, function: FunctionToOptimize, source: str | None = None) -> ConcurrencyInfo:
        """Analyze a function for concurrency patterns.

        Args:
            function: Function to analyze.
            source: Optional source code (will read from file if not provided).

        Returns:
            ConcurrencyInfo with detected concurrent patterns.

        """
        return analyze_function_concurrency(function, source, self._analyzer)

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        """Replace a function in source code with new implementation."""
        return replace_function(source, function, new_source, self._analyzer)

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format Java code."""
        project_root = file_path.parent if file_path else None
        return format_java_code(source, project_root)

    # === Test Execution ===

    def run_tests(
        self, test_files: Sequence[Path], cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[list[TestResult], Path]:
        """Run tests and return results."""
        return run_tests(list(test_files), cwd, env, timeout)

    def parse_test_results(self, junit_xml_path: Path, stdout: str) -> list[TestResult]:
        """Parse test results from JUnit XML."""
        return parse_test_results(junit_xml_path, stdout)

    # === Instrumentation ===

    def instrument_for_behavior(self, source: str, functions: Sequence[FunctionToOptimize]) -> str:
        """Add behavior instrumentation to capture inputs/outputs."""
        return instrument_for_behavior(source, functions, self._analyzer)

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionToOptimize) -> str:
        """Add timing instrumentation to test code."""
        return instrument_for_benchmarking(test_source, target_function, self._analyzer)

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """Check if Java source code is syntactically valid."""
        return self._analyzer.validate_syntax(source)

    def normalize_code(self, source: str) -> str:
        """Normalize code for deduplication."""
        return normalize_java_code(source)

    # === Test Editing ===

    def add_runtime_comments(
        self, test_source: str, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
    ) -> str:
        """Add runtime performance comments to test source code."""
        return add_runtime_comments(test_source, original_runtimes, optimized_runtimes, self._analyzer)

    def remove_test_functions(self, test_source: str, functions_to_remove: list[str]) -> str:
        """Remove specific test functions from test source code."""
        return remove_test_functions(test_source, functions_to_remove, self._analyzer)

    # === Test Result Comparison ===

    def compare_test_results(
        self, original_results_path: Path, candidate_results_path: Path, project_root: Path | None = None
    ) -> tuple[bool, list]:
        """Compare test results between original and candidate code."""
        return _compare_test_results(original_results_path, candidate_results_path, project_root=project_root)

    # === Configuration ===

    def get_test_file_suffix(self) -> str:
        """Get the test file suffix for Java."""
        return "Test.java"

    def get_comment_prefix(self) -> str:
        """Get the comment prefix for Java."""
        return "//"

    def find_test_root(self, project_root: Path) -> Path | None:
        """Find the test root directory for a Java project."""
        return find_test_root(project_root)

    def get_project_root(self, source_file: Path) -> Path | None:
        """Find the project root for a Java file.

        Looks for pom.xml, build.gradle, or build.gradle.kts.

        Args:
            source_file: Path to the source file.

        Returns:
            The project root directory, or None if not found.

        """
        current = source_file.parent
        while current != current.parent:
            if (current / "pom.xml").exists():
                return current
            if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
                return current
            current = current.parent
        return None

    def get_module_path(self, source_file: Path, project_root: Path, tests_root: Path | None = None) -> str:
        """Get the module path for a Java source file.

        For Java, this returns the fully qualified class name (e.g., 'com.example.Algorithms').

        Args:
            source_file: Path to the source file.
            project_root: Root of the project.
            tests_root: Not used for Java.

        Returns:
            Fully qualified class name string.

        """
        # Find the package from the file content
        try:
            content = source_file.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("package "):
                    # Extract package name (remove 'package ' prefix and ';' suffix)
                    package = line[8:].rstrip(";").strip()
                    class_name = source_file.stem
                    return f"{package}.{class_name}"
        except Exception:
            pass

        # Fallback: derive from path relative to src/main/java
        relative = source_file.relative_to(project_root)
        parts = list(relative.parts)

        # Remove src/main/java prefix if present
        if len(parts) > 3 and parts[:3] == ["src", "main", "java"]:
            parts = parts[3:]

        # Remove .java extension and join with dots
        if parts:
            parts[-1] = parts[-1].replace(".java", "")
        return ".".join(parts)

    def get_runtime_files(self) -> list[Path]:
        """Get paths to runtime files needed for Java."""
        # The Java runtime is distributed as a JAR
        return []

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        """Ensure the runtime environment is set up."""
        # Check if codeflash-runtime is available
        config = detect_java_project(project_root)
        if config is None:
            return False

        # For now, assume the runtime is available
        # A full implementation would check/install the JAR
        return True

    def instrument_existing_test(
        self,
        test_string: str,
        call_positions: Sequence[Any],
        function_to_optimize: Any,
        tests_project_root: Path,
        mode: str,
        test_path: Path | None,
    ) -> tuple[bool, str | None]:
        """Inject profiling code into an existing test file."""
        return instrument_existing_test(
            test_string=test_string, function_to_optimize=function_to_optimize, mode=mode, test_path=test_path
        )

    def instrument_source_for_line_profiler(
        self, func_info: FunctionToOptimize, line_profiler_output_file: Path
    ) -> bool:
        """Prepare line profiling via the bytecode-instrumentation agent.

        Generates a config JSON that the Java agent uses at class-load time to
        know which methods to instrument. The agent is loaded via -javaagent
        when the JVM starts. The config includes warmup iterations so the agent
        discards JIT warmup data before measurement.

        Args:
            func_info: Function to profile.
            line_profiler_output_file: Path where profiling results will be written by the agent.

        Returns:
            True if preparation succeeded, False otherwise.

        """
        from codeflash.languages.java.line_profiler import JavaLineProfiler

        try:
            source = func_info.file_path.read_text(encoding="utf-8")

            profiler = JavaLineProfiler(output_file=line_profiler_output_file)

            config_path = line_profiler_output_file.with_suffix(".config.json")
            profiler.generate_agent_config(
                source=source,
                file_path=func_info.file_path,
                functions=[func_info],
                config_output_path=config_path,
            )

            self._line_profiler_agent_arg = profiler.build_javaagent_arg(config_path)
            self._line_profiler_warmup_iterations = profiler.warmup_iterations
            return True
        except Exception:
            logger.exception("Failed to prepare line profiling for %s", func_info.function_name)
            return False

    def parse_line_profile_results(self, line_profiler_output_file: Path) -> dict:
        """Parse line profiler output for Java.

        Args:
            line_profiler_output_file: Path to profiler output file.

        Returns:
            Dict with timing information in standard format.

        """
        from codeflash.languages.java.line_profiler import JavaLineProfiler

        return JavaLineProfiler.parse_results(line_profiler_output_file)

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
        """Run behavioral tests for Java."""
        return run_behavioral_tests(test_paths, test_env, cwd, timeout, project_root, enable_coverage, candidate_index)

    def run_benchmarking_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        min_loops: int = 1,
        max_loops: int = 3,
        target_duration_seconds: float = 10.0,
        inner_iterations: int = 10,
    ) -> tuple[Path, Any]:
        """Run benchmarking tests for Java with inner loop for JIT warmup."""
        return run_benchmarking_tests(
            test_paths,
            test_env,
            cwd,
            timeout,
            project_root,
            min_loops,
            max_loops,
            target_duration_seconds,
            inner_iterations,
        )

    def run_line_profile_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        line_profile_output_file: Path | None = None,
    ) -> tuple[Path, Any]:
        """Run tests with the profiler agent attached.

        Args:
            test_paths: TestFiles object containing test file information.
            test_env: Environment variables for test execution.
            cwd: Working directory for running tests.
            timeout: Optional timeout in seconds.
            project_root: Project root directory.
            line_profile_output_file: Path where profiling results will be written.

        Returns:
            Tuple of (result_file_path, subprocess_result).

        """
        from codeflash.languages.java.test_runner import run_line_profile_tests as _run_line_profile_tests

        return _run_line_profile_tests(
            test_paths=test_paths,
            test_env=test_env,
            cwd=cwd,
            timeout=timeout,
            project_root=project_root,
            line_profile_output_file=line_profile_output_file,
            javaagent_arg=getattr(self, "_line_profiler_agent_arg", None),
        )


# Create a singleton instance for the registry
_java_support: JavaSupport | None = None


def get_java_support() -> JavaSupport:
    """Get the JavaSupport singleton instance.

    Returns:
        The JavaSupport instance.

    """
    global _java_support
    if _java_support is None:
        _java_support = JavaSupport()
    return _java_support
