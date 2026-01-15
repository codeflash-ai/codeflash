"""
Python language support implementation.

This module implements the LanguageSupport protocol for Python, wrapping
the existing Python-specific implementations (LibCST, Jedi, pytest, etc.).
"""

from __future__ import annotations

import ast
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import (
    CodeContext,
    FunctionFilterCriteria,
    FunctionInfo,
    HelperFunction,
    Language,
    LanguageSupport,
    ParentInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.registry import register_language

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@register_language
class PythonSupport:
    """
    Python language support implementation.

    This class wraps the existing Python-specific implementations to conform
    to the LanguageSupport protocol. It delegates to existing code where possible
    to maintain backward compatibility.
    """

    # === Properties ===

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.PYTHON

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by Python."""
        return (".py", ".pyw")

    @property
    def test_framework(self) -> str:
        """Primary test framework for Python."""
        return "pytest"

    # === Discovery ===

    def discover_functions(
        self,
        file_path: Path,
        filter_criteria: FunctionFilterCriteria | None = None,
    ) -> list[FunctionInfo]:
        """
        Find all optimizable functions in a Python file.

        Uses LibCST to parse the file and find functions with return statements.

        Args:
            file_path: Path to the Python file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionInfo objects for discovered functions.
        """
        # Import here to avoid circular imports
        import libcst as cst
        from libcst.metadata import MetadataWrapper

        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            source = file_path.read_text()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        try:
            module = cst.parse_module(source)
            wrapper = MetadataWrapper(module)

            # Use the factory function to get properly-inheriting visitor class
            VisitorClass = _get_visitor_class()
            visitor = VisitorClass(file_path, criteria)
            wrapper.visit(visitor)

            return visitor.functions
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return []

    def discover_tests(
        self,
        test_root: Path,
        source_functions: Sequence[FunctionInfo],
    ) -> dict[str, list[TestInfo]]:
        """
        Map source functions to their tests via static analysis.

        For Python, this uses static analysis to find test files and
        match them to source functions based on imports and function calls.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.
        """
        # For now, return empty dict - the full implementation would
        # use the existing discover_unit_tests module
        # This is a placeholder that maintains the interface
        result: dict[str, list[TestInfo]] = {}

        # Find all test files
        test_files = list(test_root.rglob("test_*.py")) + list(test_root.rglob("*_test.py"))

        for test_file in test_files:
            try:
                source = test_file.read_text()
                tree = ast.parse(source)

                # Find test functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        # For each source function, check if it might be tested here
                        # This is a simplified heuristic - real implementation would
                        # analyze imports and function calls
                        for func in source_functions:
                            if func.name in source or func.qualified_name in source:
                                if func.qualified_name not in result:
                                    result[func.qualified_name] = []
                                result[func.qualified_name].append(
                                    TestInfo(
                                        test_name=node.name,
                                        test_file=test_file,
                                        test_class=None,
                                    )
                                )
            except Exception as e:
                logger.debug(f"Failed to analyze test file {test_file}: {e}")

        return result

    # === Code Analysis ===

    def extract_code_context(
        self,
        function: FunctionInfo,
        project_root: Path,
        module_root: Path,
    ) -> CodeContext:
        """
        Extract function code and its dependencies.

        Uses Jedi for dependency resolution (via existing code_context_extractor).

        Args:
            function: The function to extract context for.
            project_root: Root of the project.
            module_root: Root of the module containing the function.

        Returns:
            CodeContext with target code and dependencies.
        """
        # Read the source file
        try:
            source = function.file_path.read_text()
        except Exception as e:
            logger.error(f"Failed to read {function.file_path}: {e}")
            return CodeContext(
                target_code="",
                target_file=function.file_path,
                language=Language.PYTHON,
            )

        # Extract the function source
        lines = source.splitlines(keepends=True)
        if function.start_line and function.end_line:
            target_lines = lines[function.start_line - 1 : function.end_line]
            target_code = "".join(target_lines)
        else:
            target_code = ""

        # Find helper functions
        helpers = self.find_helper_functions(function, project_root)

        # Build context
        return CodeContext(
            target_code=target_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context="",  # Would be populated by full implementation
            imports=[],
            language=Language.PYTHON,
        )

    def find_helper_functions(
        self,
        function: FunctionInfo,
        project_root: Path,
    ) -> list[HelperFunction]:
        """
        Find helper functions called by the target function.

        Uses Jedi for call resolution.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.
        """
        # This would use the existing Jedi-based implementation
        # For now, return empty list as a placeholder
        return []

    # === Code Transformation ===

    def replace_function(
        self,
        source: str,
        function: FunctionInfo,
        new_source: str,
    ) -> str:
        """
        Replace a function in source code with new implementation.

        Uses LibCST for AST-aware replacement.

        Args:
            source: Original source code.
            function: FunctionInfo identifying the function to replace.
            new_source: New function source code.

        Returns:
            Modified source code with function replaced.
        """
        if function.start_line is None or function.end_line is None:
            logger.error(f"Function {function.name} has no line information")
            return source

        # Use text-based replacement (proven in experiments)
        lines = source.splitlines(keepends=True)

        # Handle case where source doesn't end with newline
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        # Get indentation from original function's first line
        if function.start_line <= len(lines):
            original_first_line = lines[function.start_line - 1]
            original_indent = len(original_first_line) - len(original_first_line.lstrip())
        else:
            original_indent = 0

        # Get indentation from new function's first line
        new_lines = new_source.splitlines(keepends=True)
        if new_lines:
            new_first_line = new_lines[0]
            new_indent = len(new_first_line) - len(new_first_line.lstrip())
        else:
            new_indent = 0

        # Calculate indent adjustment needed
        indent_diff = original_indent - new_indent

        # Adjust indentation of new function if needed
        if indent_diff != 0:
            adjusted_new_lines = []
            for line in new_lines:
                if line.strip():  # Non-empty line
                    if indent_diff > 0:
                        adjusted_new_lines.append(" " * indent_diff + line)
                    else:
                        current_indent = len(line) - len(line.lstrip())
                        remove_amount = min(current_indent, abs(indent_diff))
                        adjusted_new_lines.append(line[remove_amount:])
                else:
                    adjusted_new_lines.append(line)
            new_lines = adjusted_new_lines

        # Ensure new function ends with newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Build result
        before = lines[: function.start_line - 1]
        after = lines[function.end_line :]

        result_lines = before + new_lines + after
        return "".join(result_lines)

    def format_code(
        self,
        source: str,
        file_path: Path | None = None,
    ) -> str:
        """
        Format Python code using black and isort.

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.
        """
        try:
            import black

            formatted = black.format_str(source, mode=black.FileMode())
            return formatted
        except ImportError:
            logger.debug("Black not available, skipping formatting")
            return source
        except Exception as e:
            logger.debug(f"Black formatting failed: {e}")
            return source

    # === Test Execution ===

    def run_tests(
        self,
        test_files: Sequence[Path],
        cwd: Path,
        env: dict[str, str],
        timeout: int,
    ) -> tuple[list[TestResult], Path]:
        """
        Run pytest and return results.

        Args:
            test_files: Paths to test files to run.
            cwd: Working directory for test execution.
            env: Environment variables.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (list of TestResults, path to JUnit XML).
        """
        import sys
        import tempfile

        # Create temp file for JUnit XML output
        junit_xml = cwd / ".codeflash" / "pytest_results.xml"
        junit_xml.parent.mkdir(parents=True, exist_ok=True)

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            f"--junitxml={junit_xml}",
            "-v",
        ] + [str(f) for f in test_files]

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {timeout}s")
            return [], junit_xml
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return [], junit_xml

    def parse_test_results(
        self,
        junit_xml_path: Path,
        stdout: str,
    ) -> list[TestResult]:
        """
        Parse test results from JUnit XML.

        Args:
            junit_xml_path: Path to JUnit XML results file.
            stdout: Standard output from test execution.

        Returns:
            List of TestResult objects.
        """
        import xml.etree.ElementTree as ET

        results = []

        if not junit_xml_path.exists():
            return results

        try:
            tree = ET.parse(junit_xml_path)
            root = tree.getroot()

            for testcase in root.iter("testcase"):
                name = testcase.get("name", "unknown")
                classname = testcase.get("classname", "")
                time_str = testcase.get("time", "0")

                # Convert time to nanoseconds
                try:
                    runtime_ns = int(float(time_str) * 1_000_000_000)
                except ValueError:
                    runtime_ns = None

                # Check for failure/error
                failure = testcase.find("failure")
                error = testcase.find("error")
                passed = failure is None and error is None

                error_message = None
                if failure is not None:
                    error_message = failure.get("message", failure.text)
                elif error is not None:
                    error_message = error.get("message", error.text)

                # Determine test file from classname
                test_file = Path(classname.replace(".", "/") + ".py")

                results.append(
                    TestResult(
                        test_name=name,
                        test_file=test_file,
                        passed=passed,
                        runtime_ns=runtime_ns,
                        error_message=error_message,
                        stdout=stdout,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to parse JUnit XML: {e}")

        return results

    # === Instrumentation ===

    def instrument_for_behavior(
        self,
        source: str,
        functions: Sequence[FunctionInfo],
    ) -> str:
        """
        Add behavior instrumentation to capture inputs/outputs.

        For Python, this adds decorators to wrap function calls.

        Args:
            source: Source code to instrument.
            functions: Functions to add behavior capture.

        Returns:
            Instrumented source code.
        """
        # This would use the existing tracing implementation
        # For now, return source unchanged
        return source

    def instrument_for_benchmarking(
        self,
        test_source: str,
        target_function: FunctionInfo,
    ) -> str:
        """
        Add timing instrumentation to test code.

        Args:
            test_source: Test source code to instrument.
            target_function: Function being benchmarked.

        Returns:
            Instrumented test source code.
        """
        # This would use the existing instrumentation code
        # For now, return source unchanged
        return test_source

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """
        Check if Python source code is syntactically valid.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.
        """
        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    def normalize_code(self, source: str) -> str:
        """
        Normalize Python code for deduplication.

        Removes comments, docstrings, and normalizes whitespace.

        Args:
            source: Source code to normalize.

        Returns:
            Normalized source code.
        """
        try:
            tree = ast.parse(source)
            # Remove docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ):
                        node.body.pop(0)
            return ast.unparse(tree)
        except Exception:
            return source


# Make the visitor inherit from CSTVisitor at runtime to avoid import issues
def _create_visitor_class():
    """Create the visitor class with proper inheritance."""
    import libcst as cst

    class _LibCSTFunctionVisitorImpl(cst.CSTVisitor):
        """LibCST visitor for discovering functions with return statements."""

        METADATA_DEPENDENCIES = (
            cst.metadata.PositionProvider,
            cst.metadata.ParentNodeProvider,
        )

        def __init__(
            self,
            file_path: Path,
            filter_criteria: FunctionFilterCriteria,
        ):
            super().__init__()
            self.file_path = file_path
            self.filter_criteria = filter_criteria
            self.functions: list[FunctionInfo] = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            """Visit a function definition."""
            # Check for return statement
            has_return = _has_return_statement(node)

            if not has_return and self.filter_criteria.require_return:
                return

            # Get position
            try:
                pos = self.get_metadata(cst.metadata.PositionProvider, node)
            except (KeyError, AttributeError):
                return

            # Get parents
            parents: list[ParentInfo] = []
            try:
                parent_node = self.get_metadata(
                    cst.metadata.ParentNodeProvider, node, default=None
                )
                while parent_node is not None:
                    if isinstance(parent_node, (cst.FunctionDef, cst.ClassDef)):
                        parents.append(
                            ParentInfo(
                                name=parent_node.name.value,
                                type=parent_node.__class__.__name__,
                            )
                        )
                    parent_node = self.get_metadata(
                        cst.metadata.ParentNodeProvider, parent_node, default=None
                    )
            except (KeyError, AttributeError):
                pass

            # Check async
            is_async = bool(node.asynchronous)
            if not self.filter_criteria.include_async and is_async:
                return

            # Check if method
            is_method = any(p.type == "ClassDef" for p in parents)
            if not self.filter_criteria.include_methods and is_method:
                return

            self.functions.append(
                FunctionInfo(
                    name=node.name.value,
                    file_path=self.file_path,
                    start_line=pos.start.line,
                    end_line=pos.end.line,
                    parents=tuple(reversed(parents)),
                    is_async=is_async,
                    is_method=is_method,
                    language=Language.PYTHON,
                )
            )

    return _LibCSTFunctionVisitorImpl


# Lazily create the visitor class
_CachedVisitorClass = None


def _get_visitor_class():
    """Get the visitor class, creating it lazily."""
    global _CachedVisitorClass
    if _CachedVisitorClass is None:
        _CachedVisitorClass = _create_visitor_class()
    return _CachedVisitorClass


def _has_return_statement(node: Any) -> bool:
    """Check if a function has a return statement."""
    import libcst as cst
    import libcst.matchers as m

    # Use matcher to find return statements in the function body
    # We need to search the body for any Return nodes
    def search_for_return(n: cst.CSTNode) -> bool:
        """Recursively search for return statements."""
        if isinstance(n, cst.Return):
            return True
        # Check all children
        for child in n.children:
            if search_for_return(child):
                return True
        return False

    # Search in the function body
    if hasattr(node, "body"):
        return search_for_return(node.body)
    return False
