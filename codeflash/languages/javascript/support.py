"""
JavaScript language support implementation.

This module implements the LanguageSupport protocol for JavaScript,
using tree-sitter for code analysis and Jest for test execution.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import (
    CodeContext,
    FunctionFilterCriteria,
    FunctionInfo,
    HelperFunction,
    Language,
    ParentInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.registry import register_language
from codeflash.languages.treesitter_utils import (
    FunctionNode,
    TreeSitterAnalyzer,
    TreeSitterLanguage,
    get_analyzer_for_file,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@register_language
class JavaScriptSupport:
    """
    JavaScript language support implementation.

    This class implements the LanguageSupport protocol for JavaScript/JSX files,
    using tree-sitter for code analysis and Jest for test execution.
    """

    # === Properties ===

    @property
    def language(self) -> Language:
        """The language this implementation supports."""
        return Language.JAVASCRIPT

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """File extensions supported by JavaScript."""
        return (".js", ".jsx", ".mjs", ".cjs")

    @property
    def test_framework(self) -> str:
        """Primary test framework for JavaScript."""
        return "jest"

    # === Discovery ===

    def discover_functions(
        self,
        file_path: Path,
        filter_criteria: FunctionFilterCriteria | None = None,
    ) -> list[FunctionInfo]:
        """
        Find all optimizable functions in a JavaScript file.

        Uses tree-sitter to parse the file and find functions.

        Args:
            file_path: Path to the JavaScript file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionInfo objects for discovered functions.
        """
        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            source = file_path.read_text()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        try:
            analyzer = get_analyzer_for_file(file_path)
            tree_functions = analyzer.find_functions(
                source,
                include_methods=criteria.include_methods,
                include_arrow_functions=True,
                require_name=True,
            )

            functions: list[FunctionInfo] = []
            for func in tree_functions:
                # Check for return statement if required
                if criteria.require_return and not analyzer.has_return_statement(func, source):
                    continue

                # Check async filter
                if not criteria.include_async and func.is_async:
                    continue

                # Build parents list
                parents: list[ParentInfo] = []
                if func.class_name:
                    parents.append(ParentInfo(name=func.class_name, type="ClassDef"))
                if func.parent_function:
                    parents.append(ParentInfo(name=func.parent_function, type="FunctionDef"))

                functions.append(
                    FunctionInfo(
                        name=func.name,
                        file_path=file_path,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        start_col=func.start_col,
                        end_col=func.end_col,
                        parents=tuple(parents),
                        is_async=func.is_async,
                        is_method=func.is_method,
                        language=Language.JAVASCRIPT,
                    )
                )

            return functions

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

        For JavaScript, this uses static analysis to find test files
        and match them to source functions based on imports and function calls.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.
        """
        result: dict[str, list[TestInfo]] = {}

        # Find all test files (Jest conventions)
        test_patterns = [
            "*.test.js",
            "*.test.jsx",
            "*.spec.js",
            "*.spec.jsx",
            "__tests__/**/*.js",
            "__tests__/**/*.jsx",
        ]

        test_files: list[Path] = []
        for pattern in test_patterns:
            test_files.extend(test_root.rglob(pattern))

        for test_file in test_files:
            try:
                source = test_file.read_text()
                analyzer = get_analyzer_for_file(test_file)
                imports = analyzer.find_imports(source)

                # Build a set of imported function names
                imported_names: set[str] = set()
                for imp in imports:
                    if imp.default_import:
                        imported_names.add(imp.default_import)
                    for name, alias in imp.named_imports:
                        imported_names.add(alias or name)

                # Find test functions (describe/it/test blocks)
                test_functions = self._find_jest_tests(source, analyzer)

                # Match source functions to tests
                for func in source_functions:
                    if func.name in imported_names or func.name in source:
                        if func.qualified_name not in result:
                            result[func.qualified_name] = []
                        for test_name in test_functions:
                            result[func.qualified_name].append(
                                TestInfo(
                                    test_name=test_name,
                                    test_file=test_file,
                                    test_class=None,
                                )
                            )
            except Exception as e:
                logger.debug(f"Failed to analyze test file {test_file}: {e}")

        return result

    def _find_jest_tests(self, source: str, analyzer: TreeSitterAnalyzer) -> list[str]:
        """Find Jest test function names in source code."""
        test_names: list[str] = []
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)

        self._walk_for_jest_tests(tree.root_node, source_bytes, test_names)
        return test_names

    def _walk_for_jest_tests(
        self, node: Any, source_bytes: bytes, test_names: list[str]
    ) -> None:
        """Walk tree to find Jest test/it/describe calls."""
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                func_name = source_bytes[func_node.start_byte : func_node.end_byte].decode(
                    "utf8"
                )
                if func_name in ("test", "it", "describe"):
                    # Get the first string argument as the test name
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        for child in args_node.children:
                            if child.type == "string":
                                test_name = source_bytes[
                                    child.start_byte : child.end_byte
                                ].decode("utf8")
                                test_names.append(test_name.strip("'\""))
                                break

        for child in node.children:
            self._walk_for_jest_tests(child, source_bytes, test_names)

    # === Code Analysis ===

    def extract_code_context(
        self,
        function: FunctionInfo,
        project_root: Path,
        module_root: Path,
    ) -> CodeContext:
        """
        Extract function code and its dependencies.

        Uses tree-sitter to analyze imports and find helper functions.

        Args:
            function: The function to extract context for.
            project_root: Root of the project.
            module_root: Root of the module containing the function.

        Returns:
            CodeContext with target code and dependencies.
        """
        try:
            source = function.file_path.read_text()
        except Exception as e:
            logger.error(f"Failed to read {function.file_path}: {e}")
            return CodeContext(
                target_code="",
                target_file=function.file_path,
                language=Language.JAVASCRIPT,
            )

        # Extract the function source
        lines = source.splitlines(keepends=True)
        if function.start_line and function.end_line:
            target_lines = lines[function.start_line - 1 : function.end_line]
            target_code = "".join(target_lines)
        else:
            target_code = ""

        # Find imports and helper functions
        analyzer = get_analyzer_for_file(function.file_path)
        imports = analyzer.find_imports(source)

        # Find helper functions called by target
        helpers = self._find_helper_functions(
            function, source, analyzer, imports, module_root
        )

        # Extract import statements as strings
        import_lines = []
        for imp in imports:
            imp_lines = lines[imp.start_line - 1 : imp.end_line]
            import_lines.append("".join(imp_lines).strip())

        return CodeContext(
            target_code=target_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context="",
            imports=import_lines,
            language=Language.JAVASCRIPT,
        )

    def _find_helper_functions(
        self,
        function: FunctionInfo,
        source: str,
        analyzer: TreeSitterAnalyzer,
        imports: list[Any],
        module_root: Path,
    ) -> list[HelperFunction]:
        """Find helper functions called by the target function."""
        helpers: list[HelperFunction] = []

        # Get all functions in the same file
        all_functions = analyzer.find_functions(source, include_methods=True)

        # Find the target function's tree-sitter node
        target_func = None
        for func in all_functions:
            if (
                func.name == function.name
                and func.start_line == function.start_line
            ):
                target_func = func
                break

        if not target_func:
            return helpers

        # Find function calls within target
        calls = analyzer.find_function_calls(source, target_func)

        # Match calls to functions in the same file
        for func in all_functions:
            if func.name in calls and func.name != function.name:
                helpers.append(
                    HelperFunction(
                        name=func.name,
                        qualified_name=func.name,
                        file_path=function.file_path,
                        source_code=func.source_text,
                        start_line=func.start_line,
                        end_line=func.end_line,
                    )
                )

        # TODO: Follow imports to find helpers in other files

        return helpers

    def find_helper_functions(
        self,
        function: FunctionInfo,
        project_root: Path,
    ) -> list[HelperFunction]:
        """
        Find helper functions called by the target function.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.
        """
        try:
            source = function.file_path.read_text()
            analyzer = get_analyzer_for_file(function.file_path)
            imports = analyzer.find_imports(source)
            return self._find_helper_functions(
                function, source, analyzer, imports, project_root
            )
        except Exception as e:
            logger.warning(f"Failed to find helpers for {function.name}: {e}")
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

        Uses text-based replacement with line numbers.

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
        Format JavaScript code using prettier (if available).

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.
        """
        try:
            # Try to use prettier via npx
            result = subprocess.run(
                ["npx", "prettier", "--stdin-filepath", "file.js"],
                input=source,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug(f"Prettier formatting failed: {e}")

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
        Run Jest tests and return results.

        Args:
            test_files: Paths to test files to run.
            cwd: Working directory for test execution.
            env: Environment variables.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (list of TestResults, path to JUnit XML).
        """
        # Create output directory for results
        output_dir = cwd / ".codeflash"
        output_dir.mkdir(parents=True, exist_ok=True)
        junit_xml = output_dir / "jest-results.xml"

        # Build Jest command
        test_pattern = "|".join(str(f) for f in test_files)
        cmd = [
            "npx",
            "jest",
            "--reporters=default",
            "--reporters=jest-junit",
            f"--testPathPattern={test_pattern}",
            "--runInBand",  # Sequential for deterministic timing
            "--forceExit",
        ]

        test_env = env.copy()
        test_env["JEST_JUNIT_OUTPUT_FILE"] = str(junit_xml)

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=test_env,
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
        results: list[TestResult] = []

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
                # Jest typically uses the file path as classname
                test_file = Path(classname) if classname else Path("unknown")

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

    def instrument_for_tracing(
        self,
        source: str,
        functions: Sequence[FunctionInfo],
    ) -> str:
        """
        Add tracing instrumentation to capture inputs/outputs.

        For JavaScript, this wraps functions to capture their arguments
        and return values.

        Args:
            source: Source code to instrument.
            functions: Functions to add tracing to.

        Returns:
            Instrumented source code.
        """
        # For now, return source unchanged
        # Full implementation would add wrapper code
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
        # For now, return source unchanged
        # Full implementation would add timing wrappers
        return test_source

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """
        Check if JavaScript source code is syntactically valid.

        Uses tree-sitter to parse and check for errors.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.
        """
        try:
            analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)
            tree = analyzer.parse(source)
            # Check if tree has errors
            return not tree.root_node.has_error
        except Exception:
            return False

    def normalize_code(self, source: str) -> str:
        """
        Normalize JavaScript code for deduplication.

        Removes comments and normalizes whitespace.

        Args:
            source: Source code to normalize.

        Returns:
            Normalized source code.
        """
        # Simple normalization: remove extra whitespace
        # A full implementation would use tree-sitter to strip comments
        lines = source.splitlines()
        normalized_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                normalized_lines.append(stripped)
        return "\n".join(normalized_lines)
