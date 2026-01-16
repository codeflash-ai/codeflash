#!/usr/bin/env python3
from __future__ import annotations

import logging
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
from codeflash.languages.treesitter_utils import get_analyzer_for_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeflash.languages.treesitter_utils import GOFunctionNode, GoImportInfo, TreeSitterAnalyzer

logger = logging.getLogger(__name__)


@register_language
class GolangSupport:
    @property
    def language(self) -> Language:
        return Language.GOLANG

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".go",)

    @property
    def test_framework(self) -> str:
        return "go test"  # built-in go test framework

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionInfo]:
        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            source = file_path.read_text()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")  # noqa: G004
            return []

        try:
            analyzer = get_analyzer_for_file(file_path)
            tree_functions = analyzer.find_functions(source)

            functions: list[FunctionInfo] = []
            for func in tree_functions:
                # Check for return statement if required
                if criteria.require_return and not analyzer.has_return_statement(func, source):
                    continue

                # Check method filter
                if not criteria.include_methods and func.is_method:
                    continue

                # Build parents list for methods (using receiver type)
                parents: list[ParentInfo] = []
                if func.is_method and func.receiver_type:
                    # Extract the base type name (remove * for pointer receivers)
                    receiver_type = func.receiver_type.lstrip("*")
                    parents.append(ParentInfo(name=receiver_type, type="ReceiverType"))

                functions.append(
                    FunctionInfo(
                        name=func.name,
                        file_path=file_path,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        start_col=func.start_col,
                        end_col=func.end_col,
                        parents=tuple(parents),
                        is_method=func.is_method,
                        language=Language.GOLANG,
                    )
                )

            return functions  # noqa: TRY300

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")  # noqa: G004
            return []

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionInfo) -> CodeContext:
        """Extract function code and its dependencies for Go.

        This method extracts:
        - The target function code including leading doc comments
        - All import statements in the file
        - Helper functions called by the target (in the same file)
        - Structs/types referenced by the target
        - Package-level variables/constants used by the target

        Args:
            function: The function to extract context for.

        Returns:
            CodeContext with target code and dependencies.

        """
        try:
            source = function.file_path.read_text()
        except Exception as e:
            logger.error(f"Failed to read {function.file_path}: {e}")  # noqa: G004, TRY400
            return CodeContext(target_code="", target_file=function.file_path, language=Language.GOLANG)

        analyzer = get_analyzer_for_file(function.file_path)

        # Extract the function source with leading comments
        if function.start_line and function.end_line:
            _actual_start, target_code = analyzer.get_node_with_leading_comments(
                source, function.start_line, function.end_line
            )
        else:
            target_code = ""

        # Find imports
        go_imports = analyzer.find_go_imports(source)
        import_lines = self._format_go_imports(go_imports)

        # Find helper functions, structs, and variables
        helpers = self._find_helper_functions(function, source, analyzer)

        # Build read-only context (structs and variables used by target)
        read_only_context = self._build_read_only_context(function, source, analyzer)

        return CodeContext(
            target_code=target_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context=read_only_context,
            imports=import_lines,
            language=Language.GOLANG,
        )

    def _format_go_imports(self, imports: list[GoImportInfo]) -> list[str]:
        """Format Go imports as strings.

        Args:
            imports: List of GoImportInfo objects.

        Returns:
            List of import statement strings.

        """
        if not imports:
            return []

        import_strings: list[str] = []

        for imp in imports:
            if imp.alias:
                import_strings.append(f'import {imp.alias} "{imp.package_path}"')
            else:
                import_strings.append(f'import "{imp.package_path}"')

        return import_strings

    def _find_helper_functions(
        self, function: FunctionInfo, source: str, analyzer: TreeSitterAnalyzer
    ) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Args:
            function: The target function.
            source: The source code.
            analyzer: The TreeSitterAnalyzer instance.

        Returns:
            List of HelperFunction objects.

        """
        helpers: list[HelperFunction] = []

        # Get all functions in the same file
        all_functions = analyzer.find_functions(source)

        # Find the target function's tree-sitter node
        target_func: GOFunctionNode | None = None
        for func in all_functions:
            if func.name == function.name and func.start_line == function.start_line:
                target_func = func
                break

        if not target_func:
            return helpers

        # Find function calls within target
        calls = analyzer.find_go_function_calls(source, target_func)

        # Match calls to functions in the same file
        for func in all_functions:
            if func.name in calls and func.name != function.name:
                # Get the function with its leading comments
                actual_start, source_with_comments = analyzer.get_node_with_leading_comments(
                    source, func.start_line, func.end_line
                )

                helpers.append(
                    HelperFunction(
                        name=func.name,
                        qualified_name=func.name,
                        file_path=function.file_path,
                        source_code=source_with_comments,
                        start_line=actual_start,
                        end_line=func.end_line,
                    )
                )

        return helpers

    def _build_read_only_context(self, function: FunctionInfo, source: str, analyzer: TreeSitterAnalyzer) -> str:
        """Build read-only context including structs and variables.

        This includes:
        - Structs/types referenced by the function
        - Package-level variables/constants that might be used

        Args:
            function: The target function.
            source: The source code.
            analyzer: The TreeSitterAnalyzer instance.

        Returns:
            Read-only context as a string.

        """
        context_parts: list[str] = []

        # Get all functions to find the target
        all_functions = analyzer.find_functions(source)
        target_func: GOFunctionNode | None = None
        for func in all_functions:
            if func.name == function.name and func.start_line == function.start_line:
                target_func = func
                break

        if not target_func:
            return ""

        # Find type references in the function
        type_refs = analyzer.find_go_type_references(source, target_func)

        # Get all structs/types in the file
        structs = analyzer.find_go_structs(source)

        # Include structs that are referenced
        for struct in structs:
            if struct.name in type_refs:
                # Get struct with leading comments
                _actual_start, source_with_comments = analyzer.get_node_with_leading_comments(
                    source, struct.start_line, struct.end_line
                )
                # Wrap in type declaration
                context_parts.append(source_with_comments)

        # Find function calls to identify used variables/constants
        calls = analyzer.find_go_function_calls(source, target_func)

        # Get all package-level variables/constants
        variables = analyzer.find_go_variables(source)

        # Include variables that might be used (simple heuristic: name appears in function)
        func_source = target_func.source_text
        for var in variables:
            if var.name in func_source or var.name in calls:
                _actual_start, source_with_comments = analyzer.get_node_with_leading_comments(
                    source, var.start_line, var.end_line
                )
                prefix = "const" if var.is_const else "var"
                context_parts.append(f"{prefix} {source_with_comments}")

        return "\n\n".join(context_parts)

    def find_helper_functions(self, function: FunctionInfo) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Args:
            function: The target function to analyze.

        Returns:
            List of HelperFunction objects.

        """
        try:
            source = function.file_path.read_text()
            analyzer = get_analyzer_for_file(function.file_path)
            return self._find_helper_functions(function, source, analyzer)
        except Exception as e:
            logger.warning(f"Failed to find helpers for {function.name}: {e}")  # noqa: G004
            return []

    # === Test Discovery ===

    def discover_tests(self, test_root: Path, source_functions: Sequence[FunctionInfo]) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis.

        For Go, this finds *_test.go files and matches test functions
        (starting with Test) to source functions based on naming conventions
        and function calls within the tests.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.

        """
        result: dict[str, list[TestInfo]] = {}

        # Find all test files (Go convention: *_test.go)
        test_files = list(test_root.rglob("*_test.go"))

        for test_file in test_files:
            try:
                source = test_file.read_text()
                analyzer = get_analyzer_for_file(test_file)

                # Find test functions in this file
                test_functions = self._find_go_tests(source, analyzer)

                # Match source functions to tests using tree-sitter to find actual function calls
                for test_func in test_functions:
                    # Get actual function calls from the test function's AST
                    calls_in_test = set(analyzer.find_go_function_calls(source, test_func))

                    for func in source_functions:
                        # Check if the source function is called in this test
                        if func.name in calls_in_test:
                            if func.qualified_name not in result:
                                result[func.qualified_name] = []
                            result[func.qualified_name].append(
                                TestInfo(
                                    test_name=test_func.name,
                                    test_file=test_file,
                                    test_class=None,  # Go doesn't have test classes
                                )
                            )
            except Exception as e:
                logger.debug(f"Failed to analyze test file {test_file}: {e}")  # noqa: G004

        return result

    def _find_go_tests(self, source: str, analyzer: TreeSitterAnalyzer) -> list[GOFunctionNode]:
        """Find Go test functions in source code.

        Go test functions must:
        - Start with "Test"
        - Take *testing.T as the only parameter

        Args:
            source: The source code to analyze.
            analyzer: The TreeSitterAnalyzer instance.

        Returns:
            List of GOFunctionNode objects representing test functions.

        """
        test_functions: list[GOFunctionNode] = []
        all_functions = analyzer.find_functions(source)

        for func in all_functions:
            # Test functions must start with "Test"
            if not func.name.startswith("Test"):
                continue

            # Test functions should not be methods (they're standalone functions)
            if func.is_method:
                continue

            # Check if the function takes *testing.T as parameter
            # We can verify this by checking the function signature in source
            if self._is_test_function(func, source):
                test_functions.append(func)

        return test_functions

    def _is_test_function(self, func: GOFunctionNode, source: str) -> bool:
        """Check if a function is a valid Go test function.

        A valid test function takes *testing.T as its only parameter.

        Args:
            func: The function node to check.
            source: The source code.

        Returns:
            True if the function is a valid test function.

        """
        # Get the function node and check its parameters
        node = func.node
        params_node = node.child_by_field_name("parameters")

        if not params_node:
            return False

        # Count parameter declarations (excluding parentheses)
        param_declarations = [child for child in params_node.children if child.type == "parameter_declaration"]

        # Test functions should have exactly one parameter
        if len(param_declarations) != 1:
            return False

        # Check if the parameter type is *testing.T
        param_node = param_declarations[0]
        type_node = param_node.child_by_field_name("type")

        if type_node:
            source_bytes = source.encode("utf8")
            type_text = source_bytes[type_node.start_byte : type_node.end_byte].decode("utf8")
            # Check for *testing.T or t *testing.T patterns
            if type_text == "*testing.T" or type_text.endswith("testing.T"):
                return True

        return False

    # === Test Execution ===

    def run_tests(
        self, test_files: Sequence[Path], cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[list[TestResult], Path]:
        """Run Go tests and return results.

        Uses 'go test' with gotestsum for JUnit XML output, or falls back
        to go-junit-report if gotestsum is not available.

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
        junit_xml = output_dir / "go-test-results.xml"

        # Get unique package directories from test files
        packages: set[str] = set()
        for test_file in test_files:
            # Get the package path relative to cwd
            try:
                rel_path = test_file.parent.relative_to(cwd)
                packages.add(f"./{rel_path}")
            except ValueError:
                # If not relative, use the directory directly
                packages.add(str(test_file.parent))

        if not packages:
            packages.add("./...")

        test_env = env.copy()

        # Try using gotestsum first (better JUnit output)
        try:
            cmd = [
                "gotestsum",
                "--junitfile",
                str(junit_xml),
                "--format",
                "standard-verbose",
                "--",
                "-v",
                *list(packages),
            ]

            result = subprocess.run(
                cmd, check=False, cwd=cwd, env=test_env, capture_output=True, text=True, timeout=timeout
            )

            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml  # noqa: TRY300

        except FileNotFoundError:
            # gotestsum not available, fall back to go test with go-junit-report
            logger.debug("gotestsum not found, falling back to go test with go-junit-report")

        # Fallback: Use go test with JSON output and convert to JUnit
        try:
            cmd = ["go", "test", "-v", "-json", *list(packages)]

            result = subprocess.run(
                cmd, check=False, cwd=cwd, env=test_env, capture_output=True, text=True, timeout=timeout
            )

            # Try to convert JSON output to JUnit using go-junit-report
            try:
                convert_cmd = ["go-junit-report", "-out", str(junit_xml)]
                subprocess.run(
                    convert_cmd,
                    check=False,
                    input=result.stdout,
                    cwd=cwd,
                    env=test_env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except FileNotFoundError:
                # go-junit-report not available, parse JSON output directly
                logger.debug("go-junit-report not found, parsing JSON output directly")
                results = self._parse_go_test_json(result.stdout)
                return results, junit_xml

            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml  # noqa: TRY300

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {timeout}s")  # noqa: G004
            return [], junit_xml
        except Exception as e:
            logger.error(f"Test execution failed: {e}")  # noqa: G004, TRY400
            return [], junit_xml

    def parse_test_results(self, junit_xml_path: Path, stdout: str) -> list[TestResult]:
        """Parse test results from JUnit XML.

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

                # Check for failure/error/skipped
                failure = testcase.find("failure")
                error = testcase.find("error")
                skipped = testcase.find("skipped")

                passed = failure is None and error is None and skipped is None

                error_message = None
                if failure is not None:
                    error_message = failure.get("message") or failure.text
                elif error is not None:
                    error_message = error.get("message") or error.text

                # Determine test file from classname
                # Go JUnit output typically uses package/filename as classname
                test_file = Path(classname.replace(".", "/") + "_test.go") if classname else Path("unknown")

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
            logger.warning(f"Failed to parse JUnit XML: {e}")  # noqa: G004

        return results

    def _parse_go_test_json(self, json_output: str) -> list[TestResult]:
        """Parse Go test JSON output directly when JUnit converters aren't available.

        Go test -json outputs newline-delimited JSON events.

        Args:
            json_output: The JSON output from go test -json.

        Returns:
            List of TestResult objects.

        """
        import json

        results: list[TestResult] = []
        test_results: dict[str, dict[str, Any]] = {}

        for line in json_output.strip().split("\n"):
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = event.get("Action")
            test_name = event.get("Test")
            package = event.get("Package", "")
            elapsed = event.get("Elapsed", 0)
            output = event.get("Output", "")

            if not test_name:
                continue

            # Initialize test result tracking
            key = f"{package}:{test_name}"
            if key not in test_results:
                test_results[key] = {
                    "name": test_name,
                    "package": package,
                    "passed": None,
                    "elapsed": 0,
                    "output": [],
                    "error_message": None,
                }

            # Track output
            if output:
                test_results[key]["output"].append(output)

            # Track final status
            if action == "pass":
                test_results[key]["passed"] = True
                test_results[key]["elapsed"] = elapsed
            elif action == "fail":
                test_results[key]["passed"] = False
                test_results[key]["elapsed"] = elapsed
                # Collect error message from output
                test_results[key]["error_message"] = "".join(test_results[key]["output"])
            elif action == "skip":
                test_results[key]["passed"] = None  # Skipped tests are neither pass nor fail

        # Convert to TestResult objects
        for data in test_results.values():
            if data["passed"] is None:
                continue  # Skip tests that were skipped

            runtime_ns = int(data["elapsed"] * 1_000_000_000) if data["elapsed"] else None

            results.append(
                TestResult(
                    test_name=data["name"],
                    test_file=Path(data["package"].replace("/", "_") + "_test.go"),
                    passed=data["passed"],
                    runtime_ns=runtime_ns,
                    error_message=data["error_message"],
                    stdout="".join(data["output"]),
                )
            )

        return results


if __name__ == "__main__":
    golang_support = GolangSupport()

    root = Path("/home/mohammed/Documents/golang-projects/etcd")
    functions = golang_support.discover_functions(root / "server/etcdserver/raft.go")
    context = golang_support.extract_code_context(functions[1])
    print(context)
    # dicovered_tests = golang_support.discover_tests(root, functions)
