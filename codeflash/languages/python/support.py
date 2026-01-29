"""Python language support implementation."""

from __future__ import annotations

import logging
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

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@register_language
class PythonSupport:
    """Python language support implementation.

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

    @property
    def comment_prefix(self) -> str:
        return "#"

    # === Discovery ===

    def discover_functions(
        self, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionInfo]:
        """Find all optimizable functions in a Python file.

        Uses libcst to parse the file and find functions with return statements.

        Args:
            file_path: Path to the Python file to analyze.
            filter_criteria: Optional criteria to filter functions.

        Returns:
            List of FunctionInfo objects for discovered functions.

        """
        import libcst as cst

        from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize, FunctionVisitor

        criteria = filter_criteria or FunctionFilterCriteria()

        try:
            # Read and parse the file using libcst with metadata
            source = file_path.read_text(encoding="utf-8")
            try:
                tree = cst.parse_module(source)
            except Exception:
                return []

            # Use the libcst-based FunctionVisitor for accurate line numbers
            wrapper = cst.metadata.MetadataWrapper(tree)
            function_visitor = FunctionVisitor(file_path=str(file_path))
            wrapper.visit(function_visitor)

            functions: list[FunctionInfo] = []
            for func in function_visitor.functions:
                if not isinstance(func, FunctionToOptimize):
                    continue

                # Apply filter criteria
                if not criteria.include_async and func.is_async:
                    continue

                if not criteria.include_methods and func.parents:
                    continue

                # Check for return statement requirement (FunctionVisitor already filters this)
                # but we double-check here for consistency
                if criteria.require_return and func.starting_line is None:
                    continue

                # Convert FunctionToOptimize to FunctionInfo
                parents = tuple(ParentInfo(name=p.name, type=p.type) for p in func.parents)

                functions.append(
                    FunctionInfo(
                        name=func.function_name,
                        file_path=file_path,
                        start_line=func.starting_line or 1,
                        end_line=func.ending_line or 1,
                        start_col=func.starting_col,
                        end_col=func.ending_col,
                        parents=parents,
                        is_async=func.is_async,
                        is_method=len(func.parents) > 0,
                        language=Language.PYTHON,
                    )
                )

            return functions

        except Exception as e:
            logger.warning(f"Failed to discover functions in {file_path}: {e}")
            return []

    def discover_tests(self, test_root: Path, source_functions: Sequence[FunctionInfo]) -> dict[str, list[TestInfo]]:
        """Map source functions to their tests via static analysis.

        Args:
            test_root: Root directory containing tests.
            source_functions: Functions to find tests for.

        Returns:
            Dict mapping qualified function names to lists of TestInfo.

        """
        # For Python, the existing test discovery is done through pytest collection
        # This is a simplified implementation that can be enhanced
        result: dict[str, list[TestInfo]] = {}

        # Find test files
        test_files = list(test_root.rglob("test_*.py")) + list(test_root.rglob("*_test.py"))

        for func in source_functions:
            result[func.qualified_name] = []
            for test_file in test_files:
                try:
                    source = test_file.read_text()
                    # Check if function name appears in test file
                    if func.name in source:
                        result[func.qualified_name].append(
                            TestInfo(test_name=test_file.stem, test_file=test_file, test_class=None)
                        )
                except Exception:
                    pass

        return result

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionInfo, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies.

        Uses jedi and libcst for Python code analysis.

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
            return CodeContext(target_code="", target_file=function.file_path, language=Language.PYTHON)

        # Extract the function source
        lines = source.splitlines(keepends=True)
        if function.start_line and function.end_line:
            target_lines = lines[function.start_line - 1 : function.end_line]
            target_code = "".join(target_lines)
        else:
            target_code = ""

        # Find helper functions
        helpers = self.find_helper_functions(function, project_root)

        # Extract imports
        import_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(stripped)
            elif stripped and not stripped.startswith("#"):
                # Stop at first non-import, non-comment line
                break

        return CodeContext(
            target_code=target_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context="",
            imports=import_lines,
            language=Language.PYTHON,
        )

    def find_helper_functions(self, function: FunctionInfo, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function.

        Uses jedi for Python code analysis.

        Args:
            function: The target function to analyze.
            project_root: Root of the project.

        Returns:
            List of HelperFunction objects.

        """
        helpers: list[HelperFunction] = []

        try:
            import jedi

            from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
            from codeflash.optimization.function_context import belongs_to_function_qualified

            script = jedi.Script(path=function.file_path, project=jedi.Project(path=project_root))
            file_refs = script.get_names(all_scopes=True, definitions=False, references=True)

            qualified_name = function.qualified_name

            for ref in file_refs:
                if not ref.full_name or not belongs_to_function_qualified(ref, qualified_name):
                    continue

                try:
                    definitions = ref.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception:
                    continue

                for definition in definitions:
                    definition_path = definition.module_path
                    if definition_path is None:
                        continue

                    # Check if it's a valid helper (in project, not in target function)
                    is_valid = (
                        str(definition_path).startswith(str(project_root))
                        and not path_belongs_to_site_packages(definition_path)
                        and definition.full_name
                        and not belongs_to_function_qualified(definition, qualified_name)
                        and definition.type == "function"
                    )

                    if is_valid:
                        helper_qualified_name = get_qualified_name(definition.module_name, definition.full_name)
                        # Get source code
                        try:
                            helper_source = definition.get_line_code()
                        except Exception:
                            helper_source = ""

                        helpers.append(
                            HelperFunction(
                                name=definition.name,
                                qualified_name=helper_qualified_name,
                                file_path=definition_path,
                                source_code=helper_source,
                                start_line=definition.line or 1,
                                end_line=definition.line or 1,
                            )
                        )

        except Exception as e:
            logger.warning(f"Failed to find helpers for {function.name}: {e}")

        return helpers

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionInfo, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Uses libcst for Python code transformation.

        Args:
            source: Original source code.
            function: FunctionInfo identifying the function to replace.
            new_source: New function source code.

        Returns:
            Modified source code with function replaced.

        """
        from codeflash.code_utils.code_replacer import replace_functions_in_file

        try:
            # Determine the function names to replace
            original_function_names = [function.qualified_name]

            # Use the existing replacer
            result = replace_functions_in_file(
                source_code=source,
                original_function_names=original_function_names,
                optimized_code=new_source,
                preexisting_objects=set(),
            )
            return result
        except Exception as e:
            logger.warning(f"Failed to replace function {function.name}: {e}")
            return source

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format Python code using ruff or black.

        Args:
            source: Source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.

        """
        import subprocess

        # Try ruff first
        try:
            result = subprocess.run(
                ["ruff", "format", "-"],
                check=False,
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
            logger.debug(f"Ruff formatting failed: {e}")

        # Try black as fallback
        try:
            result = subprocess.run(
                ["black", "-q", "-"],
                check=False,
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
            logger.debug(f"Black formatting failed: {e}")

        return source

    # === Test Execution ===

    def run_tests(
        self, test_files: Sequence[Path], cwd: Path, env: dict[str, str], timeout: int
    ) -> tuple[list[TestResult], Path]:
        """Run pytest tests and return results.

        Args:
            test_files: Paths to test files to run.
            cwd: Working directory for test execution.
            env: Environment variables.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (list of TestResults, path to JUnit XML).

        """
        import subprocess

        # Create output directory for results
        output_dir = cwd / ".codeflash"
        output_dir.mkdir(parents=True, exist_ok=True)
        junit_xml = output_dir / "pytest-results.xml"

        # Build pytest command
        cmd = [
            "python",
            "-m",
            "pytest",
            f"--junitxml={junit_xml}",
            "-v",
        ]
        cmd.extend(str(f) for f in test_files)

        try:
            result = subprocess.run(
                cmd, check=False, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
            )
            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml

        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {timeout}s")
            return [], junit_xml
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return [], junit_xml

    def parse_test_results(self, junit_xml_path: Path, stdout: str) -> list[TestResult]:
        """Parse test results from JUnit XML.

        Args:
            junit_xml_path: Path to JUnit XML results file.
            stdout: Standard output from test execution.

        Returns:
            List of TestResult objects.

        """
        import xml.etree.ElementTree as ET

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
                test_file = Path(classname.replace(".", "/") + ".py") if classname else Path("unknown")

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

    def instrument_for_behavior(self, source: str, functions: Sequence[FunctionInfo]) -> str:
        """Add behavior instrumentation to capture inputs/outputs.

        Args:
            source: Source code to instrument.
            functions: Functions to add behavior capture.

        Returns:
            Instrumented source code.

        """
        # Python uses its own instrumentation through pytest plugin
        # This is a pass-through for now
        return source

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionInfo) -> str:
        """Add timing instrumentation to test code.

        Args:
            test_source: Test source code to instrument.
            target_function: Function being benchmarked.

        Returns:
            Instrumented test source code.

        """
        # Python uses pytest-benchmark or custom timing
        return test_source

    # === Validation ===

    def validate_syntax(self, source: str) -> bool:
        """Check if Python source code is syntactically valid.

        Uses Python's compile() to validate syntax.

        Args:
            source: Source code to validate.

        Returns:
            True if valid, False otherwise.

        """
        try:
            compile(source, "<string>", "exec")
            return True
        except SyntaxError:
            return False

    def normalize_code(self, source: str) -> str:
        """Normalize Python code for deduplication.

        Removes comments, normalizes whitespace, and replaces variable names.

        Args:
            source: Source code to normalize.

        Returns:
            Normalized source code.

        """
        from codeflash.code_utils.deduplicate_code import normalize_code

        try:
            return normalize_code(source, remove_docstrings=True, language=Language.PYTHON)
        except Exception:
            return source

    # === Test Editing ===

    def add_runtime_comments(
        self, test_source: str, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
    ) -> str:
        """Add runtime performance comments to Python test source.

        Args:
            test_source: Test source code to annotate.
            original_runtimes: Map of invocation IDs to original runtimes (ns).
            optimized_runtimes: Map of invocation IDs to optimized runtimes (ns).

        Returns:
            Test source code with runtime comments added.

        """
        # For Python, we typically don't modify test source directly
        return test_source

    def remove_test_functions(self, test_source: str, functions_to_remove: list[str]) -> str:
        """Remove specific test functions from Python test source.

        Args:
            test_source: Test source code.
            functions_to_remove: List of function names to remove.

        Returns:
            Test source code with specified functions removed.

        """
        import libcst as cst

        class TestFunctionRemover(cst.CSTTransformer):
            def __init__(self, names_to_remove: list[str]):
                self.names_to_remove = set(names_to_remove)

            def leave_FunctionDef(
                self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
            ) -> cst.FunctionDef | cst.RemovalSentinel:
                if original_node.name.value in self.names_to_remove:
                    return cst.RemovalSentinel.REMOVE
                return updated_node

        try:
            tree = cst.parse_module(test_source)
            modified = tree.visit(TestFunctionRemover(functions_to_remove))
            return modified.code
        except Exception:
            return test_source

    # === Test Result Comparison ===

    def compare_test_results(
        self, original_results_path: Path, candidate_results_path: Path, project_root: Path | None = None
    ) -> tuple[bool, list]:
        """Compare test results between original and candidate code.

        Args:
            original_results_path: Path to original test results.
            candidate_results_path: Path to candidate test results.
            project_root: Project root directory.

        Returns:
            Tuple of (are_equivalent, list of TestDiff objects).

        """
        # For Python, comparison is done through the verification module
        # This is a simplified implementation
        return True, []

    # === Configuration ===

    def get_test_file_suffix(self) -> str:
        """Get the test file suffix for Python.

        Returns:
            Python test file suffix (.py for display, matching test_xxx.py convention).

        """
        return ".py"

    def get_comment_prefix(self) -> str:
        """Get the comment prefix for Python.

        Returns:
            Python single-line comment prefix.

        """
        return "#"

    def find_test_root(self, project_root: Path) -> Path | None:
        """Find the test root directory for a Python project.

        Args:
            project_root: Root directory of the project.

        Returns:
            Path to test root, or None if not found.

        """
        # Common test directory patterns for Python
        test_dirs = [
            project_root / "tests",
            project_root / "test",
            project_root / "spec",
        ]

        for test_dir in test_dirs:
            if test_dir.exists() and test_dir.is_dir():
                return test_dir

        # Check for pytest.ini or pyproject.toml
        if (project_root / "pytest.ini").exists() or (project_root / "pyproject.toml").exists():
            return project_root

        return None

    def get_module_path(
        self, source_file: Path, project_root: Path, tests_root: Path | None = None
    ) -> str:
        """Get the module path for importing a Python source file.

        For Python, this returns a dot-separated module path (e.g., 'mypackage.mymodule').

        Args:
            source_file: Path to the source file.
            project_root: Root of the project.
            tests_root: Not used for Python (imports use module paths, not relative paths).

        Returns:
            Dot-separated module path string.

        """
        from codeflash.code_utils.code_utils import module_name_from_file_path

        return module_name_from_file_path(source_file, project_root)

    def get_runtime_files(self) -> list[Path]:
        """Get paths to runtime files for Python.

        Returns:
            Empty list - Python doesn't need separate runtime files.

        """
        return []

    def ensure_runtime_environment(self, project_root: Path) -> bool:
        """Ensure Python runtime environment is set up.

        For Python, this is typically a no-op as pytest handles most things.

        Args:
            project_root: The project root directory.

        Returns:
            True - Python runtime is always available.

        """
        return True

    def instrument_existing_test(
        self,
        test_path: Path,
        call_positions: Sequence[Any],
        function_to_optimize: Any,
        tests_project_root: Path,
        mode: str,
    ) -> tuple[bool, str | None]:
        """Inject profiling code into an existing Python test file.

        Args:
            test_path: Path to the test file.
            call_positions: List of code positions where the function is called.
            function_to_optimize: The function being optimized.
            tests_project_root: Root directory of tests.
            mode: Testing mode - "behavior" or "performance".

        Returns:
            Tuple of (success, instrumented_code).

        """
        from codeflash.code_utils.instrument_existing_tests import inject_profiling_into_existing_test
        from codeflash.models.models import TestingMode

        testing_mode = TestingMode.BEHAVIOR if mode == "behavior" else TestingMode.PERFORMANCE

        return inject_profiling_into_existing_test(
            test_path=test_path,
            call_positions=list(call_positions),
            function_to_optimize=function_to_optimize,
            tests_project_root=tests_project_root,
            mode=testing_mode,
        )

    def instrument_source_for_line_profiler(self, func_info: FunctionInfo, line_profiler_output_file: Path) -> bool:
        """Instrument source code for line profiling.

        Args:
            func_info: Information about the function to profile.
            line_profiler_output_file: Output file for profiling results.

        Returns:
            True if instrumentation succeeded, False otherwise.

        """
        # Python line profiling uses the line_profiler package
        # This is handled through the existing infrastructure
        return True

    def parse_line_profile_results(self, line_profiler_output_file: Path) -> dict:
        """Parse line profiler output for Python.

        Args:
            line_profiler_output_file: Path to profiler output file.

        Returns:
            Dict with timing information.

        """
        # Python uses line_profiler which has its own output format
        return {"timings": {}, "unit": 0, "str_out": ""}

    # === Test Execution (Full Protocol) ===
    # Note: For Python, test execution is handled by the main test_runner.py
    # which has special Python-specific logic. These methods are not called
    # for Python as the test_runner checks is_python() and uses the existing path.
    # They are defined here only for protocol compliance.