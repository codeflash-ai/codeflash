"""Python language support implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import libcst as cst

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import (
    CodeContext,
    FunctionFilterCriteria,
    HelperFunction,
    Language,
    ReferenceInfo,
    TestInfo,
    TestResult,
)
from codeflash.languages.registry import register_language
from codeflash.models.function_types import FunctionParent

if TYPE_CHECKING:
    import ast
    from collections.abc import Sequence

    from libcst import CSTNode
    from libcst.metadata import CodeRange

    from codeflash.languages.base import DependencyResolver
    from codeflash.models.models import FunctionSource, GeneratedTestsList, InvocationId, ValidCode
    from codeflash.verification.verification_utils import TestConfig

logger = logging.getLogger(__name__)


def function_sources_to_helpers(sources: list[FunctionSource]) -> list[HelperFunction]:
    return [
        HelperFunction(
            name=fs.only_function_name,
            qualified_name=fs.qualified_name,
            file_path=fs.file_path,
            source_code=fs.source_code,
            start_line=fs.jedi_definition.line if fs.jedi_definition else 1,
            end_line=fs.jedi_definition.line if fs.jedi_definition else 1,
        )
        for fs in sources
    ]


class ReturnStatementVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_return_statement: bool = False

    def visit_Return(self, node: cst.Return) -> None:
        self.has_return_statement = True


class FunctionVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider, cst.metadata.ParentNodeProvider)

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.file_path: Path = file_path
        self.functions: list[FunctionToOptimize] = []

    @staticmethod
    def is_pytest_fixture(node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            dec = decorator.decorator
            if isinstance(dec, cst.Call):
                dec = dec.func
            if isinstance(dec, cst.Attribute) and dec.attr.value == "fixture":
                if isinstance(dec.value, cst.Name) and dec.value.value == "pytest":
                    return True
            if isinstance(dec, cst.Name) and dec.value == "fixture":
                return True
        return False

    @staticmethod
    def is_property(node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            dec = decorator.decorator
            if isinstance(dec, cst.Name) and dec.value in ("property", "cached_property"):
                return True
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        return_visitor: ReturnStatementVisitor = ReturnStatementVisitor()
        node.visit(return_visitor)
        if return_visitor.has_return_statement and not self.is_pytest_fixture(node) and not self.is_property(node):
            pos: CodeRange = self.get_metadata(cst.metadata.PositionProvider, node)
            parents: CSTNode | None = self.get_metadata(cst.metadata.ParentNodeProvider, node)
            ast_parents: list[FunctionParent] = []
            while parents is not None:
                if isinstance(parents, cst.FunctionDef):
                    # Skip nested functions — only discover top-level and class-level functions
                    return
                if isinstance(parents, cst.ClassDef):
                    ast_parents.append(FunctionParent(parents.name.value, parents.__class__.__name__))
                parents = self.get_metadata(cst.metadata.ParentNodeProvider, parents, default=None)
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name.value,
                    file_path=self.file_path,
                    parents=list(reversed(ast_parents)),
                    starting_line=pos.start.line,
                    ending_line=pos.end.line,
                    is_async=bool(node.asynchronous),
                )
            )


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
    def default_file_extension(self) -> str:
        """Default file extension for Python."""
        return ".py"

    @property
    def test_framework(self) -> str:
        """Primary test framework for Python."""
        return "pytest"

    @property
    def comment_prefix(self) -> str:
        return "#"

    @property
    def dir_excludes(self) -> frozenset[str]:
        return frozenset(
            {
                "__pycache__",
                ".venv",
                "venv",
                ".tox",
                ".nox",
                ".eggs",
                ".mypy_cache",
                ".ruff_cache",
                ".pytest_cache",
                ".hypothesis",
                "htmlcov",
                ".pytype",
                ".pyre",
                ".pybuilder",
                ".ipynb_checkpoints",
                ".codeflash",
                ".cache",
                ".complexipy_cache",
                "build",
                "dist",
                "sdist",
                ".coverage*",
                ".pyright*",
                "*.egg-info",
            }
        )

    @property
    def default_language_version(self) -> str | None:
        return None

    @property
    def valid_test_frameworks(self) -> tuple[str, ...]:
        return ("pytest", "unittest")

    @property
    def test_result_serialization_format(self) -> str:
        return "pickle"

    def load_coverage(
        self,
        coverage_database_file: Path,
        function_name: str,
        code_context: Any,
        source_file: Path,
        coverage_config_file: Path | None = None,
    ) -> Any:
        from codeflash.verification.coverage_utils import CoverageUtils

        return CoverageUtils.load_from_sqlite_database(
            database_path=coverage_database_file,
            config_path=coverage_config_file,
            source_code_path=source_file,
            code_context=code_context,
            function_name=function_name,
        )

    def process_generated_test_strings(
        self,
        generated_test_source: str,
        instrumented_behavior_test_source: str,
        instrumented_perf_test_source: str,
        function_to_optimize: Any,
        test_path: Path,
        test_cfg: Any,
        project_module_system: str | None,
    ) -> tuple[str, str, str]:
        from codeflash.code_utils.code_utils import get_run_tmp_file

        temp_run_dir = get_run_tmp_file(Path()).as_posix()
        instrumented_behavior_test_source = instrumented_behavior_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
        instrumented_perf_test_source = instrumented_perf_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
        return generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source

    def adjust_test_config_for_discovery(self, test_cfg: Any) -> None:
        pass

    def detect_module_system(self, project_root: Path, source_file: Path) -> str | None:
        return None

    # === Discovery ===

    def discover_functions(
        self, source: str, file_path: Path, filter_criteria: FunctionFilterCriteria | None = None
    ) -> list[FunctionToOptimize]:
        criteria = filter_criteria or FunctionFilterCriteria()

        tree = cst.parse_module(source)

        wrapper = cst.metadata.MetadataWrapper(tree)
        function_visitor = FunctionVisitor(file_path=file_path)
        wrapper.visit(function_visitor)

        functions: list[FunctionToOptimize] = []
        for func in function_visitor.functions:
            if not criteria.include_async and func.is_async:
                continue

            if not criteria.include_methods and func.parents:
                continue

            if criteria.require_return and func.starting_line is None:
                continue

            func_with_is_method = FunctionToOptimize(
                function_name=func.function_name,
                file_path=file_path,
                parents=func.parents,
                starting_line=func.starting_line,
                ending_line=func.ending_line,
                starting_col=func.starting_col,
                ending_col=func.ending_col,
                is_async=func.is_async,
                is_method=len(func.parents) > 0 and any(p.type == "ClassDef" for p in func.parents),
                language="python",
            )
            functions.append(func_with_is_method)

        return functions

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
                    if func.function_name in source:
                        result[func.qualified_name].append(
                            TestInfo(test_name=test_file.stem, test_file=test_file, test_class=None)
                        )
                except Exception:
                    pass

        return result

    # === Code Analysis ===

    def extract_code_context(self, function: FunctionToOptimize, project_root: Path, module_root: Path) -> CodeContext:
        """Extract function code and its dependencies via the canonical context pipeline."""
        from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context

        try:
            result = get_code_optimization_context(function, project_root)
        except Exception as e:
            logger.warning("Failed to extract code context for %s: %s", function.function_name, e)
            return CodeContext(target_code="", target_file=function.file_path, language=Language.PYTHON)

        helpers = function_sources_to_helpers(result.helper_functions)

        return CodeContext(
            target_code=result.read_writable_code.markdown,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context=result.read_only_context_code,
            imports=[],
            language=Language.PYTHON,
        )

    def find_helper_functions(self, function: FunctionToOptimize, project_root: Path) -> list[HelperFunction]:
        """Find helper functions called by the target function via the canonical jedi pipeline."""
        from codeflash.languages.python.context.code_context_extractor import get_function_sources_from_jedi

        try:
            _dict, sources = get_function_sources_from_jedi(
                {function.file_path: {function.qualified_name}}, project_root
            )
        except Exception as e:
            logger.warning("Failed to find helpers for %s: %s", function.function_name, e)
            return []

        return function_sources_to_helpers(sources)

    def find_references(
        self, function: FunctionToOptimize, project_root: Path, tests_root: Path | None = None, max_files: int = 500
    ) -> list[ReferenceInfo]:
        """Find all references (call sites) to a function across the codebase.

        Uses jedi to find all places where a Python function is called.

        Args:
            function: The function to find references for.
            project_root: Root of the project to search.
            tests_root: Root of tests directory (references in tests are excluded).
            max_files: Maximum number of files to search.

        Returns:
            List of ReferenceInfo objects describing each reference location.

        """
        try:
            import jedi

            source = function.file_path.read_text()

            # Find the function position
            script = jedi.Script(code=source, path=function.file_path)
            names = script.get_names(all_scopes=True, definitions=True)

            function_pos = None
            for name in names:
                if name.type == "function" and name.name == function.function_name:
                    # Check for class parent if it's a method
                    if function.class_name:
                        parent = name.parent()
                        if parent and parent.name == function.class_name and parent.type == "class":
                            function_pos = (name.line, name.column)
                            break
                    else:
                        function_pos = (name.line, name.column)
                        break

            if function_pos is None:
                return []

            # Get references using jedi
            script = jedi.Script(code=source, path=function.file_path, project=jedi.Project(path=project_root))
            references = script.get_references(line=function_pos[0], column=function_pos[1])

            result: list[ReferenceInfo] = []
            seen_locations: set[tuple[Path, int, int]] = set()

            for ref in references:
                if not ref.module_path:
                    continue

                ref_path = Path(ref.module_path)

                # Skip the definition itself
                if ref_path == function.file_path and ref.line == function_pos[0]:
                    continue

                # Skip test files
                if tests_root:
                    try:
                        ref_path.relative_to(tests_root)
                        continue
                    except ValueError:
                        pass

                # Avoid duplicates
                loc_key = (ref_path, ref.line, ref.column)
                if loc_key in seen_locations:
                    continue
                seen_locations.add(loc_key)

                # Get context line
                try:
                    ref_source = ref_path.read_text()
                    lines = ref_source.splitlines()
                    context = lines[ref.line - 1] if ref.line <= len(lines) else ""
                except Exception:
                    context = ""

                # Determine caller function
                caller_function = None
                try:
                    parent = ref.parent()
                    if parent and parent.type == "function":
                        caller_function = parent.name
                except Exception:
                    pass

                result.append(
                    ReferenceInfo(
                        file_path=ref_path,
                        line=ref.line,
                        column=ref.column,
                        end_line=ref.line,
                        end_column=ref.column + len(function.function_name),
                        context=context.strip(),
                        reference_type="call",
                        import_name=function.function_name,
                        caller_function=caller_function,
                    )
                )

            return result

        except Exception as e:
            logger.warning("Failed to find references for %s: %s", function.function_name, e)
            return []

    # === Code Transformation ===

    def replace_function(self, source: str, function: FunctionToOptimize, new_source: str) -> str:
        """Replace a function in source code with new implementation.

        Uses libcst for Python code transformation.

        Args:
            source: Original source code.
            function: FunctionToOptimize identifying the function to replace.
            new_source: New function source code.

        Returns:
            Modified source code with function replaced.

        """
        from codeflash.languages.python.static_analysis.code_replacer import replace_functions_in_file

        try:
            # Determine the function names to replace
            original_function_names = [function.qualified_name]

            # Use the existing replacer
            return replace_functions_in_file(
                source_code=source,
                original_function_names=original_function_names,
                optimized_code=new_source,
                preexisting_objects=set(),
            )
        except Exception as e:
            logger.warning("Failed to replace function %s: %s", function.function_name, e)
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
                ["ruff", "format", "-"], check=False, input=source, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug("Ruff formatting failed: %s", e)

        # Try black as fallback
        try:
            result = subprocess.run(
                ["black", "-q", "-"], check=False, input=source, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug("Black formatting failed: %s", e)

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
        cmd = ["python", "-m", "pytest", f"--junitxml={junit_xml}", "-v"]
        cmd.extend(str(f) for f in test_files)

        try:
            result = subprocess.run(cmd, check=False, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
            results = self.parse_test_results(junit_xml, result.stdout)
            return results, junit_xml

        except subprocess.TimeoutExpired:
            logger.warning("Test execution timed out after %ss", timeout)
            return [], junit_xml
        except Exception as e:
            logger.exception("Test execution failed: %s", e)
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
            logger.warning("Failed to parse JUnit XML: %s", e)

        return results

    # === Instrumentation ===

    def instrument_for_behavior(self, source: str, functions: Sequence[FunctionToOptimize]) -> str:
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

    def instrument_for_benchmarking(self, test_source: str, target_function: FunctionToOptimize) -> str:
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
        from codeflash.languages.python.normalizer import normalize_python_code

        try:
            return normalize_python_code(source, remove_docstrings=True)
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
            def __init__(self, names_to_remove: list[str]) -> None:
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

    def postprocess_generated_tests(
        self, generated_tests: GeneratedTestsList, test_framework: str, project_root: Path, source_file_path: Path
    ) -> GeneratedTestsList:
        """Apply language-specific postprocessing to generated tests."""
        _ = test_framework, project_root, source_file_path
        return generated_tests

    def remove_test_functions_from_generated_tests(
        self, generated_tests: GeneratedTestsList, functions_to_remove: list[str]
    ) -> GeneratedTestsList:
        """Remove specific test functions from generated tests."""
        from codeflash.languages.python.static_analysis.edit_generated_tests import (
            remove_functions_from_generated_tests,
        )

        return remove_functions_from_generated_tests(generated_tests, functions_to_remove)

    def add_runtime_comments_to_generated_tests(
        self,
        generated_tests: GeneratedTestsList,
        original_runtimes: dict[InvocationId, list[int]],
        optimized_runtimes: dict[InvocationId, list[int]],
        tests_project_rootdir: Path | None = None,
    ) -> GeneratedTestsList:
        """Add runtime comments to generated tests."""
        from codeflash.languages.python.static_analysis.edit_generated_tests import (
            add_runtime_comments_to_generated_tests,
        )

        return add_runtime_comments_to_generated_tests(
            generated_tests, original_runtimes, optimized_runtimes, tests_project_rootdir
        )

    def add_global_declarations(self, optimized_code: str, original_source: str, module_abspath: Path) -> str:
        _ = optimized_code, module_abspath
        return original_source

    def extract_calling_function_source(self, source_code: str, function_name: str, ref_line: int) -> str | None:
        """Extract the source code of a calling function in Python."""
        try:
            import ast

            lines = source_code.splitlines()
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                    end_line = node.end_lineno or node.lineno
                    if node.lineno <= ref_line <= end_line:
                        return "\n".join(lines[node.lineno - 1 : end_line])
        except Exception:
            return None
        return None

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

    def find_test_root(self, project_root: Path) -> Path | None:
        """Find the test root directory for a Python project.

        Args:
            project_root: Root directory of the project.

        Returns:
            Path to test root, or None if not found.

        """
        # Common test directory patterns for Python
        test_dirs = [project_root / "tests", project_root / "test", project_root / "spec"]

        for test_dir in test_dirs:
            if test_dir.exists() and test_dir.is_dir():
                return test_dir

        # Check for pytest.ini or pyproject.toml
        if (project_root / "pytest.ini").exists() or (project_root / "pyproject.toml").exists():
            return project_root

        return None

    def get_module_path(self, source_file: Path, project_root: Path, tests_root: Path | None = None) -> str:
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

    def create_dependency_resolver(self, project_root: Path) -> DependencyResolver | None:
        from codeflash.languages.python.reference_graph import ReferenceGraph

        try:
            return ReferenceGraph(project_root, language=self.language.value)
        except Exception:
            logger.debug("Failed to initialize ReferenceGraph, falling back to per-function Jedi analysis")
            return None

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

    def instrument_source_for_line_profiler(
        self, func_info: FunctionToOptimize, line_profiler_output_file: Path
    ) -> bool:
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

    @property
    def function_optimizer_class(self) -> type:
        from codeflash.languages.python.function_optimizer import PythonFunctionOptimizer

        return PythonFunctionOptimizer

    def prepare_module(
        self, module_code: str, module_path: Path, project_root: Path
    ) -> tuple[dict[Path, ValidCode], ast.Module] | None:
        from codeflash.languages.python.optimizer import prepare_python_module

        return prepare_python_module(module_code, module_path, project_root)

    pytest_cmd: str = "pytest"

    def setup_test_config(self, test_cfg: TestConfig, file_path: Path) -> None:
        self.pytest_cmd = test_cfg.pytest_cmd or "pytest"

    def pytest_cmd_tokens(self, is_posix: bool) -> list[str]:
        import shlex

        return shlex.split(self.pytest_cmd, posix=is_posix)

    def build_pytest_cmd(self, safe_sys_executable: str, is_posix: bool) -> list[str]:
        return [safe_sys_executable, "-m", *self.pytest_cmd_tokens(is_posix)]

    # === Test Execution (Full Protocol) ===

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
        import contextlib
        import shlex
        import sys

        from codeflash.code_utils.code_utils import get_run_tmp_file
        from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
        from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
        from codeflash.languages.python.static_analysis.coverage_utils import prepare_coverage_files
        from codeflash.models.models import TestType
        from codeflash.verification.test_runner import execute_test_subprocess

        blocklisted_plugins = ["benchmark", "codspeed", "xdist", "sugar"]

        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type == TestType.REPLAY_TEST:
                if file.tests_in_file:
                    test_files.extend(
                        [
                            str(file.instrumented_behavior_file_path) + "::" + test.test_function
                            for test in file.tests_in_file
                        ]
                    )
            else:
                test_files.append(str(file.instrumented_behavior_file_path))

        pytest_cmd_list = self.build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
        test_files = list(set(test_files))

        common_pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
        ]
        if timeout is not None:
            common_pytest_args.append(f"--timeout={timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

        coverage_database_file: Path | None = None
        coverage_config_file: Path | None = None

        if enable_coverage:
            coverage_database_file, coverage_config_file = prepare_coverage_files()
            pytest_test_env["NUMBA_DISABLE_JIT"] = str(1)
            pytest_test_env["TORCHDYNAMO_DISABLE"] = str(1)
            pytest_test_env["PYTORCH_JIT"] = str(0)
            pytest_test_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
            pytest_test_env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
            pytest_test_env["JAX_DISABLE_JIT"] = str(0)

            is_windows = sys.platform == "win32"
            if is_windows:
                if coverage_database_file.exists():
                    with contextlib.suppress(PermissionError, OSError):
                        coverage_database_file.unlink()
            else:
                cov_erase = execute_test_subprocess(
                    shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env, timeout=30
                )
                logger.debug(cov_erase)
            coverage_cmd = [
                SAFE_SYS_EXECUTABLE,
                "-m",
                "coverage",
                "run",
                f"--rcfile={coverage_config_file.as_posix()}",
                "-m",
            ]
            coverage_cmd.extend(self.pytest_cmd_tokens(IS_POSIX))

            blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins if plugin != "cov"]
            results = execute_test_subprocess(
                coverage_cmd + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                timeout=600,
            )
            logger.debug("Result return code: %s, %s", results.returncode, results.stderr or "")
        else:
            blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]

            results = execute_test_subprocess(
                pytest_cmd_list + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                timeout=600,
            )
            logger.debug("Result return code: %s, %s", results.returncode, results.stderr or "")

        return result_file_path, results, coverage_database_file, coverage_config_file

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

        from codeflash.code_utils.code_utils import get_run_tmp_file
        from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
        from codeflash.verification.test_runner import execute_test_subprocess

        blocklisted_plugins = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]

        pytest_cmd_list = self.build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
        test_files: list[str] = list({str(file.benchmarking_file_path) for file in test_paths.test_files})
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            f"--codeflash_min_loops={min_loops}",
            f"--codeflash_max_loops={max_loops}",
            f"--codeflash_seconds={target_duration_seconds}",
            "--codeflash_stability_check=true",
        ]
        if timeout is not None:
            pytest_args.append(f"--timeout={timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        return result_file_path, results

    def run_line_profile_tests(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None = None,
        project_root: Path | None = None,
        line_profile_output_file: Path | None = None,
    ) -> tuple[Path, Any]:

        from codeflash.code_utils.code_utils import get_run_tmp_file
        from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
        from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
        from codeflash.verification.test_runner import execute_test_subprocess

        blocklisted_plugins = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]

        pytest_cmd_list = self.build_pytest_cmd(SAFE_SYS_EXECUTABLE, IS_POSIX)
        test_files: list[str] = list({str(file.benchmarking_file_path) for file in test_paths.test_files})
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={TOTAL_LOOPING_TIME_EFFECTIVE}",
        ]
        if timeout is not None:
            pytest_args.append(f"--timeout={timeout}")
        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in blocklisted_plugins]
        pytest_test_env["LINE_PROFILE"] = "1"
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        return result_file_path, results

    def generate_concolic_tests(
        self, test_cfg: Any, project_root: Path, function_to_optimize: FunctionToOptimize, function_to_optimize_ast: Any
    ) -> tuple[dict, str]:
        import ast
        import importlib.util
        import subprocess
        import tempfile
        import time

        from codeflash.cli_cmds.console import console
        from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
        from codeflash.code_utils.shell_utils import make_env_with_project_root
        from codeflash.discovery.discover_unit_tests import discover_unit_tests
        from codeflash.languages.python.static_analysis.concolic_utils import (
            clean_concolic_tests,
            is_valid_concolic_test,
        )
        from codeflash.languages.python.static_analysis.static_analysis import has_typed_parameters
        from codeflash.lsp.helpers import is_LSP_enabled
        from codeflash.telemetry.posthog_cf import ph
        from codeflash.verification.verification_utils import TestConfig

        crosshair_available = importlib.util.find_spec("crosshair") is not None

        start_time = time.perf_counter()
        function_to_concolic_tests: dict = {}
        concolic_test_suite_code = ""

        if not crosshair_available:
            logger.debug("Skipping concolic test generation (crosshair-tool is not installed)")
            return function_to_concolic_tests, concolic_test_suite_code

        if is_LSP_enabled():
            logger.debug("Skipping concolic test generation in LSP mode")
            return function_to_concolic_tests, concolic_test_suite_code

        if (
            test_cfg.concolic_test_root_dir
            and isinstance(function_to_optimize_ast, ast.FunctionDef)
            and has_typed_parameters(function_to_optimize_ast, function_to_optimize.parents)
        ):
            logger.info("Generating concolic opcode coverage tests for the original code…")
            console.rule()
            try:
                env = make_env_with_project_root(project_root)
                cover_result = subprocess.run(
                    [
                        SAFE_SYS_EXECUTABLE,
                        "-m",
                        "crosshair",
                        "cover",
                        "--example_output_format=pytest",
                        "--per_condition_timeout=20",
                        ".".join(
                            [
                                function_to_optimize.file_path.relative_to(project_root)
                                .with_suffix("")
                                .as_posix()
                                .replace("/", "."),
                                function_to_optimize.qualified_name,
                            ]
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    check=False,
                    timeout=600,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                logger.debug("CrossHair Cover test generation timed out")
                return function_to_concolic_tests, concolic_test_suite_code

            if cover_result.returncode == 0:
                generated_concolic_test: str = cover_result.stdout
                if not is_valid_concolic_test(generated_concolic_test, project_root=str(project_root)):
                    logger.debug("CrossHair generated invalid test, skipping")
                    console.rule()
                    return function_to_concolic_tests, concolic_test_suite_code
                concolic_test_suite_code = clean_concolic_tests(generated_concolic_test)
                concolic_test_suite_dir = Path(tempfile.mkdtemp(dir=test_cfg.concolic_test_root_dir))
                concolic_test_suite_path = concolic_test_suite_dir / "test_concolic_coverage.py"
                concolic_test_suite_path.write_text(concolic_test_suite_code, encoding="utf8")

                concolic_test_cfg = TestConfig(
                    tests_root=concolic_test_suite_dir,
                    tests_project_rootdir=test_cfg.concolic_test_root_dir,
                    project_root_path=project_root,
                )
                function_to_concolic_tests, num_discovered_concolic_tests, _ = discover_unit_tests(concolic_test_cfg)
                logger.info(
                    "Created %d concolic unit test case%s ",
                    num_discovered_concolic_tests,
                    "s" if num_discovered_concolic_tests != 1 else "",
                )
                console.rule()
                ph("cli-optimize-concolic-tests", {"num_tests": num_discovered_concolic_tests})

            else:
                logger.debug(
                    "Error running CrossHair Cover%s", ": " + cover_result.stderr if cover_result.stderr else "."
                )
                console.rule()
        end_time = time.perf_counter()
        logger.debug("Generated concolic tests in %.2f seconds", end_time - start_time)
        return function_to_concolic_tests, concolic_test_suite_code
