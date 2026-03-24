"""PythonPlugin — adapter wiring codeflash to the codeflash_core LanguagePlugin protocol."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.plugin_ai_ops import PluginAiOpsMixin
from codeflash.plugin_helpers import (
    format_code_with_ruff_or_black,
    make_test_env,
    read_return_values,
    replace_function_simple,
)
from codeflash.plugin_results import PluginResultsMixin
from codeflash.plugin_test_lifecycle import PluginTestLifecycleMixin
from codeflash.verification.test_runner import run_tests
from codeflash_core.models import BenchmarkResults, CodeContext, TestOutcome, TestOutcomeStatus, TestResults

if TYPE_CHECKING:
    import threading

    from codeflash.api.aiservice import AiServiceClient
    from codeflash.models.models import CodeOptimizationContext
    from codeflash_core.config import TestConfig
    from codeflash_core.models import CoverageData, FunctionToOptimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class PythonPlugin(PluginAiOpsMixin, PluginTestLifecycleMixin, PluginResultsMixin):
    """Implements the codeflash_core LanguagePlugin protocol for Python.

    Converts between core types and internal types at the boundary.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.last_internal_context: CodeOptimizationContext | None = None  # cache for get_candidates
        self.current_function: FunctionToOptimize | None = None  # cache for coverage
        self.tests_project_rootdir: Path | None = None  # cached from test_config
        self.is_numerical_code: bool | None = None  # cached from generate_tests
        self.ai_client: AiServiceClient | None = None
        self.pending_code_markdown: str = ""  # set by optimizer before replace_function
        self.cancel_event: threading.Event | None = None  # set by optimizer for cooperative cancellation
        self.dependency_counts: dict[str, int] = {}

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def get_ai_client(self) -> AiServiceClient:
        if self.ai_client is not None:
            return self.ai_client
        from codeflash.api.aiservice import AiServiceClient

        client = AiServiceClient()
        self.ai_client = client
        return client

    # -- cleanup, comparison, environment validation --------------------------

    def cleanup_run(self, tests_root: Path) -> None:
        import contextlib
        import shutil

        from codeflash.code_utils.code_utils import get_run_tmp_file
        from codeflash.optimization.optimizer import Optimizer as PyOptimizer

        # Remove leftover instrumented test files
        if tests_root.exists():
            leftover = PyOptimizer.find_leftover_instrumented_test_files(tests_root)
            for p in leftover:
                with contextlib.suppress(OSError):
                    p.unlink(missing_ok=True)

        # Remove leftover return-value files (indices 0-30 match max_total in evaluate_candidates)
        for i in range(31):
            with contextlib.suppress(OSError):
                get_run_tmp_file(Path(f"test_return_values_{i}.bin")).unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                get_run_tmp_file(Path(f"test_return_values_{i}.sqlite")).unlink(missing_ok=True)

        # Remove the shared temp directory
        if hasattr(get_run_tmp_file, "tmpdir_path"):
            shutil.rmtree(get_run_tmp_file.tmpdir_path, ignore_errors=True)
            del get_run_tmp_file.tmpdir_path

    def compare_outputs(self, baseline_output: object, candidate_output: object) -> bool:
        from codeflash.verification.comparator import comparator

        return comparator(baseline_output, candidate_output)

    def validate_environment(self, config: object) -> bool:
        from codeflash.code_utils.env_utils import check_formatter_installed

        if hasattr(config, "formatter_cmds") and config.formatter_cmds:
            return check_formatter_installed(config.formatter_cmds)
        return True

    # -- discover_functions --------------------------------------------------

    def discover_functions(self, paths: list[Path]) -> list[FunctionToOptimize]:
        from codeflash.languages.python.support import PythonSupport

        support = PythonSupport()
        results: list[FunctionToOptimize] = []
        for path in paths:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue

            try:
                internal_fns = support.discover_functions(source, path)
            except Exception as exc:
                logger.warning("Skipping %s: failed to parse (%s)", path, exc)
                continue
            for fn in internal_fns:
                # Attach source code so the core optimizer has it
                lines = source.splitlines()
                if fn.starting_line and fn.ending_line:
                    fn.source_code = "\n".join(lines[fn.starting_line - 1 : fn.ending_line])
                results.append(fn)
        return results

    # -- build_index / rank_functions -----------------------------------------

    def build_index(self, files: list[Path], on_progress: object = None) -> None:
        # CallGraphIndex not available in main repo — no-op for now
        pass

    def rank_functions(
        self,
        functions: list[FunctionToOptimize],
        trace_file: Path | None = None,
        test_counts: dict[tuple[Path, str], int] | None = None,
    ) -> list[FunctionToOptimize]:
        if not functions:
            return functions

        # Primary: rank by trace-based addressable time (filters low-importance functions)
        if trace_file and trace_file.exists():
            try:
                from codeflash.benchmarking.function_ranker import FunctionRanker

                ranker = FunctionRanker(trace_file)
                ranked = ranker.rank_functions(functions)
                if test_counts:
                    ranked.sort(
                        key=lambda f: (
                            -ranker.get_function_addressable_time(f),
                            -test_counts.get((f.file_path, f.qualified_name), 0),
                        )
                    )
                logger.debug(
                    "Ranked %d functions by addressable time (filtered %d low-importance)",
                    len(ranked),
                    len(functions) - len(ranked),
                )
                return ranked
            except Exception:
                logger.warning("Trace-based ranking failed, falling back to original order")

        # Fallback: return as-is (no CallGraphIndex available)
        return functions

    def get_dependency_counts(self) -> dict[str, int]:
        return self.dependency_counts

    # -- extract_context -----------------------------------------------------

    def extract_context(self, function: FunctionToOptimize) -> CodeContext:
        from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context
        from codeflash.languages.python.support import function_sources_to_helpers

        internal_fn = function
        ctx = get_code_optimization_context(internal_fn, self.project_root, call_graph=None)
        self.last_internal_context = ctx
        self.current_function = function

        helpers = function_sources_to_helpers(ctx.helper_functions)

        return CodeContext(
            target_function=function,
            target_code=ctx.read_writable_code.flat if ctx.read_writable_code else function.source_code,
            target_file=function.file_path,
            helper_functions=helpers,
            read_only_context=ctx.read_only_context_code,
        )

    # -- run_tests -----------------------------------------------------------

    def run_tests(
        self,
        test_config: TestConfig,
        test_files: list[Path] | None = None,
        test_iteration: int = 0,
        enable_coverage: bool = False,
    ) -> TestResults | tuple[TestResults, CoverageData | None]:
        if test_files is not None:
            files_to_run = test_files
        else:
            files_to_run = sorted(test_config.tests_root.rglob("test_*.py"))
            if not files_to_run:
                files_to_run = sorted(test_config.tests_root.rglob("*_test.py"))

        if not files_to_run:
            return TestResults(passed=True)

        # Clean up stale return-value files before this iteration (matches original)
        from codeflash.code_utils.code_utils import get_run_tmp_file

        for ext in (".bin", ".sqlite"):
            get_run_tmp_file(Path(f"test_return_values_{test_iteration}{ext}")).unlink(missing_ok=True)

        env = make_test_env(test_config.project_root, test_iteration=test_iteration)
        timeout = int(test_config.timeout)

        results, _, cov_db, cov_config = run_tests(
            test_files=files_to_run,
            cwd=test_config.project_root,
            env=env,
            timeout=timeout,
            enable_coverage=enable_coverage,
        )

        # Read return values from SQLite written by instrumented tests
        return_values = read_return_values(test_iteration)

        outcomes = []
        for r in results:
            # Match JUnit test name to SQLite test_function_name
            # The pytest plugin strips parametrize brackets from CODEFLASH_TEST_FUNCTION
            base_name = r.test_name.split("[", 1)[0] if "[" in r.test_name else r.test_name
            ret_vals = return_values.get(base_name)
            output = tuple(ret_vals) if ret_vals else None

            outcomes.append(
                TestOutcome(
                    test_id=r.test_name,
                    status=TestOutcomeStatus.PASSED if r.passed else TestOutcomeStatus.FAILED,
                    duration=r.runtime_ns / 1e9 if r.runtime_ns else 0.0,
                    error_message=r.error_message or "",
                    output=output,
                )
            )

        test_results = TestResults(passed=all(r.passed for r in results), outcomes=outcomes, error=None)

        if enable_coverage:
            coverage_data = self.load_coverage(cov_db, cov_config)
            return test_results, coverage_data

        return test_results

    def load_coverage(self, cov_db: Path | None, cov_config: Path | None) -> CoverageData | None:
        """Load coverage data from SQLite database and convert to core CoverageData."""
        if cov_db is None or cov_config is None:
            return None

        function = self.current_function
        code_context = self.last_internal_context
        if function is None or code_context is None:
            return None

        try:
            from codeflash.verification.coverage_utils import CoverageUtils
            from codeflash_core.models import CoverageData as CoreCoverageData
            from codeflash_core.models import FunctionCoverage as CoreFunctionCoverage

            internal_cov = CoverageUtils.load_from_sqlite_database(
                database_path=cov_db,
                config_path=cov_config,
                function_name=function.qualified_name,
                code_context=code_context,
                source_code_path=function.file_path,
            )

            main_fc = internal_cov.main_func_coverage
            core_main = CoreFunctionCoverage(
                name=main_fc.name,
                coverage=main_fc.coverage,
                executed_lines=list(main_fc.executed_lines),
                unexecuted_lines=list(main_fc.unexecuted_lines),
                executed_branches=list(main_fc.executed_branches),
                unexecuted_branches=list(main_fc.unexecuted_branches),
            )

            core_dep = None
            if internal_cov.dependent_func_coverage:
                dep = internal_cov.dependent_func_coverage
                core_dep = CoreFunctionCoverage(
                    name=dep.name,
                    coverage=dep.coverage,
                    executed_lines=list(dep.executed_lines),
                    unexecuted_lines=list(dep.unexecuted_lines),
                    executed_branches=list(dep.executed_branches),
                    unexecuted_branches=list(dep.unexecuted_branches),
                )

            from codeflash.code_utils.config_consts import COVERAGE_THRESHOLD

            return CoreCoverageData(
                file_path=function.file_path,
                coverage=internal_cov.coverage,
                function_name=function.qualified_name,
                main_func_coverage=core_main,
                dependent_func_coverage=core_dep,
                threshold_percentage=COVERAGE_THRESHOLD,
            )
        except Exception:
            logger.debug("Failed to load coverage data", exc_info=True)
            return None

    # -- replace_function ----------------------------------------------------

    def replace_function(self, file: Path, function: FunctionToOptimize, new_code: str) -> None:
        internal_ctx = self.last_internal_context
        code_markdown = self.pending_code_markdown

        if internal_ctx is not None and code_markdown:
            try:
                self.replace_function_full(function, internal_ctx, code_markdown)
                return
            except Exception:
                logger.debug("Full replace_function failed, falling back to simple replacement", exc_info=True)

        # Fallback: simple single-file replacement
        source = file.read_text(encoding="utf-8")
        internal_fn = function
        modified = replace_function_simple(source, internal_fn, new_code)
        file.write_text(modified, encoding="utf-8")

    def replace_function_full(
        self, function: FunctionToOptimize, internal_ctx: CodeOptimizationContext, code_markdown: str
    ) -> None:
        """Port of FunctionOptimizer.replace_function_and_helpers_with_optimized_code."""
        from collections import defaultdict

        from codeflash.languages.python.context.unused_definition_remover import (
            detect_unused_helper_functions,
            revert_unused_helper_functions,
        )
        from codeflash.languages.python.static_analysis.code_replacer import replace_function_definitions_in_module
        from codeflash.models.models import CodeStringsMarkdown

        optimized_code = CodeStringsMarkdown.parse_markdown_code(code_markdown)

        internal_fn = function

        # Group functions by file (target + helpers where definition_type in ("function", None))
        functions_by_file: dict[Path, set[str]] = defaultdict(set)
        functions_by_file[function.file_path].add(internal_fn.qualified_name)
        for helper in internal_ctx.helper_functions:
            if helper.definition_type in ("function", None):
                functions_by_file[helper.file_path].add(helper.qualified_name)

        # Capture original helper code for unused-helper revert
        original_helper_code: dict[Path, str] = {}
        for hp in functions_by_file:
            if hp != function.file_path and hp.exists():
                original_helper_code[hp] = hp.read_text("utf-8")

        # Replace in each file
        for module_abspath, qualified_names in functions_by_file.items():
            replace_function_definitions_in_module(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                module_abspath=module_abspath,
                preexisting_objects=internal_ctx.preexisting_objects,
                project_root_path=self.project_root,
            )

        # Detect and revert unused helpers
        unused_helpers = detect_unused_helper_functions(internal_fn, internal_ctx, optimized_code)
        if unused_helpers:
            revert_unused_helper_functions(self.project_root, unused_helpers, original_helper_code)

    # -- restore_function ----------------------------------------------------

    def restore_function(self, file: Path, function: FunctionToOptimize, original_code: str) -> None:
        self.replace_function(file, function, original_code)

    # -- run_benchmarks ------------------------------------------------------

    def run_benchmarks(
        self,
        function: FunctionToOptimize,
        test_config: TestConfig,
        test_files: list[Path] | None = None,
        test_iteration: int = 0,
    ) -> BenchmarkResults:
        if test_files is not None:
            files_to_run = test_files
        else:
            files_to_run = sorted(test_config.tests_root.rglob("test_*.py"))
            if not files_to_run:
                files_to_run = sorted(test_config.tests_root.rglob("*_test.py"))

        if not files_to_run:
            return BenchmarkResults()

        env = make_test_env(test_config.project_root, test_iteration=test_iteration)
        timeout = int(test_config.timeout)

        results, *_ = run_tests(
            test_files=files_to_run,
            cwd=test_config.project_root,
            env=env,
            timeout=timeout,
            min_loops=5,
            max_loops=100_000,
            target_seconds=10.0,
            stability_check=True,
        )

        timings: dict[str, float] = {}
        total = 0.0
        for r in results:
            if r.runtime_ns:
                secs = r.runtime_ns / 1e9
                timings[r.test_name] = secs
                total += secs

        return BenchmarkResults(timings=timings, total_time=total)

    # -- format_code ---------------------------------------------------------

    def format_code(self, code: str, file: Path) -> str:
        return format_code_with_ruff_or_black(code, file)

    def validate_candidate(self, code: str) -> bool:
        import ast

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def normalize_code(self, code: str) -> str:
        from codeflash.languages.python.normalizer import normalize_python_code

        try:
            return normalize_python_code(code, remove_docstrings=True)
        except Exception:
            return code

    # ========================================================================
    # Phase 2: Split behavioral / performance test running
    # ========================================================================

    def run_behavioral_tests(self, test_files: list[Path], test_config: TestConfig) -> TestResults:
        result = self.run_tests(test_config, test_files=test_files)
        if isinstance(result, tuple):
            return result[0]
        return result

    def run_performance_tests(
        self, test_files: list[Path], function: FunctionToOptimize, test_config: TestConfig
    ) -> BenchmarkResults:
        return self.run_benchmarks(function, test_config, test_files=test_files)

    # ========================================================================
    # Phase 3: Line profiler (stays here — uses run_tests directly)
    # ========================================================================

    def run_line_profiler(
        self, function: FunctionToOptimize, test_config: TestConfig, test_files: list[Path] | None = None
    ) -> str:
        """Run line profiler on the target function and return formatted output.

        Returns empty string if profiling fails or is not applicable.
        """
        from codeflash.languages.python.parse_line_profile_test_output import parse_line_profile_results
        from codeflash.languages.python.static_analysis.line_profile_utils import (
            add_decorator_imports,
            contains_jit_decorator,
        )

        internal_fn = function
        code_context = self.last_internal_context
        if code_context is None:
            logger.warning("No code context available for line profiler")
            return ""

        # Read original source of function file + helper files for restore
        original_sources: dict[Path, str] = {}
        try:
            original_sources[function.file_path] = function.file_path.read_text("utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("Cannot read function file %s for line profiler", function.file_path)
            return ""

        # Check JIT decorators in function file
        if contains_jit_decorator(original_sources[function.file_path]):
            logger.info("Skipping line profiler for %s - code contains JIT decorator", function.function_name)
            return ""

        # Save and check helper file sources
        for helper in code_context.helper_functions:
            hp = helper.file_path
            if hp not in original_sources:
                try:
                    content = hp.read_text("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                original_sources[hp] = content
                if contains_jit_decorator(content):
                    logger.info(
                        "Skipping line profiler for %s - helper code contains JIT decorator", function.function_name
                    )
                    return ""

        # Determine test files
        if test_files is not None:
            files_to_run = test_files
        else:
            files_to_run = sorted(test_config.tests_root.rglob("test_*.py"))
            if not files_to_run:
                files_to_run = sorted(test_config.tests_root.rglob("*_test.py"))
        if not files_to_run:
            return ""

        try:
            # Inject line profiler decorators and imports into function + helper files
            lprof_output_file = add_decorator_imports(internal_fn, code_context)

            # Run tests with LINE_PROFILE=1 env var
            env = make_test_env(test_config.project_root, test_iteration=0)
            env["LINE_PROFILE"] = "1"

            run_tests(test_files=files_to_run, cwd=test_config.project_root, env=env, timeout=int(test_config.timeout))

            # Parse line profiler results from .lprof file
            results, _ = parse_line_profile_results(lprof_output_file)
            return str(results.get("str_out", ""))
        except Exception:
            logger.debug("Line profiler failed for %s", function.function_name, exc_info=True)
            return ""
        finally:
            # Restore original source files
            for file_path, content in original_sources.items():
                try:
                    file_path.write_text(content, "utf-8")
                except OSError:
                    logger.warning("Failed to restore %s after line profiler", file_path)
