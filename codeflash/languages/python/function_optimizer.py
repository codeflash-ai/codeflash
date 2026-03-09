from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from rich.syntax import Syntax

from codeflash.cli_cmds.console import code_print, console, logger
from codeflash.code_utils.code_utils import unified_diff_strings
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.either import Failure, Success
from codeflash.languages.python.context.unused_definition_remover import (
    detect_unused_helper_functions,
    revert_unused_helper_functions,
)
from codeflash.languages.python.optimizer import resolve_python_function_ast
from codeflash.languages.python.static_analysis.code_extractor import get_opt_review_metrics, is_numerical_code
from codeflash.languages.python.static_analysis.code_replacer import (
    add_custom_marker_to_all_tests,
    modify_autouse_fixture,
)
from codeflash.languages.python.static_analysis.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash.models.models import TestingMode, TestResults
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.parse_test_output import calculate_function_throughput_from_test_results

if TYPE_CHECKING:
    from typing import Any

    from codeflash.either import Result
    from codeflash.languages.base import Language
    from codeflash.models.function_types import FunctionParent
    from codeflash.models.models import (
        CodeOptimizationContext,
        CodeStringsMarkdown,
        ConcurrencyMetrics,
        CoverageData,
        GeneratedTestsList,
        OriginalCodeBaseline,
        TestDiff,
        TestFileReview,
    )


class PythonFunctionOptimizer(FunctionOptimizer):
    def get_code_optimization_context(self) -> Result[CodeOptimizationContext, str]:
        from codeflash.languages.python.context import code_context_extractor

        try:
            return Success(
                code_context_extractor.get_code_optimization_context(
                    self.function_to_optimize, self.project_root, call_graph=self.call_graph
                )
            )
        except ValueError as e:
            return Failure(str(e))

    def _resolve_function_ast(
        self, source_code: str, function_name: str, parents: list[FunctionParent]
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        original_module_ast = ast.parse(source_code)
        return resolve_python_function_ast(function_name, parents, original_module_ast)

    def requires_function_ast(self) -> bool:
        return True

    def analyze_code_characteristics(self, code_context: CodeOptimizationContext) -> None:
        self.is_numerical_code = is_numerical_code(code_string=code_context.read_writable_code.flat)

    def get_optimization_review_metrics(
        self,
        source_code: str,
        file_path: Path,
        qualified_name: str,
        project_root: Path,
        tests_root: Path,
        language: Language,
    ) -> str:
        return get_opt_review_metrics(source_code, file_path, qualified_name, project_root, tests_root, language)

    def instrument_test_fixtures(self, test_paths: list[Path]) -> dict[Path, list[str]] | None:
        logger.info("Disabling all autouse fixtures associated with the generated test files")
        original_conftest_content = modify_autouse_fixture(test_paths)
        logger.info("Add custom marker to generated test files")
        add_custom_marker_to_all_tests(test_paths)
        return original_conftest_content

    def instrument_capture(self, file_path_to_helper_classes: dict[Path, set[str]]) -> None:
        from codeflash.verification.instrument_codeflash_capture import instrument_codeflash_capture

        instrument_codeflash_capture(self.function_to_optimize, file_path_to_helper_classes, self.test_cfg.tests_root)

    def display_repaired_functions(
        self, generated_tests: GeneratedTestsList, reviews: list[TestFileReview], original_sources: dict[int, str]
    ) -> None:
        """Display per-function diffs of repaired tests using libcst."""
        import libcst as cst

        def extract_functions(source: str, names: set[str]) -> dict[str, str]:
            """Extract functions by name from top-level and class bodies."""
            try:
                tree = cst.parse_module(source)
            except cst.ParserSyntaxError:
                logger.debug("Failed to parse source for diff display", exc_info=True)
                return {}
            result: dict[str, str] = {}
            for node in tree.body:
                if isinstance(node, cst.FunctionDef) and node.name.value in names:
                    result[node.name.value] = tree.code_for_node(node)
                elif isinstance(node, cst.ClassDef):
                    for child in node.body.body:
                        if isinstance(child, cst.FunctionDef) and child.name.value in names:
                            result[child.name.value] = tree.code_for_node(child)
            return result

        for review in reviews:
            gt = generated_tests.generated_tests[review.test_index]
            repaired_names = {f.function_name for f in review.functions_to_repair}
            new_source = gt.generated_original_test_source
            old_source = original_sources.get(review.test_index, "")

            old_funcs = extract_functions(old_source, repaired_names)
            new_funcs = extract_functions(new_source, repaired_names)

            for name in repaired_names:
                old_func = old_funcs.get(name, "")
                new_func = new_funcs.get(name, "")
                if not new_func:
                    continue
                console.rule()
                if old_func and old_func != new_func:
                    diff = unified_diff_strings(
                        old_func, new_func, fromfile=f"{name} (before)", tofile=f"{name} (after)"
                    )
                    if diff:
                        logger.info(f"Repaired: {name}")
                        console.print(Syntax(diff, "diff", theme="monokai"))
                        continue
                logger.info(f"Repaired: {name}")
                code_print(new_func, language=self.function_to_optimize.language)

    def should_check_coverage(self) -> bool:
        return True

    def collect_async_metrics(
        self,
        benchmarking_results: TestResults,
        code_context: CodeOptimizationContext,
        helper_code: dict[Path, str],
        test_env: dict[str, str],
    ) -> tuple[int | None, ConcurrencyMetrics | None]:
        if not self.function_to_optimize.is_async:
            return None, None

        async_throughput = calculate_function_throughput_from_test_results(
            benchmarking_results, self.function_to_optimize.function_name
        )
        logger.debug(f"Async function throughput: {async_throughput} calls/second")

        concurrency_metrics = self.run_concurrency_benchmark(
            code_context=code_context, original_helper_code=helper_code, test_env=test_env
        )
        if concurrency_metrics:
            logger.debug(
                f"Concurrency metrics: ratio={concurrency_metrics.concurrency_ratio:.2f}, "
                f"seq={concurrency_metrics.sequential_time_ns}ns, conc={concurrency_metrics.concurrent_time_ns}ns"
            )
        return async_throughput, concurrency_metrics

    def instrument_async_for_mode(self, mode: TestingMode) -> None:
        from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

        add_async_decorator_to_function(
            self.function_to_optimize.file_path, self.function_to_optimize, mode, project_root=self.project_root
        )

    def should_skip_sqlite_cleanup(self, testing_type: TestingMode, optimization_iteration: int) -> bool:
        return False

    def parse_line_profile_test_results(
        self, line_profiler_output_file: Path | None
    ) -> tuple[TestResults | dict, CoverageData | None]:
        from codeflash.verification.parse_line_profile_test_output import parse_line_profile_results

        return parse_line_profile_results(line_profiler_output_file=line_profiler_output_file)

    def compare_candidate_results(
        self,
        baseline_results: OriginalCodeBaseline,
        candidate_behavior_results: TestResults,
        optimization_candidate_index: int,
    ) -> tuple[bool, list[TestDiff]]:
        from codeflash.verification.equivalence import compare_test_results

        return compare_test_results(baseline_results.behavior_test_results, candidate_behavior_results)

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        from codeflash.languages.python.static_analysis.code_replacer import replace_function_definitions_in_module

        did_update = False
        for module_abspath, qualified_names in self.group_functions_by_file(code_context).items():
            did_update |= replace_function_definitions_in_module(
                function_names=list(qualified_names),
                optimized_code=optimized_code,
                module_abspath=module_abspath,
                preexisting_objects=code_context.preexisting_objects,
                project_root_path=self.project_root,
            )

        unused_helpers = detect_unused_helper_functions(self.function_to_optimize, code_context, optimized_code)
        if unused_helpers:
            revert_unused_helper_functions(self.project_root, unused_helpers, original_helper_code)
        return did_update

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict[str, Any]:
        candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
        if contains_jit_decorator(candidate_fto_code):
            logger.info(
                f"Skipping line profiler for {self.function_to_optimize.function_name} - code contains JIT decorator"
            )
            return {"timings": {}, "unit": 0, "str_out": ""}

        for module_abspath in original_helper_code:
            candidate_helper_code = Path(module_abspath).read_text("utf-8")
            if contains_jit_decorator(candidate_helper_code):
                logger.info(
                    f"Skipping line profiler for {self.function_to_optimize.function_name} - helper code contains JIT decorator"
                )
                return {"timings": {}, "unit": 0, "str_out": ""}

        try:
            console.rule()

            test_env = self.get_test_env(
                codeflash_loop_index=0, codeflash_test_iteration=candidate_index, codeflash_tracer_disable=1
            )
            line_profiler_output_file = add_decorator_imports(self.function_to_optimize, code_context)
            line_profile_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.LINE_PROFILE,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
                code_context=code_context,
                line_profiler_output_file=line_profiler_output_file,
            )
        finally:
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
        if isinstance(line_profile_results, TestResults) and not line_profile_results.test_results:
            logger.warning(
                f"Timeout occurred while running line profiler for original function {self.function_to_optimize.function_name}"
            )
            return {"timings": {}, "unit": 0, "str_out": ""}
        if line_profile_results["str_out"] == "":
            logger.warning(
                f"Couldn't run line profiler for original function {self.function_to_optimize.function_name}"
            )
        return line_profile_results
