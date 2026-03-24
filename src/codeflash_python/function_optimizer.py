from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import logging
import os
import random
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast

from codeflash.models.models import OptimizationSet, TestFiles, TestingMode, TestResults
from codeflash_core.danom import Err, Ok
from codeflash_python.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path, unified_diff_strings
from codeflash_python.code_utils.config_consts import (
    PYTHON_LANGUAGE_VERSION,
    REPEAT_OPTIMIZATION_PROBABILITY,
    TOTAL_LOOPING_TIME_EFFECTIVE,
    EffortKeys,
    EffortLevel,
    get_effort_value,
)
from codeflash_python.code_utils.shell_utils import make_env_with_project_root
from codeflash_python.context.unused_helper_detection import (
    detect_unused_helper_functions,
    revert_unused_helper_functions,
)
from codeflash_python.discovery.function_filtering import was_function_previously_optimized
from codeflash_python.models.experiment_metadata import ExperimentMetadata
from codeflash_python.optimizer import resolve_python_function_ast
from codeflash_python.optimizer_mixins import (
    BaselineEstablishmentMixin,
    CandidateEvaluationMixin,
    CodeReplacementMixin,
    RefinementMixin,
    ResultProcessingMixin,
    TestExecutionMixin,
    TestGenerationMixin,
    TestReviewMixin,
)
from codeflash_python.static_analysis.code_replacer import add_custom_marker_to_all_tests, modify_autouse_fixture
from codeflash_python.static_analysis.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash_python.static_analysis.numerical_detection import is_numerical_code
from codeflash_python.static_analysis.reference_analysis import get_opt_review_metrics
from codeflash_python.telemetry.posthog_cf import ph
from codeflash_python.verification.equivalence import compare_test_results
from codeflash_python.verification.path_utils import file_name_from_test_module_name
from codeflash_python.verification.test_output_utils import calculate_function_throughput_from_test_results

if TYPE_CHECKING:
    from argparse import Namespace
    from typing import Any

    from codeflash.models.models import (
        BenchmarkKey,
        BestOptimization,
        CodeOptimizationContext,
        CodeStringsMarkdown,
        ConcurrencyMetrics,
        CoverageData,
        FunctionCalledInTest,
        GeneratedTestsList,
        OriginalCodeBaseline,
    )
    from codeflash_core.config import TestConfig
    from codeflash_core.danom import Result
    from codeflash_core.models import FunctionToOptimize
    from codeflash_python.api.aiservice import AiServiceClient
    from codeflash_python.api.types import TestDiff, TestFileReview
    from codeflash_python.context.types import DependencyResolver

logger = logging.getLogger("codeflash_python")


class FunctionOptimizer(
    TestGenerationMixin,
    TestExecutionMixin,
    TestReviewMixin,
    BaselineEstablishmentMixin,
    CandidateEvaluationMixin,
    RefinementMixin,
    ResultProcessingMixin,
    CodeReplacementMixin,
):
    def __init__(
        self,
        function_to_optimize: FunctionToOptimize,
        test_cfg: TestConfig,
        function_to_optimize_source_code: str = "",
        function_to_tests: dict[str, set[FunctionCalledInTest]] | None = None,
        function_to_optimize_ast: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        aiservice_client: AiServiceClient | None = None,
        function_benchmark_timings: dict[BenchmarkKey, int] | None = None,
        total_benchmark_timings: dict[BenchmarkKey, int] | None = None,
        args: Namespace | None = None,
        replay_tests_dir: Path | None = None,
        call_graph: DependencyResolver | None = None,
        effort_override: str | None = None,
    ) -> None:
        self.project_root = test_cfg.project_root.resolve()
        self.test_cfg = test_cfg
        self.aiservice_client = aiservice_client
        resolved_file_path = function_to_optimize.file_path.resolve()
        if resolved_file_path != function_to_optimize.file_path:
            function_to_optimize = dataclasses.replace(function_to_optimize, file_path=resolved_file_path)
        self.function_to_optimize = function_to_optimize
        self.function_to_optimize_source_code = (
            function_to_optimize_source_code
            if function_to_optimize_source_code
            else function_to_optimize.file_path.read_text(encoding="utf8")
        )
        if not function_to_optimize_ast:
            try:
                original_module_ast = ast.parse(self.function_to_optimize_source_code)
                self.function_to_optimize_ast = resolve_python_function_ast(
                    function_to_optimize.function_name, function_to_optimize.parents, original_module_ast
                )
            except SyntaxError:
                self.function_to_optimize_ast = None
        else:
            self.function_to_optimize_ast = function_to_optimize_ast
        self.function_to_tests = function_to_tests if function_to_tests else {}

        self.experiment_id = os.getenv("CODEFLASH_EXPERIMENT_ID", None)
        from codeflash_python.api.aiservice import LocalAiServiceClient

        self.local_aiservice_client = LocalAiServiceClient() if self.experiment_id else None
        self.test_files = TestFiles(test_files=[])

        default_effort = getattr(args, "effort", EffortLevel.MEDIUM.value) if args else EffortLevel.MEDIUM.value
        self.effort = effort_override or default_effort

        self.args = args  # Check defaults for these
        self.function_trace_id: str = str(uuid.uuid4())
        self.original_module_path = module_name_from_file_path(self.function_to_optimize.file_path, self.project_root)

        self.function_benchmark_timings = function_benchmark_timings if function_benchmark_timings else {}
        self.total_benchmark_timings = total_benchmark_timings if total_benchmark_timings else {}
        self.replay_tests_dir = replay_tests_dir if replay_tests_dir else None
        self.call_graph = call_graph
        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=n_tests + 3 if self.experiment_id is None else n_tests + 4
        )
        self.optimization_review = ""
        self.future_all_code_repair: list[concurrent.futures.Future] = []
        self.future_all_refinements: list[concurrent.futures.Future] = []
        self.future_adaptive_optimizations: list[concurrent.futures.Future] = []
        self.repair_counter = 0  # track how many repairs we did for each function
        self.adaptive_optimization_counter = 0  # track how many adaptive optimizations we did for each function
        self.is_numerical_code: bool | None = None
        self.code_already_exists: bool = False

    # --- Utility methods (from UtilitiesMixin) ---

    def get_trace_id(self, exp_type: str) -> str:
        """Get the trace ID for the current experiment type."""
        if self.experiment_id:
            return self.function_trace_id[:-4] + exp_type
        return self.function_trace_id

    def get_test_env(
        self, codeflash_loop_index: int, codeflash_test_iteration: int, codeflash_tracer_disable: int = 1
    ) -> dict:
        assert self.args is not None
        test_env = make_env_with_project_root(self.args.project_root)
        test_env["CODEFLASH_TEST_ITERATION"] = str(codeflash_test_iteration)
        test_env["CODEFLASH_TRACER_DISABLE"] = str(codeflash_tracer_disable)
        test_env["CODEFLASH_LOOP_INDEX"] = str(codeflash_loop_index)
        return test_env

    @staticmethod
    def cleanup_leftover_test_return_values() -> None:
        # remove leftovers from previous run
        get_run_tmp_file(Path("test_return_values_0.bin")).unlink(missing_ok=True)
        get_run_tmp_file(Path("test_return_values_0.sqlite")).unlink(missing_ok=True)

    def cleanup_generated_files(self) -> None:
        from codeflash_python.code_utils.code_utils import cleanup_paths

        paths_to_cleanup = []
        for test_file in self.test_files:
            paths_to_cleanup.append(test_file.instrumented_behavior_file_path)
            paths_to_cleanup.append(test_file.benchmarking_file_path)

        cleanup_paths(paths_to_cleanup)

    def cleanup_async_helper_file(self) -> None:
        from codeflash_python.verification.async_instrumentation import ASYNC_HELPER_FILENAME

        helper_path = self.project_root / ASYNC_HELPER_FILENAME
        helper_path.unlink(missing_ok=True)

    def get_results_not_matched_error(self) -> Err:
        logger.info("h4|Test results did not match the test results of the original code ❌")
        return Err("Test results did not match the test results of the original code.")

    # --- Python-specific implementations ---

    def get_code_optimization_context(self) -> Result[CodeOptimizationContext, str]:
        from codeflash_python.context import code_context_extractor

        try:
            return Ok(
                code_context_extractor.get_code_optimization_context(
                    self.function_to_optimize, self.project_root, call_graph=self.call_graph
                )
            )
        except ValueError as e:
            return Err(str(e))

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
        language: str,
    ) -> str:
        return get_opt_review_metrics(source_code, file_path, qualified_name, project_root, tests_root, language)

    def instrument_test_fixtures(self, test_paths: list[Path]) -> dict[Path, str] | None:
        logger.info("Disabling all autouse fixtures associated with the generated test files")
        original_conftest_content = modify_autouse_fixture(test_paths)
        logger.info("Add custom marker to generated test files")
        add_custom_marker_to_all_tests(test_paths)
        return original_conftest_content

    def instrument_capture(self, file_path_to_helper_classes: dict[Path, set[str]]) -> None:
        from codeflash_python.verification.instrument_codeflash_capture import instrument_codeflash_capture

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
                if old_func and old_func != new_func:
                    diff = unified_diff_strings(
                        old_func, new_func, fromfile=f"{name} (before)", tofile=f"{name} (after)"
                    )
                    if diff:
                        logger.info("Repaired: %s", name)
                        continue
                logger.info("Repaired: %s", name)

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
        logger.debug("Async function throughput: %s calls/second", async_throughput)

        concurrency_metrics = self.run_concurrency_benchmark(
            code_context=code_context, original_helper_code=helper_code, test_env=test_env
        )
        if concurrency_metrics:
            logger.debug(
                "Concurrency metrics: ratio=%.2f, seq=%sns, conc=%sns",
                concurrency_metrics.concurrency_ratio,
                concurrency_metrics.sequential_time_ns,
                concurrency_metrics.concurrent_time_ns,
            )
        return async_throughput, concurrency_metrics

    def instrument_async_for_mode(self, mode: TestingMode) -> None:
        from codeflash_python.verification.async_instrumentation import add_async_decorator_to_function

        add_async_decorator_to_function(
            self.function_to_optimize.file_path, self.function_to_optimize, mode, project_root=self.project_root
        )

    def parse_line_profile_test_results(
        self, line_profiler_output_file: Path | None
    ) -> tuple[TestResults | dict, CoverageData | None]:
        from codeflash_python.benchmarking.parse_line_profile_test_output import parse_line_profile_results

        return parse_line_profile_results(line_profiler_output_file=line_profiler_output_file)

    def compare_candidate_results(
        self,
        baseline_results: OriginalCodeBaseline,
        candidate_behavior_results: TestResults,
        optimization_candidate_index: int,
    ) -> tuple[bool, list[TestDiff]]:
        return compare_test_results(baseline_results.behavior_test_results, candidate_behavior_results)

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool:
        from codeflash_python.static_analysis.code_replacer import replace_function_definitions_in_module

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

    def fixup_generated_tests(self, generated_tests: GeneratedTestsList) -> GeneratedTestsList:
        return generated_tests

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict[str, Any]:
        candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
        if contains_jit_decorator(candidate_fto_code):
            logger.info(
                "Skipping line profiler for %s - code contains JIT decorator", self.function_to_optimize.function_name
            )
            return {"timings": {}, "unit": 0, "str_out": ""}

        for module_abspath in original_helper_code:
            candidate_helper_code = Path(module_abspath).read_text("utf-8")
            if contains_jit_decorator(candidate_helper_code):
                logger.info(
                    "Skipping line profiler for %s - helper code contains JIT decorator",
                    self.function_to_optimize.function_name,
                )
                return {"timings": {}, "unit": 0, "str_out": ""}

        try:
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
                "Timeout occurred while running line profiler for original function %s",
                self.function_to_optimize.function_name,
            )
            return {"timings": {}, "unit": 0, "str_out": ""}
        if not isinstance(line_profile_results, TestResults) and line_profile_results.get("str_out") == "":
            logger.warning(
                "Couldn't run line profiler for original function %s", self.function_to_optimize.function_name
            )
        return (
            line_profile_results
            if not isinstance(line_profile_results, TestResults)
            else {"timings": {}, "unit": 0, "str_out": ""}
        )

    # --- Core orchestration ---

    def can_be_optimized(self) -> Result[tuple[bool, CodeOptimizationContext, dict[Path, str]], str]:
        should_run_experiment = self.experiment_id is not None
        ph("cli-optimize-function-start", {"function_trace_id": self.function_trace_id})

        # Early check: if --no-gen-tests is set, verify there are existing tests for this function
        assert self.args is not None
        if self.args.no_gen_tests:
            from codeflash_python.models.function_types import qualified_name_with_modules_from_root

            func_qualname = qualified_name_with_modules_from_root(self.function_to_optimize, self.project_root)
            if not self.function_to_tests.get(func_qualname):
                return Err(
                    f"No existing tests found for '{self.function_to_optimize.function_name}'. "
                    f"Cannot optimize without tests when --no-gen-tests is set."
                )

        self.cleanup_leftover_test_return_values()
        file_name_from_test_module_name.cache_clear()
        ctx_result = self.get_code_optimization_context()
        if not ctx_result.is_ok():
            return Err(cast("Err", ctx_result).error)
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code

        # Random here means that we still attempt optimization with a fractional chance to see if
        # last time we could not find an optimization, maybe this time we do.
        # Random is before as a performance optimization, swapping the two 'and' statements has the same effect
        assert self.args is not None
        self.code_already_exists = was_function_previously_optimized(self.function_to_optimize, code_context, self.args)
        if random.random() > REPEAT_OPTIMIZATION_PROBABILITY and self.code_already_exists:  # noqa: S311
            return Err("Function optimization previously attempted, skipping.")

        return Ok((should_run_experiment, code_context, original_helper_code))

    # note: this isn't called by the lsp, only called by cli
    def optimize_function(self) -> Result[BestOptimization, str]:
        from codeflash_python.code_utils.code_utils import restore_conftest

        initialization_result = self.can_be_optimized()
        if not initialization_result.is_ok():
            return Err(cast("Err", initialization_result).error)
        should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()
        self.analyze_code_characteristics(code_context)

        new_code_context = code_context
        # Generate tests and optimizations in parallel
        future_tests = self.executor.submit(self.generate_and_instrument_tests, new_code_context)
        assert self.args is not None
        future_optimizations = self.executor.submit(
            self.generate_optimizations,
            read_writable_code=code_context.read_writable_code,
            read_only_context_code=code_context.read_only_context_code,
            run_experiment=should_run_experiment,
            is_numerical_code=self.is_numerical_code and not self.args.no_jit_opts,
        )

        concurrent.futures.wait([future_tests, future_optimizations])

        test_setup_result = future_tests.result()
        optimization_result = future_optimizations.result()

        if not test_setup_result.is_ok():
            return Err(cast("Err", test_setup_result).error)

        if not optimization_result.is_ok():
            return Err(cast("Err", optimization_result).error)

        (
            generated_tests,
            function_to_concolic_tests,
            concolic_test_str,
            generated_test_paths,
            generated_perf_test_paths,
            instrumented_unittests_created_for_function,
            original_conftest_content,
        ) = test_setup_result.unwrap()

        optimizations_set, function_references = optimization_result.unwrap()

        precomputed_behavioral: tuple[TestResults, CoverageData | None] | None = None
        assert self.args is not None
        if generated_tests.generated_tests and self.args.testgen_review:
            review_result = self.review_and_repair_tests(
                generated_tests=generated_tests, code_context=code_context, original_helper_code=original_helper_code
            )
            if not review_result.is_ok():
                return Err(cast("Err", review_result).error)
            generated_tests, review_behavioral, review_coverage = review_result.unwrap()
            if review_behavioral is not None:
                precomputed_behavioral = (review_behavioral, review_coverage)

        # Full baseline (behavioral + benchmarking) runs once on the final approved tests
        baseline_setup_result = self.setup_and_establish_baseline(
            code_context=code_context,
            original_helper_code=original_helper_code,
            function_to_concolic_tests=function_to_concolic_tests,
            generated_test_paths=generated_test_paths,
            generated_perf_test_paths=generated_perf_test_paths,
            instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
            original_conftest_content=original_conftest_content,
            precomputed_behavioral=precomputed_behavioral,
        )

        if not baseline_setup_result.is_ok():
            return Err(cast("Err", baseline_setup_result).error)

        (
            function_to_optimize_qualified_name,
            function_to_all_tests,
            original_code_baseline,
            test_functions_to_remove,
            file_path_to_helper_classes,
        ) = baseline_setup_result.unwrap()

        best_optimization = self.find_and_process_best_optimization(
            optimizations_set=optimizations_set,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            function_to_optimize_qualified_name=function_to_optimize_qualified_name,
            function_to_all_tests=function_to_all_tests,
            generated_tests=generated_tests,
            test_functions_to_remove=test_functions_to_remove,
            concolic_test_str=concolic_test_str,
            function_references=function_references,
        )

        # Add function to code context hash if in gh actions and code doesn't already exist
        from codeflash_python.api.cfapi import add_code_context_hash

        if not self.code_already_exists:
            add_code_context_hash(code_context.hashing_code_context_hash)

        assert self.args is not None
        if self.args.override_fixtures and original_conftest_content is not None:
            restore_conftest(original_conftest_content)
        if not best_optimization:
            return Err(f"No best optimizations found for function {self.function_to_optimize.qualified_name}")
        return Ok(best_optimization)

    def generate_optimizations(
        self,
        read_writable_code: CodeStringsMarkdown,
        read_only_context_code: str,
        run_experiment: bool = False,
        is_numerical_code: bool | None = None,
    ) -> Result[tuple[OptimizationSet, str], str]:
        """Generate optimization candidates for the function. Backend handles multi-model diversity."""
        assert self.aiservice_client is not None
        n_candidates = get_effort_value(EffortKeys.N_OPTIMIZER_CANDIDATES, self.effort)
        future_optimization_candidates = self.executor.submit(
            self.aiservice_client.optimize_code,
            read_writable_code.markdown,
            read_only_context_code,
            self.function_trace_id[:-4] + "EXP0" if run_experiment else self.function_trace_id,
            ExperimentMetadata(id=self.experiment_id, group="control") if run_experiment else None,
            language=self.function_to_optimize.language,
            language_version=PYTHON_LANGUAGE_VERSION,
            is_async=self.function_to_optimize.is_async,
            n_candidates=n_candidates,
            is_numerical_code=is_numerical_code,
        )

        future_references = self.executor.submit(
            self.get_optimization_review_metrics,
            self.function_to_optimize_source_code,
            self.function_to_optimize.file_path,
            self.function_to_optimize.qualified_name,
            self.project_root,
            self.test_cfg.tests_root,
            self.function_to_optimize.language,
        )

        futures = [future_optimization_candidates, future_references]
        future_candidates_exp = None

        if run_experiment:
            assert self.local_aiservice_client is not None
            future_candidates_exp = self.executor.submit(
                self.local_aiservice_client.optimize_code,
                read_writable_code.markdown,
                read_only_context_code,
                self.function_trace_id[:-4] + "EXP1",
                ExperimentMetadata(id=self.experiment_id, group="experiment"),
                language=self.function_to_optimize.language,
                language_version=PYTHON_LANGUAGE_VERSION,
                is_async=self.function_to_optimize.is_async,
                n_candidates=n_candidates,
            )
            futures.append(future_candidates_exp)

        # Wait for optimization futures to complete
        concurrent.futures.wait(futures)

        # Retrieve results - optimize_python_code returns list of candidates
        candidates = future_optimization_candidates.result()

        if not candidates:
            return Err(f"/!\\ NO OPTIMIZATIONS GENERATED for {self.function_to_optimize.function_name}")

        # Handle experiment results
        candidates_experiment = None
        if future_candidates_exp:
            candidates_experiment = future_candidates_exp.result()
        function_references = future_references.result()

        return Ok((OptimizationSet(control=candidates, experiment=candidates_experiment), function_references))
