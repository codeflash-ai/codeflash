"""Type-only protocol declaring the shared interface of FunctionOptimizer.

Mixin classes inherit from this under TYPE_CHECKING so that type checkers can
resolve cross-mixin attribute and method accesses on ``self``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import ast
    import concurrent.futures
    from argparse import Namespace
    from pathlib import Path

    from codeflash_core.config import TestConfig
    from codeflash_core.danom import Err, Result
    from codeflash_core.models import FunctionToOptimize
    from codeflash_python.api.aiservice import AiServiceClient
    from codeflash_python.api.types import TestDiff, TestFileReview
    from codeflash_python.context.types import DependencyResolver
    from codeflash_python.models.models import (
        BenchmarkKey,
        BestOptimization,
        CodeOptimizationContext,
        CodeStringsMarkdown,
        ConcurrencyMetrics,
        CoverageData,
        FunctionCalledInTest,
        FunctionSource,
        GeneratedTestsList,
        OptimizationSet,
        OptimizedCandidate,
        OptimizedCandidateResult,
        OriginalCodeBaseline,
        TestFiles,
        TestingMode,
        TestResults,
    )
    from codeflash_python.optimizer_mixins.candidate_structures import CandidateEvaluationContext, CandidateNode
    from codeflash_python.result.explanation import Explanation


class FunctionOptimizerProtocol(Protocol):
    # -- Instance attributes (set in FunctionOptimizer.__init__) --

    project_root: Path
    test_cfg: TestConfig
    aiservice_client: AiServiceClient | None
    local_aiservice_client: AiServiceClient | None
    function_to_optimize: FunctionToOptimize
    function_to_optimize_source_code: str
    function_to_optimize_ast: ast.FunctionDef | ast.AsyncFunctionDef | None
    function_to_tests: dict[str, set[FunctionCalledInTest]]
    experiment_id: str | None
    test_files: TestFiles
    effort: str
    args: Namespace | None
    function_trace_id: str
    original_module_path: str
    function_benchmark_timings: dict[BenchmarkKey, int]
    total_benchmark_timings: dict[BenchmarkKey, int]
    replay_tests_dir: Path | None
    call_graph: DependencyResolver | None
    executor: concurrent.futures.ThreadPoolExecutor
    optimization_review: str
    future_all_code_repair: list[concurrent.futures.Future]
    future_all_refinements: list[concurrent.futures.Future]
    future_adaptive_optimizations: list[concurrent.futures.Future]
    repair_counter: int
    adaptive_optimization_counter: int
    is_numerical_code: bool | None
    code_already_exists: bool

    # -- Methods defined in FunctionOptimizer --

    def get_test_env(
        self, codeflash_loop_index: int, codeflash_test_iteration: int, codeflash_tracer_disable: int = ...
    ) -> dict: ...

    def get_trace_id(self, exp_type: str) -> str: ...

    def cleanup_async_helper_file(self) -> None: ...

    def get_results_not_matched_error(self) -> Err: ...

    def instrument_capture(self, file_path_to_helper_classes: dict[Path, set[str]]) -> None: ...

    def instrument_async_for_mode(self, mode: TestingMode) -> None: ...

    def should_check_coverage(self) -> bool: ...

    def parse_line_profile_test_results(
        self, line_profiler_output_file: Path | None
    ) -> tuple[TestResults | dict, CoverageData | None]: ...

    def compare_candidate_results(
        self,
        baseline_results: OriginalCodeBaseline,
        candidate_behavior_results: TestResults,
        optimization_candidate_index: int,
    ) -> tuple[bool, list[TestDiff]]: ...

    def replace_function_and_helpers_with_optimized_code(
        self,
        code_context: CodeOptimizationContext,
        optimized_code: CodeStringsMarkdown,
        original_helper_code: dict[Path, str],
    ) -> bool: ...

    def collect_async_metrics(
        self,
        benchmarking_results: TestResults,
        code_context: CodeOptimizationContext,
        helper_code: dict[Path, str],
        test_env: dict[str, str],
    ) -> tuple[int | None, ConcurrencyMetrics | None]: ...

    def line_profiler_step(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], candidate_index: int
    ) -> dict[str, Any]: ...

    def display_repaired_functions(
        self, generated_tests: GeneratedTestsList, reviews: list[TestFileReview], original_sources: dict[int, str]
    ) -> None: ...

    def instrument_test_fixtures(self, test_paths: list[Path]) -> dict[Path, str] | None: ...

    def fixup_generated_tests(self, generated_tests: GeneratedTestsList) -> GeneratedTestsList: ...

    # -- Methods from CodeReplacementMixin --

    @staticmethod
    def write_code_and_helpers(original_code: str, original_helper_code: dict[Path, str], path: Path) -> None: ...

    def reformat_code_and_helpers(
        self,
        helper_functions: list[FunctionSource],
        path: Path,
        original_code: str,
        optimized_context: CodeStringsMarkdown,
    ) -> tuple[str, dict[Path, str]]: ...

    def group_functions_by_file(self, code_context: CodeOptimizationContext) -> dict[Path, set[str]]: ...

    def revert_code_and_helpers(self, original_helper_code: dict[Path, str]) -> None: ...

    # -- Methods from TestExecutionMixin --

    def run_and_parse_tests(
        self,
        testing_type: TestingMode,
        test_env: dict[str, str],
        test_files: TestFiles,
        optimization_iteration: int,
        testing_time: float = ...,
        *,
        enable_coverage: bool = ...,
        pytest_min_loops: int = ...,
        pytest_max_loops: int = ...,
        code_context: CodeOptimizationContext | None = ...,
        line_profiler_output_file: Path | None = ...,
    ) -> tuple[TestResults | dict, CoverageData | None]: ...

    def run_behavioral_validation(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> tuple[TestResults, CoverageData | None] | None: ...

    def instrument_existing_tests(self, function_to_all_tests: dict[str, set[FunctionCalledInTest]]) -> set[Path]: ...

    def run_concurrency_benchmark(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], test_env: dict[str, str]
    ) -> ConcurrencyMetrics | None: ...

    # -- Methods from BaselineEstablishmentMixin --

    def establish_original_code_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        precomputed_behavioral: tuple[TestResults, CoverageData | None] | None = ...,
    ) -> Result[tuple[OriginalCodeBaseline, list[str]], str]: ...

    def setup_and_establish_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        function_to_concolic_tests: dict[str, set[FunctionCalledInTest]],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        instrumented_unittests_created_for_function: set[Path],
        original_conftest_content: dict[Path, str] | None,
        precomputed_behavioral: tuple[TestResults, CoverageData | None] | None = ...,
    ) -> Result[
        tuple[str, dict[str, set[FunctionCalledInTest]], OriginalCodeBaseline, list[str], dict[Path, set[str]]], str
    ]: ...

    def build_helper_classes_map(self, code_context: CodeOptimizationContext) -> dict[Path, set[str]]: ...

    # -- Methods from TestGenerationMixin --

    def generate_tests(
        self,
        testgen_context: CodeStringsMarkdown,
        helper_functions: list[FunctionSource],
        testgen_helper_fqns: list[str],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> Result[tuple[int, GeneratedTestsList, dict[str, set[FunctionCalledInTest]], str], str]: ...

    def submit_test_generation_tasks(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        source_code_being_tested: str,
        helper_function_names: list[str],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> list[concurrent.futures.Future]: ...

    def generate_and_instrument_tests(
        self, code_context: CodeOptimizationContext
    ) -> Result[
        tuple[
            GeneratedTestsList,
            dict[str, set[FunctionCalledInTest]],
            str,
            list[Path],
            list[Path],
            set[Path],
            dict[Path, str] | None,
        ],
        str,
    ]: ...

    # -- Methods from TestReviewMixin --

    def review_and_repair_tests(
        self,
        generated_tests: GeneratedTestsList,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
    ) -> Result[tuple[GeneratedTestsList, TestResults | None, CoverageData | None], str]: ...

    # -- Methods from CandidateEvaluationMixin --

    def handle_successful_candidate(
        self,
        candidate: OptimizedCandidate,
        candidate_result: OptimizedCandidateResult,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        candidate_index: int,
        eval_ctx: CandidateEvaluationContext,
    ) -> BestOptimization: ...

    def select_best_optimization(
        self,
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None: ...

    def log_evaluation_results(
        self,
        eval_ctx: CandidateEvaluationContext,
        best_optimization: BestOptimization,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
    ) -> None: ...

    def run_optimized_candidate(
        self,
        *,
        optimization_candidate_index: int,
        baseline_results: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        candidate: OptimizedCandidate,
        exp_type: str,
    ) -> Result[OptimizedCandidateResult, str]: ...

    def process_single_candidate(
        self,
        candidate_node: CandidateNode,
        candidate_index: int,
        total_candidates: int,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        eval_ctx: CandidateEvaluationContext,
        exp_type: str,
        function_references: str,
        normalized_original: str,
    ) -> BestOptimization | None: ...

    def determine_best_candidate(
        self,
        *,
        candidates: list[OptimizedCandidate],
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None: ...

    def call_adaptive_optimize(
        self,
        trace_id: str,
        original_source_code: str,
        prev_candidates: list[OptimizedCandidate],
        eval_ctx: CandidateEvaluationContext,
        ai_service_client: AiServiceClient,
    ) -> concurrent.futures.Future[OptimizedCandidate | None] | None: ...

    def repair_optimization(
        self,
        original_source_code: str,
        modified_source_code: str,
        test_diffs: list[TestDiff],
        trace_id: str,
        optimization_id: str,
        ai_service_client: AiServiceClient,
        executor: concurrent.futures.ThreadPoolExecutor,
        language: str = ...,
    ) -> concurrent.futures.Future[OptimizedCandidate | None]: ...

    def repair_if_possible(
        self,
        candidate: OptimizedCandidate,
        diffs: list[TestDiff],
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        test_results_count: int,
        exp_type: str,
    ) -> None: ...

    # -- Methods from ResultProcessingMixin --

    def find_and_process_best_optimization(
        self,
        optimizations_set: OptimizationSet,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        function_to_optimize_qualified_name: str,
        function_to_all_tests: dict[str, set[FunctionCalledInTest]],
        generated_tests: GeneratedTestsList,
        test_functions_to_remove: list[str],
        concolic_test_str: str | None,
        function_references: str,
    ) -> BestOptimization | None: ...

    def process_review(
        self,
        original_code_baseline: OriginalCodeBaseline,
        best_optimization: BestOptimization,
        generated_tests: GeneratedTestsList,
        test_functions_to_remove: list[str],
        concolic_test_str: str | None,
        original_code_combined: dict[Path, str],
        new_code_combined: dict[Path, str],
        explanation: Explanation,
        function_to_all_tests: dict[str, set[FunctionCalledInTest]],
        exp_type: str,
        original_helper_code: dict[Path, str],
        code_context: CodeOptimizationContext,
        function_references: str,
    ) -> None: ...

    def log_successful_optimization(
        self, explanation: Explanation, generated_tests: GeneratedTestsList, exp_type: str
    ) -> None: ...
