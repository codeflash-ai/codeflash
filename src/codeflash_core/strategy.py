"""Optimization strategies for the core optimizer.

An OptimizationStrategy controls the full pipeline for optimizing a single
function: context extraction, candidate generation, test review, evaluation,
ranking, and explanation. The Optimizer delegates to the active strategy,
keeping discovery, indexing, and the per-function loop in the orchestrator.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar

from codeflash_core.config import EffortKeys, get_effort_value
from codeflash_core.diff import unified_diff
from codeflash_core.models import OptimizationResult
from codeflash_core.strategy_evaluation import DefaultStrategyEvaluationMixin
from codeflash_core.strategy_utils import StageSpec, cleanup_generated_tests, log_optimization_run
from codeflash_core.ui import code_print, progress_bar
from codeflash_core.ui import logger as ui_logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.models import Candidate, CodeContext, FunctionToOptimize, GeneratedTestSuite, ScoredCandidate
    from codeflash_core.strategy_utils import OptimizationRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default strategy
# ---------------------------------------------------------------------------


class DefaultStrategy(DefaultStrategyEvaluationMixin):
    """The default optimization strategy.

    Pipeline: context -> parallel testgen + candidates -> test review/repair ->
    baseline tests & benchmarks -> multi-round candidate evaluation ->
    AI-assisted ranking -> explanation.
    """

    stages: ClassVar[list[StageSpec]] = [
        StageSpec("context", "Extracting source code and dependencies to build the optimization context for the AI."),
        StageSpec("generating", "Generating unit tests for correctness validation and optimized candidates."),
        StageSpec(
            "test_review", "Reviewing generated tests for correctness and repairing any issues before validation."
        ),
        StageSpec(
            "baseline",
            "Running the original code against tests and benchmarks to establish a performance reference point.",
        ),
        StageSpec("evaluating", "Testing each candidate for correctness, then benchmarking the ones that pass."),
        StageSpec(
            "ranking", "Comparing candidate benchmarks against the baseline to find the fastest correct optimization."
        ),
        StageSpec(
            "explaining",
            "Generating a human-readable explanation of what the best optimization changed and why it's faster.",
        ),
    ]

    def optimize_function(
        self, function: FunctionToOptimize, runtime: OptimizationRuntime
    ) -> OptimizationResult | None:
        context: CodeContext | None = None
        candidates: list[Candidate] | None = None
        generated_tests: GeneratedTestSuite | None = None
        original_generated_tests: GeneratedTestSuite | None = None
        scored: list[ScoredCandidate] = []
        result: OptimizationResult | None = None
        stage = "context"
        exit_reason = ""

        try:
            head = self.extract_and_generate(function, runtime)
            if head is None:
                exit_reason = "cancelled" if runtime.is_cancelled() else "no_candidates"
                return None
            context, candidates, generated_tests = head
            original_generated_tests = generated_tests

            # Phase 2: Review and repair generated tests
            stage = "test_review"
            with progress_bar("Reviewing and repairing tests..."):
                generated_tests = self.review_and_repair_tests(generated_tests, context, runtime)
            if runtime.is_cancelled():
                exit_reason = "cancelled"
                return None

            # Determine test files
            behavior_files: list[Path] | None = None
            perf_files: list[Path] | None = None
            if generated_tests and generated_tests.test_files:
                behavior_files = generated_tests.behavior_test_paths
                perf_files = generated_tests.perf_test_paths

            # Run baseline tests
            stage = "baseline"
            with progress_bar("Running baseline tests..."):
                baseline_tests = runtime.plugin.run_tests(runtime.test_config, test_files=behavior_files)
            if runtime.is_cancelled():
                exit_reason = "cancelled"
                return None

            if not baseline_tests.passed:
                exit_reason = "baseline_failed"
                ui_logger.warning("Baseline tests failed for %s, skipping", function.qualified_name)
                return None

            # Run baseline benchmarks
            with progress_bar("Running baseline benchmarks..."):
                baseline_bench = runtime.plugin.run_benchmarks(function, runtime.test_config, test_files=perf_files)
            if runtime.is_cancelled():
                exit_reason = "cancelled"
                return None

            # Line profiler
            lp_candidates = self.get_line_profiler_candidates(function, context, runtime, perf_files)
            if lp_candidates:
                candidates.extend(lp_candidates)
            if runtime.is_cancelled():
                exit_reason = "cancelled"
                return None

            # Phase 3: Multi-round candidate evaluation
            stage = "evaluating"
            ui_logger.info("Evaluating %d candidates...", len(candidates))
            scored, all_speedups, all_runtimes, all_correct = self.evaluate_candidates(
                function,
                context,
                candidates,
                baseline_tests,
                baseline_bench,
                runtime=runtime,
                behavior_test_files=behavior_files,
                perf_test_files=perf_files,
            )
            if runtime.is_cancelled():
                exit_reason = "cancelled"
                return None

            # Phase 4+5: Rank, explain, log, finish
            stage = "ranking"
            with progress_bar("Ranking results..."):
                result = self.rank_explain_finish(
                    function, context, scored, generated_tests, runtime, all_speedups, all_runtimes, all_correct
                )
            if result:
                stage = "done"
                exit_reason = "optimized"
                code_print(result.diff)
            else:
                exit_reason = "cancelled" if runtime.is_cancelled() else "no_improvement"
            return result
        finally:
            log_optimization_run(
                function,
                runtime,
                context=context,
                candidates=candidates,
                generated_tests=generated_tests,
                scored=scored or None,
                result=result,
                stage_reached=stage,
                exit_reason=exit_reason,
            )
            if original_generated_tests:
                cleanup_generated_tests(original_generated_tests)

    # -- Building blocks (override individually or call from custom strategies) --

    def extract_and_generate(
        self, function: FunctionToOptimize, runtime: OptimizationRuntime
    ) -> tuple[CodeContext, list[Candidate], GeneratedTestSuite | None] | None:
        """Context extraction + parallel test/candidate generation.

        Returns (context, candidates, generated_tests) or None if cancelled/no candidates.
        """
        with progress_bar("Extracting context...", transient=True):
            context = runtime.plugin.extract_context(function)
        if runtime.is_cancelled():
            return None

        with progress_bar("Generating tests and candidates..."):
            generated_tests, candidates = self.generate_tests_and_candidates(function, context, runtime)

        if runtime.is_cancelled():
            return None
        if not candidates:
            ui_logger.info("No candidates returned for %s", function.qualified_name)
            return None

        ui_logger.info("Received %d candidates for %s", len(candidates), function.qualified_name)
        return context, candidates, generated_tests

    def rank_explain_finish(
        self,
        function: FunctionToOptimize,
        context: CodeContext,
        scored: list[ScoredCandidate],
        generated_tests: GeneratedTestSuite | None,
        runtime: OptimizationRuntime,
        all_speedups: dict[str, float],
        all_runtimes: dict[str, float],
        all_correct: dict[str, bool],
    ) -> OptimizationResult | None:
        """Rank candidates, explain the best, log results, and emit completion events."""
        best = self.select_best_with_ranking(scored, context, runtime)
        if runtime.is_cancelled():
            return None
        if best is None or best.speedup <= 0:
            return None

        diff = unified_diff(context.target_code, best.candidate.code, function.file_path)
        result = OptimizationResult(
            function=function,
            original_code=context.target_code,
            optimized_code=best.candidate.code,
            speedup=best.speedup,
            candidate=best.candidate,
            test_results=best.test_results,
            benchmark_results=best.benchmark_results,
            diff=diff,
        )

        annotated_tests = ""
        if generated_tests and generated_tests.test_files:
            annotated_tests = "\n\n".join(
                tf.original_test_source for tf in generated_tests.test_files if tf.original_test_source
            )

        explanation = runtime.plugin.generate_explanation(
            result, context, trace_id=runtime.trace_id, annotated_tests=annotated_tests
        )
        if explanation:
            result.explanation = explanation

        runtime.plugin.log_results(
            result, runtime.trace_id, all_speedups=all_speedups, all_runtimes=all_runtimes, all_correct=all_correct
        )
        if runtime.config.create_pr:
            try:
                runtime.plugin.create_pr(result, context, trace_id=runtime.trace_id, generated_tests=generated_tests)
            except Exception:
                logger.debug("PR creation failed", exc_info=True)

        return result

    def generate_tests_and_candidates(
        self, function: FunctionToOptimize, context: CodeContext, runtime: OptimizationRuntime
    ) -> tuple[GeneratedTestSuite | None, list[Candidate]]:
        """Generate tests and fetch candidates in parallel."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            tests_future = executor.submit(
                runtime.plugin.generate_tests, function, context, runtime.test_config, runtime.trace_id
            )
            candidates_future = executor.submit(runtime.plugin.get_candidates, context, runtime.trace_id)

            while True:
                if runtime.is_cancelled():
                    tests_future.cancel()
                    candidates_future.cancel()
                    break
                if tests_future.done() and candidates_future.done():
                    break
                runtime.cancel_event.wait(timeout=0.1)

            try:
                generated_tests = (
                    tests_future.result() if tests_future.done() and not tests_future.cancelled() else None
                )
            except Exception:
                logger.debug("Test generation failed", exc_info=True)
                generated_tests = None

            try:
                candidates = (
                    candidates_future.result() if candidates_future.done() and not candidates_future.cancelled() else []
                )
            except Exception:
                logger.debug("Candidate generation failed", exc_info=True)
                candidates = []

        return generated_tests, candidates

    def get_line_profiler_candidates(
        self,
        function: FunctionToOptimize,
        context: CodeContext,
        runtime: OptimizationRuntime,
        perf_test_files: list[Path] | None,
    ) -> list[Candidate]:
        """Run line profiler on baseline and fetch LP-guided candidates."""
        n_lp = get_effort_value(EffortKeys.N_OPTIMIZER_LP_CANDIDATES, runtime.config.effort)
        if n_lp <= 0:
            return []

        try:
            lp_data = runtime.plugin.run_line_profiler(function, runtime.test_config, test_files=perf_test_files)
            if not lp_data:
                return []

            lp_candidates = runtime.plugin.get_line_profiler_candidates(context, lp_data, runtime.trace_id)
            for c in lp_candidates:
                c.source = "line_profiler"
            return lp_candidates
        except Exception:
            logger.debug("Line profiler step failed for %s", function.qualified_name, exc_info=True)
            return []
