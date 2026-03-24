"""Mixin: candidate evaluation, filtering, and test review/repair."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_core.config import MAX_TEST_REPAIR_CYCLES, EffortKeys, get_effort_value
from codeflash_core.models import GeneratedTestSuite, ScoredCandidate, TestOutcomeStatus
from codeflash_core.ranking import compute_speedup, score_candidate, select_best
from codeflash_core.strategy_utils import MIN_CORRECT_CANDIDATES, compute_test_diffs, restore_test_snapshots
from codeflash_core.ui import logger as ui_logger
from codeflash_core.ui import progress_bar
from codeflash_core.verification import is_equivalent

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.models import (
        BenchmarkResults,
        Candidate,
        CodeContext,
        CoverageData,
        FunctionToOptimize,
        TestResults,
    )
    from codeflash_core.strategy_utils import OptimizationRuntime

_Base = object

logger = logging.getLogger(__name__)


class DefaultStrategyEvaluationMixin(_Base):
    def evaluate_candidates(
        self,
        function: FunctionToOptimize,
        context: CodeContext,
        candidates: list[Candidate],
        baseline_tests: TestResults,
        baseline_bench: BenchmarkResults,
        runtime: OptimizationRuntime,
        behavior_test_files: list[Path] | None = None,
        perf_test_files: list[Path] | None = None,
    ) -> tuple[list[ScoredCandidate], dict[str, float], dict[str, float], dict[str, bool]]:
        """Evaluate candidates with multi-round repair, refinement, and adaptive optimization.

        Returns (scored_candidates, all_speedups, all_runtimes, all_correct) where the dicts
        accumulate data for ALL candidates (including failed ones) keyed by candidate_id.
        """
        effort = runtime.config.effort
        scored: list[ScoredCandidate] = []
        all_speedups: dict[str, float] = {}
        all_runtimes: dict[str, float] = {}
        all_correct: dict[str, bool] = {}

        queue = self.filter_candidates(function, context, candidates, runtime)

        processed = 0
        max_total = 30
        max_repairs: int = get_effort_value(EffortKeys.MAX_CODE_REPAIRS_PER_TRACE, effort)
        repair_unmatched_limit: float = get_effort_value(EffortKeys.REPAIR_UNMATCHED_PERCENTAGE_LIMIT, effort)
        top_refinement: int = get_effort_value(EffortKeys.TOP_VALID_CANDIDATES_FOR_REFINEMENT, effort)
        max_adaptive: int = get_effort_value(EffortKeys.MAX_ADAPTIVE_OPTIMIZATIONS_PER_TRACE, effort)
        adaptive_threshold: int = get_effort_value(EffortKeys.ADAPTIVE_OPTIMIZATION_THRESHOLD, effort)
        repair_counter = 0
        adaptive_counter = 0

        file_snapshots = self.capture_file_snapshots(function, context)

        while queue and processed < max_total:
            if runtime.is_cancelled():
                break
            candidate = queue.pop(0)
            processed += 1
            ui_logger.info(
                "Testing candidate %d/%d [%s]",
                processed,
                len(candidates) + processed - len(queue) - 1,
                candidate.source,
            )

            if hasattr(runtime.plugin, "_pending_code_markdown"):
                runtime.plugin._pending_code_markdown = candidate.code_markdown  # type: ignore[invalid-assignment]  # noqa: SLF001
            runtime.plugin.replace_function(function.file_path, function, candidate.code)

            try:
                with progress_bar(f"Running tests for candidate {processed}...", transient=True):
                    test_results = runtime.plugin.run_tests(
                        runtime.test_config, test_files=behavior_test_files, test_iteration=processed
                    )

                if not is_equivalent(baseline_tests, test_results, comparator=runtime.output_comparator):
                    ui_logger.info("Candidate %s failed equivalence check", candidate.candidate_id)
                    all_correct[candidate.candidate_id] = False
                    # Phase 3: Try repair if test diffs are manageable
                    # Only repair first-pass candidates (optimize, line_profiler)
                    successful_count = sum(1 for v in all_correct.values() if v)
                    if (
                        not runtime.is_cancelled()
                        and candidate.source in ("optimize", "line_profiler")
                        and repair_counter < max_repairs
                        and successful_count < MIN_CORRECT_CANDIDATES
                    ):
                        diffs = compute_test_diffs(baseline_tests, test_results)
                        total_tests = max(len(baseline_tests.outcomes), 1)
                        if diffs and len(diffs) / total_tests <= repair_unmatched_limit:
                            repaired = runtime.plugin.repair_candidate(
                                context, candidate, diffs, trace_id=runtime.trace_id
                            )
                            if repaired is not None:
                                repair_counter += 1
                                queue.append(repaired)
                    continue

                with progress_bar(f"Benchmarking candidate {processed}...", transient=True):
                    bench = runtime.plugin.run_benchmarks(
                        function, runtime.test_config, test_files=perf_test_files, test_iteration=processed
                    )
                speedup = compute_speedup(baseline_bench, bench)
                ui_logger.info("Candidate %s passed — %.2fx speedup", candidate.candidate_id, speedup)

                all_correct[candidate.candidate_id] = True
                all_speedups[candidate.candidate_id] = speedup
                all_runtimes[candidate.candidate_id] = bench.total_time

                scored.append(
                    ScoredCandidate(
                        candidate=candidate,
                        test_results=test_results,
                        benchmark_results=bench,
                        speedup=speedup,
                        score=score_candidate(speedup),
                    )
                )

                # Phase 3: Try refinement for candidates with speedup (skip already-refined)
                if not runtime.is_cancelled():
                    refinement_eligible = sum(1 for s in scored if s.speedup > 0 and s.candidate.source != "refine")
                    if speedup > 0 and candidate.source != "refine" and refinement_eligible <= top_refinement:
                        refined = runtime.plugin.refine_candidate(
                            context, scored[-1], baseline_bench, trace_id=runtime.trace_id
                        )
                        if refined:
                            queue.extend(refined)

                # Phase 3: Try adaptive after enough candidates scored
                if (
                    not runtime.is_cancelled()
                    and len(scored) >= 3
                    and adaptive_counter < max_adaptive
                    and adaptive_threshold > 0
                ):
                    adaptive_counter += 1
                    adaptive = runtime.plugin.adaptive_optimize(context, scored, trace_id=runtime.trace_id)
                    if adaptive is not None:
                        queue.append(adaptive)
            finally:
                for path, original_content in file_snapshots.items():
                    path.write_text(original_content, encoding="utf-8")

        return scored, all_speedups, all_runtimes, all_correct

    def filter_candidates(
        self,
        function: FunctionToOptimize,
        context: CodeContext,
        candidates: list[Candidate],
        runtime: OptimizationRuntime,
    ) -> list[Candidate]:
        """Validate syntax, format, and deduplicate candidates."""
        normalized_original = runtime.plugin.normalize_code(context.target_code.strip())
        seen_normalized: set[str] = {normalized_original}
        queue: list[Candidate] = []
        for c in candidates:
            if not runtime.plugin.validate_candidate(c.code):
                ui_logger.info("Candidate %s has invalid syntax, skipping", c.candidate_id)
                continue
            c.code = runtime.plugin.format_code(c.code, function.file_path)
            norm = runtime.plugin.normalize_code(c.code.strip())
            if norm in seen_normalized:
                ui_logger.info("Candidate %s is identical/duplicate, skipping", c.candidate_id)
                continue
            seen_normalized.add(norm)
            queue.append(c)
        return queue

    def capture_file_snapshots(self, function: FunctionToOptimize, context: CodeContext) -> dict[Path, str]:
        """Snapshot target + helper files for reliable restoration after code replacement."""
        snapshots: dict[Path, str] = {}
        if function.file_path.exists():
            snapshots[function.file_path] = function.file_path.read_text("utf-8")
        for h in context.helper_functions:
            if h.file_path.exists() and h.file_path not in snapshots:
                snapshots[h.file_path] = h.file_path.read_text("utf-8")
        return snapshots

    def review_and_repair_tests(
        self, generated_tests: GeneratedTestSuite | None, context: CodeContext, runtime: OptimizationRuntime
    ) -> GeneratedTestSuite | None:
        """Review and repair generated tests up to MAX_TEST_REPAIR_CYCLES iterations, then drop still-failing files."""
        if not generated_tests or not generated_tests.test_files:
            return generated_tests

        # Snapshot test file contents before repair so we can revert on failure
        pre_repair_snapshots: dict[int, tuple[str, str, str]] = {}
        for i, tf in enumerate(generated_tests.test_files):
            pre_repair_snapshots[i] = (tf.original_test_source, tf.behavior_test_source, tf.perf_test_source)

        previous_repair_errors: dict[str, str] = {}
        coverage_data: CoverageData | None = None

        for iteration in range(MAX_TEST_REPAIR_CYCLES):
            if runtime.is_cancelled():
                return generated_tests
            behavior_files = generated_tests.behavior_test_paths

            # Collect coverage on first iteration for repair guidance
            if iteration == 0:
                run_result = runtime.plugin.run_tests(
                    runtime.test_config, test_files=behavior_files, enable_coverage=True
                )
                if isinstance(run_result, tuple):
                    test_results, cov_data = run_result
                    if cov_data is not None:
                        coverage_data = cov_data
                else:
                    test_results = run_result
            else:
                test_results = runtime.plugin.run_tests(runtime.test_config, test_files=behavior_files)

            if test_results.passed:
                return generated_tests

            # Collect error messages from failing tests for next repair attempt
            failing_outcomes = [o for o in test_results.outcomes if o.status != TestOutcomeStatus.PASSED]
            for outcome in failing_outcomes:
                if outcome.error_message:
                    previous_repair_errors[outcome.test_id] = outcome.error_message

            reviews = runtime.plugin.review_generated_tests(generated_tests, context, test_results, runtime.trace_id)
            if not reviews or all(not r.functions_to_repair for r in reviews):
                restore_test_snapshots(generated_tests, pre_repair_snapshots)
                break

            repaired = runtime.plugin.repair_generated_tests(
                generated_tests,
                reviews,
                context,
                runtime.trace_id,
                previous_repair_errors=previous_repair_errors or None,
                coverage_data=coverage_data,
            )
            if repaired is None:
                restore_test_snapshots(generated_tests, pre_repair_snapshots)
                break
            generated_tests = repaired

        # Final pass: drop individual test files that still fail
        passing_files = []
        for tf in generated_tests.test_files:
            if runtime.is_cancelled():
                return generated_tests
            run_result = runtime.plugin.run_tests(runtime.test_config, test_files=[tf.behavior_test_path])
            if run_result.passed:
                passing_files.append(tf)
            else:
                ui_logger.info("Dropping failing test file: %s", tf.behavior_test_path)

        if not passing_files:
            return None

        return GeneratedTestSuite(test_files=passing_files)

    def select_best_with_ranking(
        self, scored: list[ScoredCandidate], context: CodeContext, runtime: OptimizationRuntime
    ) -> ScoredCandidate | None:
        """Select best candidate, optionally using AI ranking for multiple candidates."""
        if not scored:
            return None

        if len(scored) > 1:
            ranking = runtime.plugin.rank_candidates(scored, context, trace_id=runtime.trace_id)
            if ranking and ranking[0] < len(scored):
                return scored[ranking[0]]

        return select_best(scored)
