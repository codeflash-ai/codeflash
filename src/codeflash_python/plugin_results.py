"""Mixin: ranking, explanation, PR creation, result logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.plugin_helpers import format_speedup_pct, replace_function_simple

if TYPE_CHECKING:
    from codeflash_core.models import CodeContext, GeneratedTestSuite, OptimizationResult, ScoredCandidate
    from codeflash_python.plugin import PythonPlugin as _Base
else:
    _Base = object

logger = logging.getLogger(__name__)


class PluginResultsMixin(_Base):  # type: ignore[cyclic-class-definition]
    def rank_candidates(
        self, scored: list[ScoredCandidate], context: CodeContext, trace_id: str = ""
    ) -> list[int] | None:
        assert trace_id, "trace_id must be provided"
        from codeflash_core.diff import unified_diff

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for ranking")
            return None

        diffs = [unified_diff(context.target_code, sc.candidate.code, context.target_file) for sc in scored]
        optimization_ids = [sc.candidate.candidate_id for sc in scored]
        speedups = [sc.speedup for sc in scored]

        try:
            return client.generate_ranking(
                trace_id=trace_id, diffs=diffs, optimization_ids=optimization_ids, speedups=speedups
            )
        except Exception:
            logger.exception("Ranking API call failed")
            return None

    def generate_explanation(
        self, result: OptimizationResult, context: CodeContext, trace_id: str = "", annotated_tests: str = ""
    ) -> str:
        assert trace_id, "trace_id must be provided"

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for explanation")
            return ""

        # Convert runtimes to nanoseconds and humanize, matching original
        optimized_ns = int(result.benchmark_results.total_time * 1e9)
        baseline_ns = int(optimized_ns * result.speedup) if result.speedup > 0 else 0

        try:
            return client.get_new_explanation(
                source_code=context.target_code,
                optimized_code=result.optimized_code,
                dependency_code=context.read_only_context,
                trace_id=trace_id,
                original_line_profiler_results="",
                optimized_line_profiler_results="",
                original_code_runtime=humanize_runtime(baseline_ns),
                optimized_code_runtime=humanize_runtime(optimized_ns),
                speedup=format_speedup_pct(result.speedup),
                annotated_tests=annotated_tests,
                optimization_id=result.candidate.candidate_id,
                original_explanation=result.candidate.explanation,
            )
        except Exception:
            logger.exception("Explanation generation API call failed")
            return ""

    def create_pr(
        self,
        result: OptimizationResult,
        context: CodeContext,
        trace_id: str = "",
        generated_tests: GeneratedTestSuite | None = None,
    ) -> str | None:
        from codeflash_python.models.models import TestResults as InternalTestResults
        from codeflash_python.result.create_pr import check_create_pr
        from codeflash_python.result.explanation import Explanation

        try:
            # Build original_code: file with original function (optimizer restores before returning)
            original_code = {context.target_file: context.target_file.read_text("utf-8")}

            # Build new_code: file with optimized function applied in memory
            original_source = original_code[context.target_file]
            internal_fn = context.target_function
            new_source = replace_function_simple(original_source, internal_fn, result.optimized_code)
            new_code = {context.target_file: new_source}

            # Build Explanation from optimization result
            # Use empty internal TestResults since PR comment uses runtime/speedup fields directly
            optimized_ns = int(result.benchmark_results.total_time * 1e9)
            baseline_ns = int(optimized_ns * result.speedup) if result.speedup > 0 else 0

            explanation = Explanation(
                raw_explanation_message=result.explanation or result.candidate.explanation,
                winning_behavior_test_results=InternalTestResults(),
                winning_benchmarking_test_results=InternalTestResults(),
                original_runtime_ns=baseline_ns,
                best_runtime_ns=optimized_ns,
                function_name=context.target_function.qualified_name,
                file_path=context.target_file,
            )

            # Collect generated test source
            generated_tests_str = ""
            if generated_tests and generated_tests.test_files:
                generated_tests_str = "\n\n".join(
                    tf.original_test_source for tf in generated_tests.test_files if tf.original_test_source
                )

            check_create_pr(
                original_code=original_code,
                new_code=new_code,
                explanation=explanation,
                existing_tests_source="",
                generated_original_test_source=generated_tests_str,
                function_trace_id=trace_id,
                coverage_message="",
                replay_tests="",
                root_dir=self.project_root,
                git_remote=None,
            )
        except Exception:
            logger.exception("PR creation failed")
            return None
        else:
            return None

    def log_results(
        self,
        result: OptimizationResult,
        trace_id: str,
        all_speedups: dict[str, float] | None = None,
        all_runtimes: dict[str, float] | None = None,
        all_correct: dict[str, bool] | None = None,
    ) -> None:
        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for logging")
            return

        # Use accumulated all-candidate data if available, otherwise fall back to winner-only
        speedup_ratios = all_speedups or {result.candidate.candidate_id: result.speedup}
        is_correct = all_correct or {result.candidate.candidate_id: result.test_results.passed}

        # Convert runtimes from seconds to nanoseconds (matching original API contract)
        if all_runtimes:
            optimized_runtimes = {cid: int(t * 1e9) for cid, t in all_runtimes.items()}
        else:
            optimized_runtimes = {result.candidate.candidate_id: int(result.benchmark_results.total_time * 1e9)}

        baseline_ns = int(result.benchmark_results.total_time * 1e9 * result.speedup) if result.speedup > 0 else None

        try:
            client.log_results(
                function_trace_id=trace_id,
                speedup_ratio=speedup_ratios,
                original_runtime=baseline_ns,
                optimized_runtime=optimized_runtimes,
                is_correct=is_correct,
                optimized_line_profiler_results=None,
                metadata={"best_optimization_id": result.candidate.candidate_id},
            )
        except Exception:
            logger.exception("Result logging API call failed")
