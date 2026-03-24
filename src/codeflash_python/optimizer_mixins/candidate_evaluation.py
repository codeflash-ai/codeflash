from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst

from codeflash_core.danom import Ok
from codeflash_python.api.types import AIServiceRefinerRequest
from codeflash_python.code_utils.code_utils import get_run_tmp_file, unified_diff_strings
from codeflash_python.code_utils.config_consts import (
    PYTHON_LANGUAGE_VERSION,
    TOTAL_LOOPING_TIME_EFFECTIVE,
    EffortKeys,
    get_effort_value,
)
from codeflash_python.models.models import (
    BestOptimization,
    OptimizedCandidate,
    OptimizedCandidateResult,
    OptimizedCandidateSource,
    TestingMode,
)
from codeflash_python.optimizer_mixins.candidate_structures import CandidateEvaluationContext, CandidateProcessor
from codeflash_python.optimizer_mixins.scoring import create_rank_dictionary_compact, diff_length
from codeflash_python.result.critic import performance_gain, quantity_of_tests_critic, speedup_critic

if TYPE_CHECKING:
    from codeflash_core.danom import Result
    from codeflash_python.api.aiservice import AiServiceClient
    from codeflash_python.models.models import CodeOptimizationContext, OriginalCodeBaseline
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
    from codeflash_python.optimizer_mixins.candidate_structures import CandidateNode
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


def normalize_code(source: str) -> str:
    from codeflash_python.normalizer import normalize_python_code

    try:
        return normalize_python_code(source, remove_docstrings=True)
    except Exception:
        return source


class CandidateEvaluationMixin(_Base):
    def handle_successful_candidate(
        self,
        candidate: OptimizedCandidate,
        candidate_result: OptimizedCandidateResult,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        original_helper_code: dict[Path, str],
        candidate_index: int,
        eval_ctx: CandidateEvaluationContext,
    ) -> BestOptimization:
        """Handle a successful optimization candidate."""
        line_profile_test_results = self.line_profiler_step(
            code_context=code_context, original_helper_code=original_helper_code, candidate_index=candidate_index
        )

        eval_ctx.record_line_profiler_result(candidate.optimization_id, line_profile_test_results["str_out"])

        replay_perf_gain = {}

        assert self.args is not None
        if self.args.benchmark:
            assert self.replay_tests_dir is not None
            assert original_code_baseline.replay_benchmarking_test_results is not None
            assert self.total_benchmark_timings is not None
            test_results_by_benchmark = candidate_result.benchmarking_test_results.group_by_benchmarks(
                list(self.total_benchmark_timings.keys()), self.replay_tests_dir, self.project_root
            )
            for benchmark_key, candidate_test_results in test_results_by_benchmark.items():
                original_code_replay_runtime = original_code_baseline.replay_benchmarking_test_results[
                    benchmark_key
                ].total_passed_runtime()
                candidate_replay_runtime = candidate_test_results.total_passed_runtime()
                replay_perf_gain[benchmark_key] = performance_gain(
                    original_runtime_ns=original_code_replay_runtime, optimized_runtime_ns=candidate_replay_runtime
                )

        assert self.args is not None
        return BestOptimization(
            candidate=candidate,
            helper_functions=code_context.helper_functions,
            code_context=code_context,
            runtime=candidate_result.best_test_runtime,
            line_profiler_test_results=line_profile_test_results,
            winning_behavior_test_results=candidate_result.behavior_test_results,
            replay_performance_gain=replay_perf_gain if self.args.benchmark else None,
            winning_benchmarking_test_results=candidate_result.benchmarking_test_results,
            winning_replay_benchmarking_test_results=candidate_result.benchmarking_test_results,
            async_throughput=candidate_result.async_throughput,
            concurrency_metrics=candidate_result.concurrency_metrics,
        )

    def select_best_optimization(
        self,
        eval_ctx: CandidateEvaluationContext,
        code_context: CodeOptimizationContext,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
        function_references: str,
    ) -> BestOptimization | None:
        """Select the best optimization from valid candidates."""
        if not eval_ctx.valid_optimizations:
            return None

        valid_candidates_with_shorter_code = []
        diff_lens_list = []  # character level diff
        speedups_list = []
        optimization_ids = []
        diff_strs = []
        runtimes_list = []

        for valid_opt in eval_ctx.valid_optimizations:
            valid_opt_normalized_code = normalize_code(valid_opt.candidate.source_code.flat.strip())
            new_candidate_with_shorter_code = OptimizedCandidate(
                source_code=eval_ctx.ast_code_to_id[valid_opt_normalized_code]["shorter_source_code"],
                optimization_id=valid_opt.candidate.optimization_id,
                explanation=valid_opt.candidate.explanation,
                source=valid_opt.candidate.source,
                parent_id=valid_opt.candidate.parent_id,
            )
            new_best_opt = BestOptimization(
                candidate=new_candidate_with_shorter_code,
                helper_functions=valid_opt.helper_functions,
                code_context=valid_opt.code_context,
                runtime=valid_opt.runtime,
                line_profiler_test_results=valid_opt.line_profiler_test_results,
                winning_behavior_test_results=valid_opt.winning_behavior_test_results,
                replay_performance_gain=valid_opt.replay_performance_gain,
                winning_benchmarking_test_results=valid_opt.winning_benchmarking_test_results,
                winning_replay_benchmarking_test_results=valid_opt.winning_replay_benchmarking_test_results,
                async_throughput=valid_opt.async_throughput,
                concurrency_metrics=valid_opt.concurrency_metrics,
            )
            valid_candidates_with_shorter_code.append(new_best_opt)
            diff_lens_list.append(
                diff_length(new_best_opt.candidate.source_code.flat, code_context.read_writable_code.flat)
            )
            diff_strs.append(
                unified_diff_strings(code_context.read_writable_code.flat, new_best_opt.candidate.source_code.flat)
            )
            speedups_list.append(
                1
                + performance_gain(
                    original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=new_best_opt.runtime
                )
            )
            optimization_ids.append(new_best_opt.candidate.optimization_id)
            runtimes_list.append(new_best_opt.runtime)

        if len(optimization_ids) > 1:
            ranking = None
            future_ranking = self.executor.submit(
                ai_service_client.generate_ranking,
                diffs=diff_strs,
                optimization_ids=optimization_ids,
                speedups=speedups_list,
                trace_id=self.get_trace_id(exp_type),
                function_references=function_references,
            )
            concurrent.futures.wait([future_ranking])
            ranking = future_ranking.result()
            if ranking:
                min_key = ranking[0]
            else:
                diff_lens_ranking = create_rank_dictionary_compact(diff_lens_list)
                runtimes_ranking = create_rank_dictionary_compact(runtimes_list)
                overall_ranking = {key: diff_lens_ranking[key] + runtimes_ranking[key] for key in diff_lens_ranking}
                min_key = min(overall_ranking, key=lambda k: overall_ranking[k])
        elif len(optimization_ids) == 1:
            min_key = 0
        else:
            return None

        return valid_candidates_with_shorter_code[min_key]

    def log_evaluation_results(
        self,
        eval_ctx: CandidateEvaluationContext,
        best_optimization: BestOptimization,
        original_code_baseline: OriginalCodeBaseline,
        ai_service_client: AiServiceClient,
        exp_type: str,
    ) -> None:
        """Log evaluation results to the AI service."""
        ai_service_client.log_results(
            function_trace_id=self.get_trace_id(exp_type),
            speedup_ratio=eval_ctx.speedup_ratios,
            original_runtime=original_code_baseline.runtime,
            optimized_runtime=eval_ctx.optimized_runtimes,
            is_correct=eval_ctx.is_correct,
            optimized_line_profiler_results=eval_ctx.optimized_line_profiler_results,
            optimizations_post=eval_ctx.optimizations_post,
            metadata={"best_optimization_id": best_optimization.candidate.optimization_id},
        )

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
    ) -> Result[OptimizedCandidateResult, str]:

        test_env = self.get_test_env(
            codeflash_loop_index=0, codeflash_test_iteration=optimization_candidate_index, codeflash_tracer_disable=1
        )

        get_run_tmp_file(Path(f"test_return_values_{optimization_candidate_index}.sqlite")).unlink(missing_ok=True)
        # Instrument codeflash capture
        candidate_fto_code = Path(self.function_to_optimize.file_path).read_text("utf-8")
        candidate_helper_code = {}
        for module_abspath in original_helper_code:
            candidate_helper_code[module_abspath] = Path(module_abspath).read_text("utf-8")
        if self.function_to_optimize.is_async:
            self.instrument_async_for_mode(TestingMode.BEHAVIOR)

        try:
            self.instrument_capture(file_path_to_helper_classes)

            candidate_behavior_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.BEHAVIOR,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=optimization_candidate_index,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
            )
        finally:
            self.write_code_and_helpers(candidate_fto_code, candidate_helper_code, self.function_to_optimize.file_path)
        from codeflash_python.models.models import TestResults

        assert isinstance(candidate_behavior_results, TestResults)
        match, diffs = self.compare_candidate_results(
            baseline_results, candidate_behavior_results, optimization_candidate_index
        )

        if match:
            logger.info("h3|Test results matched ✅")
        else:
            self.repair_if_possible(candidate, diffs, eval_ctx, code_context, len(candidate_behavior_results), exp_type)
            return self.get_results_not_matched_error()

        logger.info("loading|Running performance tests for candidate %s...", optimization_candidate_index)

        if self.function_to_optimize.is_async:
            self.instrument_async_for_mode(TestingMode.PERFORMANCE)

        try:
            candidate_benchmarking_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.PERFORMANCE,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=optimization_candidate_index,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
            )
        finally:
            if self.function_to_optimize.is_async:
                self.write_code_and_helpers(
                    candidate_fto_code, candidate_helper_code, self.function_to_optimize.file_path
                )
        # Use effective_loop_count which represents the number of timing samples across all test cases.
        from codeflash_python.models.models import TestResults as TestResultsModel

        assert isinstance(candidate_benchmarking_results, TestResultsModel)
        loop_count = candidate_benchmarking_results.effective_loop_count()

        if (total_candidate_timing := candidate_benchmarking_results.total_passed_runtime()) == 0:
            logger.warning("The overall test runtime of the optimized function is 0, couldn't run tests.")

        logger.debug("Total optimized code %s runtime (ns): %s", optimization_candidate_index, total_candidate_timing)

        candidate_async_throughput, candidate_concurrency_metrics = self.collect_async_metrics(
            candidate_benchmarking_results, code_context, candidate_helper_code, test_env
        )

        assert self.args is not None
        if self.args.benchmark:
            assert self.total_benchmark_timings is not None
            assert self.replay_tests_dir is not None
            candidate_replay_benchmarking_results = candidate_benchmarking_results.group_by_benchmarks(
                list(self.total_benchmark_timings.keys()), self.replay_tests_dir, self.project_root
            )
            from codeflash_python.code_utils.time_utils import humanize_runtime

            for benchmark_name, benchmark_results in candidate_replay_benchmarking_results.items():
                logger.debug(
                    "Benchmark %s runtime (ns): %s",
                    benchmark_name,
                    humanize_runtime(benchmark_results.total_passed_runtime()),
                )
        return Ok(
            OptimizedCandidateResult(
                max_loop_count=loop_count,
                best_test_runtime=total_candidate_timing,
                behavior_test_results=candidate_behavior_results,
                benchmarking_test_results=candidate_benchmarking_results,
                replay_benchmarking_test_results=candidate_replay_benchmarking_results if self.args.benchmark else None,
                optimization_candidate_index=optimization_candidate_index,
                total_candidate_timing=total_candidate_timing,
                async_throughput=candidate_async_throughput,
                concurrency_metrics=candidate_concurrency_metrics,
            )
        )

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
    ) -> BestOptimization | None:
        """Process a single optimization candidate.

        Returns the BestOptimization if the candidate is successful, None otherwise.
        Updates eval_ctx with results and may append to all_refinements_data.
        """
        # Cleanup temp files
        get_run_tmp_file(Path(f"test_return_values_{candidate_index}.bin")).unlink(missing_ok=True)
        get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite")).unlink(missing_ok=True)

        candidate = candidate_node.candidate

        normalized_code = normalize_code(candidate.source_code.flat.strip())

        if normalized_code == normalized_original:
            logger.info("h3|Candidate %s/%s: Identical to original code, skipping.", candidate_index, total_candidates)
            return None

        if normalized_code in eval_ctx.ast_code_to_id:
            logger.info(
                "h3|Candidate %s/%s: Duplicate of a previous candidate, skipping.", candidate_index, total_candidates
            )
            eval_ctx.handle_duplicate_candidate(candidate, normalized_code, code_context)
            return None

        logger.info("h3|Optimization candidate %s/%s:", candidate_index, total_candidates)

        # Try to replace function with optimized code
        try:
            did_update = self.replace_function_and_helpers_with_optimized_code(
                code_context=code_context,
                optimized_code=candidate.source_code,
                original_helper_code=original_helper_code,
            )
            if not did_update:
                logger.info("No functions were replaced in the optimized code. Skipping optimization candidate.")
                return None
        except (ValueError, SyntaxError, cst.ParserSyntaxError, AttributeError) as e:
            logger.exception(e)
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
            return None

        eval_ctx.register_new_candidate(normalized_code, candidate, code_context)

        # Run the optimized candidate
        run_results = self.run_optimized_candidate(
            optimization_candidate_index=candidate_index,
            baseline_results=original_code_baseline,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            eval_ctx=eval_ctx,
            code_context=code_context,
            candidate=candidate,
            exp_type=exp_type,
        )

        if not run_results.is_ok():
            eval_ctx.record_failed_candidate(candidate.optimization_id)
            return None

        candidate_result: OptimizedCandidateResult = run_results.unwrap()
        perf_gain = performance_gain(
            original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=candidate_result.best_test_runtime
        )
        eval_ctx.record_successful_candidate(candidate.optimization_id, candidate_result.best_test_runtime, perf_gain)

        # Check if this is a successful optimization
        is_successful_opt = speedup_critic(
            candidate_result,
            original_code_baseline.runtime,
            best_runtime_until_now=None,
            original_async_throughput=original_code_baseline.async_throughput,
            best_throughput_until_now=None,
            original_concurrency_metrics=original_code_baseline.concurrency_metrics,
            best_concurrency_ratio_until_now=None,
        ) and quantity_of_tests_critic(candidate_result)

        best_optimization = None

        if is_successful_opt:
            best_optimization = self.handle_successful_candidate(
                candidate=candidate,
                candidate_result=candidate_result,
                code_context=code_context,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
                candidate_index=candidate_index,
                eval_ctx=eval_ctx,
            )
            eval_ctx.valid_optimizations.append(best_optimization)

            current_tree_candidates = candidate_node.path_to_root()
            is_candidate_refined_before = any(
                c.source == OptimizedCandidateSource.REFINE for c in current_tree_candidates
            )

            assert self.aiservice_client is not None
            aiservice_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
            assert aiservice_client is not None

            if is_candidate_refined_before:
                future_adaptive_optimization = self.call_adaptive_optimize(
                    trace_id=self.get_trace_id(exp_type),
                    original_source_code=code_context.read_writable_code.markdown,
                    prev_candidates=current_tree_candidates,
                    eval_ctx=eval_ctx,
                    ai_service_client=aiservice_client,
                )
                if future_adaptive_optimization:
                    self.future_adaptive_optimizations.append(future_adaptive_optimization)
            else:
                # Refinement
                future_refinement = self.executor.submit(
                    aiservice_client.optimize_code_refinement,
                    request=[
                        AIServiceRefinerRequest(
                            optimization_id=best_optimization.candidate.optimization_id,
                            original_source_code=code_context.read_writable_code.markdown,
                            read_only_dependency_code=code_context.read_only_context_code,
                            original_code_runtime=original_code_baseline.runtime,
                            optimized_source_code=best_optimization.candidate.source_code.markdown,
                            optimized_explanation=best_optimization.candidate.explanation,
                            optimized_code_runtime=best_optimization.runtime,
                            speedup=f"{int(performance_gain(original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_optimization.runtime) * 100)}%",
                            trace_id=self.get_trace_id(exp_type),
                            original_line_profiler_results=original_code_baseline.line_profile_results["str_out"],
                            optimized_line_profiler_results=best_optimization.line_profiler_test_results["str_out"],
                            function_references=function_references,
                            language=self.function_to_optimize.language,
                            language_version=PYTHON_LANGUAGE_VERSION,
                        )
                    ],
                )
                self.future_all_refinements.append(future_refinement)

        return best_optimization

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
    ) -> BestOptimization | None:
        """Determine the best optimization candidate from a list of candidates."""
        from codeflash_python.models.experiment_metadata import ExperimentMetadata

        logger.info(
            "Determining best optimization candidate (out of %s) for %s…",
            len(candidates),
            self.function_to_optimize.qualified_name,
        )

        # Initialize evaluation context and async tasks
        eval_ctx = CandidateEvaluationContext()

        self.future_all_refinements.clear()
        self.future_all_code_repair.clear()
        self.future_adaptive_optimizations.clear()

        self.repair_counter = 0
        self.adaptive_optimization_counter = 0

        ai_service_client = self.aiservice_client if exp_type == "EXP0" else self.local_aiservice_client
        assert ai_service_client is not None, "AI service client must be set for optimization"

        assert self.args is not None
        future_line_profile_results = self.executor.submit(
            ai_service_client.optimize_python_code_line_profiler,
            source_code=code_context.read_writable_code.markdown,
            dependency_code=code_context.read_only_context_code,
            trace_id=self.get_trace_id(exp_type),
            line_profiler_results=original_code_baseline.line_profile_results["str_out"],
            n_candidates=get_effort_value(EffortKeys.N_OPTIMIZER_LP_CANDIDATES, self.effort),
            experiment_metadata=ExperimentMetadata(
                id=self.experiment_id, group="control" if exp_type == "EXP0" else "experiment"
            )
            if self.experiment_id
            else None,
            is_numerical_code=self.is_numerical_code and not self.args.no_jit_opts,
            language=self.function_to_optimize.language,
            language_version=PYTHON_LANGUAGE_VERSION,
        )

        processor = CandidateProcessor(
            candidates,
            future_line_profile_results,
            eval_ctx,
            self.effort,
            code_context.read_writable_code.markdown,
            self.future_all_refinements,
            self.future_all_code_repair,
            self.future_adaptive_optimizations,
        )
        candidate_index = 0
        normalized_original = normalize_code(code_context.read_writable_code.flat.strip())

        # Process candidates using queue-based approach
        while not processor.is_done():
            candidate_node = processor.get_next_candidate()
            if candidate_node is None:
                logger.debug("everything done, exiting")
                break

            try:
                candidate_index += 1
                self.process_single_candidate(
                    candidate_node=candidate_node,
                    candidate_index=candidate_index,
                    total_candidates=processor.candidate_len,
                    code_context=code_context,
                    original_code_baseline=original_code_baseline,
                    original_helper_code=original_helper_code,
                    file_path_to_helper_classes=file_path_to_helper_classes,
                    eval_ctx=eval_ctx,
                    exp_type=exp_type,
                    function_references=function_references,
                    normalized_original=normalized_original,
                )
            except KeyboardInterrupt as e:
                logger.exception("Optimization interrupted: %s", e)
                raise
            finally:
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )

        # Select and return the best optimization
        best_optimization = self.select_best_optimization(
            eval_ctx=eval_ctx,
            code_context=code_context,
            original_code_baseline=original_code_baseline,
            ai_service_client=ai_service_client,
            exp_type=exp_type,
            function_references=function_references,
        )

        if best_optimization:
            self.log_evaluation_results(
                eval_ctx=eval_ctx,
                best_optimization=best_optimization,
                original_code_baseline=original_code_baseline,
                ai_service_client=ai_service_client,
                exp_type=exp_type,
            )

        return best_optimization
