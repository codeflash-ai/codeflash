from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_python.api.cfapi import create_staging, mark_optimization_success
from codeflash_python.benchmarking.utils import process_benchmark_data
from codeflash_python.code_utils import env_utils
from codeflash_python.code_utils.formatter import format_generated_code
from codeflash_python.code_utils.git_utils import git_root_dir
from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.result.create_pr import check_create_pr, existing_tests_source_for
from codeflash_python.result.critic import concurrency_gain, get_acceptance_reason, performance_gain, throughput_gain
from codeflash_python.result.explanation import Explanation
from codeflash_python.telemetry.posthog_cf import ph
from codeflash_python.verification.edit_generated_tests import (
    add_runtime_comments_to_generated_tests,
    remove_functions_from_generated_tests,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_python.models.models import (
        BestOptimization,
        CodeOptimizationContext,
        FunctionCalledInTest,
        GeneratedTestsList,
        OptimizationSet,
        OriginalCodeBaseline,
    )
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class ResultProcessingMixin(_Base):
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
    ) -> BestOptimization | None:
        """Find the best optimization candidate and process it with all required steps."""
        assert self.args is not None
        best_optimization = None
        for _u, (candidates, exp_type) in enumerate(
            zip([optimizations_set.control, optimizations_set.experiment], ["EXP0", "EXP1"])
        ):
            if candidates is None:
                continue

            best_optimization = self.determine_best_candidate(
                candidates=candidates,
                code_context=code_context,
                original_code_baseline=original_code_baseline,
                original_helper_code=original_helper_code,
                file_path_to_helper_classes=file_path_to_helper_classes,
                exp_type=exp_type,
                function_references=function_references,
            )
            ph(
                "cli-optimize-function-finished",
                {
                    "function_trace_id": self.function_trace_id[:-4] + exp_type
                    if self.experiment_id
                    else self.function_trace_id
                },
            )

            if best_optimization:
                logger.info("h2|Best candidate 🚀")
                processed_benchmark_info = None
                if self.args.benchmark and best_optimization.replay_performance_gain is not None:
                    processed_benchmark_info = process_benchmark_data(
                        replay_performance_gain=best_optimization.replay_performance_gain,
                        fto_benchmark_timings=self.function_benchmark_timings,
                        total_benchmark_timings=self.total_benchmark_timings,
                    )
                acceptance_reason = get_acceptance_reason(
                    original_runtime_ns=original_code_baseline.runtime,
                    optimized_runtime_ns=best_optimization.runtime,
                    original_async_throughput=original_code_baseline.async_throughput,
                    optimized_async_throughput=best_optimization.async_throughput,
                    original_concurrency_metrics=original_code_baseline.concurrency_metrics,
                    optimized_concurrency_metrics=best_optimization.concurrency_metrics,
                )
                explanation = Explanation(
                    raw_explanation_message=best_optimization.candidate.explanation,
                    winning_behavior_test_results=best_optimization.winning_behavior_test_results,
                    winning_benchmarking_test_results=best_optimization.winning_benchmarking_test_results,
                    original_runtime_ns=original_code_baseline.runtime,
                    best_runtime_ns=best_optimization.runtime,
                    function_name=function_to_optimize_qualified_name,
                    file_path=self.function_to_optimize.file_path,
                    benchmark_details=processed_benchmark_info.benchmark_details if processed_benchmark_info else None,
                    original_async_throughput=original_code_baseline.async_throughput,
                    best_async_throughput=best_optimization.async_throughput,
                    original_concurrency_metrics=original_code_baseline.concurrency_metrics,
                    best_concurrency_metrics=best_optimization.concurrency_metrics,
                    acceptance_reason=acceptance_reason,
                )

                self.replace_function_and_helpers_with_optimized_code(
                    code_context=code_context,
                    optimized_code=best_optimization.candidate.source_code,
                    original_helper_code=original_helper_code,
                )

                new_code, new_helper_code = self.reformat_code_and_helpers(
                    code_context.helper_functions,
                    explanation.file_path,
                    self.function_to_optimize_source_code,
                    optimized_context=best_optimization.candidate.source_code,
                )

                original_code_combined = original_helper_code.copy()
                original_code_combined[explanation.file_path] = self.function_to_optimize_source_code
                new_code_combined = new_helper_code.copy()
                new_code_combined[explanation.file_path] = new_code
                self.process_review(
                    original_code_baseline,
                    best_optimization,
                    generated_tests,
                    test_functions_to_remove,
                    concolic_test_str,
                    original_code_combined,
                    new_code_combined,
                    explanation,
                    function_to_all_tests,
                    exp_type,
                    original_helper_code,
                    code_context,
                    function_references,
                )
        return best_optimization

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
    ) -> None:
        from codeflash_python.api.types import OptimizationReviewResult
        from codeflash_python.models.function_types import qualified_name_with_modules_from_root

        assert self.args is not None
        assert self.aiservice_client is not None
        coverage_message = (
            original_code_baseline.coverage_results.build_message()
            if original_code_baseline.coverage_results
            else "Coverage data not available"
        )

        generated_tests = remove_functions_from_generated_tests(generated_tests, test_functions_to_remove)
        map_gen_test_file_to_no_of_tests = original_code_baseline.behavior_test_results.file_to_no_of_tests(
            test_functions_to_remove
        )

        original_runtime_by_test = original_code_baseline.benchmarking_test_results.usable_runtime_data_by_test_case()
        optimized_runtime_by_test = (
            best_optimization.winning_benchmarking_test_results.usable_runtime_data_by_test_case()
        )

        generated_tests = add_runtime_comments_to_generated_tests(
            generated_tests, original_runtime_by_test, optimized_runtime_by_test, self.test_cfg.tests_project_rootdir
        )

        generated_tests_str = ""
        code_lang = self.function_to_optimize.language
        for test in generated_tests.generated_tests:
            if any(
                test_file.name == test.behavior_file_path.name and count > 0
                for test_file, count in map_gen_test_file_to_no_of_tests.items()
            ):
                formatted_generated_test = format_generated_code(
                    test.generated_original_test_source, self.args.formatter_cmds
                )
                generated_tests_str += f"```{code_lang}\n{formatted_generated_test}\n```"
                generated_tests_str += "\n\n"

        if concolic_test_str:
            formatted_generated_test = format_generated_code(concolic_test_str, self.args.formatter_cmds)
            generated_tests_str += f"```{code_lang}\n{formatted_generated_test}\n```\n\n"

        existing_tests, replay_tests, _concolic_tests = existing_tests_source_for(
            qualified_name_with_modules_from_root(self.function_to_optimize, self.project_root),
            function_to_all_tests,
            test_cfg=self.test_cfg,
            original_runtimes_all=original_runtime_by_test,
            optimized_runtimes_all=optimized_runtime_by_test,
            test_files_registry=self.test_files,
        )
        original_throughput_str = None
        optimized_throughput_str = None
        throughput_improvement_str = None
        original_concurrency_ratio_str = None
        optimized_concurrency_ratio_str = None
        concurrency_improvement_str = None

        if (
            self.function_to_optimize.is_async
            and original_code_baseline.async_throughput is not None
            and best_optimization.async_throughput is not None
        ):
            original_throughput_str = f"{original_code_baseline.async_throughput} operations/second"
            optimized_throughput_str = f"{best_optimization.async_throughput} operations/second"
            throughput_improvement_value = throughput_gain(
                original_throughput=original_code_baseline.async_throughput,
                optimized_throughput=best_optimization.async_throughput,
            )
            throughput_improvement_str = f"{throughput_improvement_value * 100:.1f}%"

        if original_code_baseline.concurrency_metrics is not None and best_optimization.concurrency_metrics is not None:
            original_concurrency_ratio_str = f"{original_code_baseline.concurrency_metrics.concurrency_ratio:.2f}x"
            optimized_concurrency_ratio_str = f"{best_optimization.concurrency_metrics.concurrency_ratio:.2f}x"
            conc_improvement_value = concurrency_gain(
                original_code_baseline.concurrency_metrics, best_optimization.concurrency_metrics
            )
            concurrency_improvement_str = f"{conc_improvement_value * 100:.1f}%"

        new_explanation_raw_str = self.aiservice_client.get_new_explanation(
            source_code=code_context.read_writable_code.flat,
            dependency_code=code_context.read_only_context_code,
            trace_id=self.function_trace_id[:-4] + exp_type if self.experiment_id else self.function_trace_id,
            optimized_code=best_optimization.candidate.source_code.flat,
            original_line_profiler_results=original_code_baseline.line_profile_results["str_out"],
            optimized_line_profiler_results=best_optimization.line_profiler_test_results["str_out"],
            original_code_runtime=humanize_runtime(original_code_baseline.runtime),
            optimized_code_runtime=humanize_runtime(best_optimization.runtime),
            speedup=f"{int(performance_gain(original_runtime_ns=original_code_baseline.runtime, optimized_runtime_ns=best_optimization.runtime) * 100)}%",
            annotated_tests=generated_tests_str,
            optimization_id=best_optimization.candidate.optimization_id,
            original_explanation=best_optimization.candidate.explanation,
            original_throughput=original_throughput_str,
            optimized_throughput=optimized_throughput_str,
            throughput_improvement=throughput_improvement_str,
            function_references=function_references,
            acceptance_reason=explanation.acceptance_reason.value,
            original_concurrency_ratio=original_concurrency_ratio_str,
            optimized_concurrency_ratio=optimized_concurrency_ratio_str,
            concurrency_improvement=concurrency_improvement_str,
        )
        new_explanation = Explanation(
            raw_explanation_message=new_explanation_raw_str or explanation.raw_explanation_message,
            winning_behavior_test_results=explanation.winning_behavior_test_results,
            winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
            original_runtime_ns=explanation.original_runtime_ns,
            best_runtime_ns=explanation.best_runtime_ns,
            function_name=explanation.function_name,
            file_path=explanation.file_path,
            benchmark_details=explanation.benchmark_details,
            original_async_throughput=explanation.original_async_throughput,
            best_async_throughput=explanation.best_async_throughput,
            original_concurrency_metrics=explanation.original_concurrency_metrics,
            best_concurrency_metrics=explanation.best_concurrency_metrics,
            acceptance_reason=explanation.acceptance_reason,
        )
        self.log_successful_optimization(new_explanation, generated_tests, exp_type)

        best_optimization.explanation_v2 = new_explanation.explanation_message()

        data = {
            "original_code": original_code_combined,
            "new_code": new_code_combined,
            "explanation": new_explanation,
            "existing_tests_source": existing_tests,
            "generated_original_test_source": generated_tests_str,
            "function_trace_id": self.function_trace_id[:-4] + exp_type
            if self.experiment_id
            else self.function_trace_id,
            "coverage_message": coverage_message,
            "replay_tests": replay_tests,
            # "concolic_tests": concolic_tests,
            "language": self.function_to_optimize.language,
            # "original_line_profiler": original_code_baseline.line_profile_results.get("str_out", ""),
            # "optimized_line_profiler": best_optimization.line_profiler_test_results.get("str_out", ""),
        }

        raise_pr = not self.args.no_pr
        staging_review = self.args.staging_review

        opt_review_result = OptimizationReviewResult(review="", explanation="")
        # this will now run regardless of pr, staging review flags
        try:
            opt_review_result = self.aiservice_client.get_optimization_review(
                **data,
                calling_fn_details=function_references,  # type: ignore[invalid-argument-type]
            )
        except Exception as e:
            logger.debug("optimization review response failed, investigate %s", e)
        data["optimization_review"] = opt_review_result.review
        self.optimization_review = opt_review_result.review

        # Display the reviewer result to the user
        from git import Repo as GitRepo

        if raise_pr or staging_review:
            data["root_dir"] = git_root_dir(GitRepo(str(self.args.module_root), search_parent_directories=True))
        if raise_pr and not staging_review and opt_review_result.review != "low":
            # Ensure root_dir is set for PR creation (needed for async functions that skip opt_review)
            if "root_dir" not in data:
                data["root_dir"] = git_root_dir(GitRepo(str(self.args.module_root), search_parent_directories=True))
            data["git_remote"] = self.args.git_remote
            # Remove language from data dict as check_create_pr doesn't accept it
            pr_data = {k: v for k, v in data.items() if k != "language"}
            check_create_pr(**pr_data)  # type: ignore[invalid-argument-type]
        elif staging_review:
            response = create_staging(**data)  # type: ignore[invalid-argument-type]

        else:
            # Mark optimization success since no PR will be created
            mark_optimization_success(
                trace_id=self.function_trace_id, is_optimization_found=best_optimization is not None
            )

        # If worktree mode, do not revert code and helpers, otherwise we would have an empty diff when writing the patch in the lsp
        if self.args.worktree:
            return

        if raise_pr and (
            self.args.all
            or env_utils.get_pr_number()
            or self.args.replay_test
            or (self.args.file and not self.args.function)
        ):
            self.revert_code_and_helpers(original_helper_code)
            return

        if staging_review:
            # always revert code and helpers when staging review
            self.revert_code_and_helpers(original_helper_code)
            return

    def log_successful_optimization(
        self, explanation: Explanation, generated_tests: GeneratedTestsList, exp_type: str
    ) -> None:
        ph(
            "cli-optimize-success",
            {
                "function_trace_id": self.function_trace_id[:-4] + exp_type
                if self.experiment_id
                else self.function_trace_id,
                "speedup_x": explanation.speedup_x,
                "speedup_pct": explanation.speedup_pct,
                "best_runtime": explanation.best_runtime_ns,
                "original_runtime": explanation.original_runtime_ns,
                "winning_test_results": {
                    tt.to_name(): v
                    for tt, v in explanation.winning_behavior_test_results.get_test_pass_fail_report_by_type().items()
                },
            },
        )
