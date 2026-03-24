from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, cast

from codeflash_core.danom import Err, Ok
from codeflash_python.code_utils.code_utils import cleanup_paths
from codeflash_python.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.models.models import OriginalCodeBaseline, TestingMode, TestType
from codeflash_python.result.critic import coverage_critic, quantity_of_tests_critic

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.danom import Result
    from codeflash_python.models.models import CodeOptimizationContext, CoverageData, FunctionCalledInTest, TestResults
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class BaselineEstablishmentMixin(_Base):
    def establish_original_code_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
        precomputed_behavioral: tuple[TestResults, CoverageData | None] | None = None,
    ) -> Result[tuple[OriginalCodeBaseline, list[str]], str]:
        line_profile_results = {"timings": {}, "unit": 0, "str_out": ""}
        # For the original function - run the tests and get the runtime, plus coverage
        success = True

        test_env = self.get_test_env(codeflash_loop_index=0, codeflash_test_iteration=0, codeflash_tracer_disable=1)

        if precomputed_behavioral is not None:
            # Reuse behavioral results from the review cycle (no repairs were needed)
            behavioral_results, coverage_results = precomputed_behavioral
            logger.debug("[PIPELINE] Reusing behavioral results from test review cycle (no repairs were made)")
        else:
            if self.function_to_optimize.is_async:
                self.instrument_async_for_mode(TestingMode.BEHAVIOR)

            # Instrument codeflash capture
            try:
                self.instrument_capture(file_path_to_helper_classes)
                logger.debug("[PIPELINE] Establishing baseline with %s test files", len(self.test_files))
                for idx, tf in enumerate(self.test_files):
                    logger.debug(
                        "[PIPELINE] Test file %s: behavior=%s, perf=%s",
                        idx,
                        tf.instrumented_behavior_file_path,
                        tf.benchmarking_file_path,
                    )
                behavioral_results, coverage_results = self.run_and_parse_tests(
                    testing_type=TestingMode.BEHAVIOR,
                    test_env=test_env,
                    test_files=self.test_files,
                    optimization_iteration=0,
                    testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                    enable_coverage=True,
                    code_context=code_context,
                )
                assert isinstance(behavioral_results, TestResults)
            finally:
                # Remove codeflash capture
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )
        if not behavioral_results:
            logger.warning(
                "force_lsp|Couldn't run any tests for original function %s. Skipping optimization.",
                self.function_to_optimize.function_name,
            )
            return Err("Failed to establish a baseline for the original code - bevhavioral tests failed.")
        # Skip coverage check for non-Python languages (coverage not yet supported)
        if self.should_check_coverage() and not coverage_critic(coverage_results):
            did_pass_all_tests = all(result.did_pass for result in behavioral_results)
            if not did_pass_all_tests:
                return Err("Tests failed to pass for the original code.")
            coverage_pct = coverage_results.coverage if coverage_results else 0
            return Err(
                f"Test coverage is {coverage_pct}%, which is below the required threshold of {__import__('codeflash_python.code_utils.config_consts', fromlist=['COVERAGE_THRESHOLD']).COVERAGE_THRESHOLD}%."
            )

        line_profile_results = self.line_profiler_step(
            code_context=code_context, original_helper_code=original_helper_code, candidate_index=0
        )

        logger.debug(
            "[BENCHMARK-START] Starting benchmarking tests with %s test files", len(self.test_files.test_files)
        )
        for idx, tf in enumerate(self.test_files.test_files):
            logger.debug("[BENCHMARK-FILES] Test file %s: perf_file=%s", idx, tf.benchmarking_file_path)

        if self.function_to_optimize.is_async:
            self.instrument_async_for_mode(TestingMode.PERFORMANCE)

        try:
            benchmarking_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.PERFORMANCE,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=False,
                code_context=code_context,
            )
            assert isinstance(benchmarking_results, TestResults)
            logger.debug("[BENCHMARK-DONE] Got %s benchmark results", len(benchmarking_results.test_results))
        finally:
            if self.function_to_optimize.is_async:
                self.write_code_and_helpers(
                    self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
                )

        total_timing = benchmarking_results.total_passed_runtime()  # caution: doesn't handle the loop index
        functions_to_remove = [
            result.id.test_function_name
            for result in behavioral_results
            if (result.test_type == TestType.GENERATED_REGRESSION and not result.did_pass)
        ]

        if total_timing == 0:
            logger.warning("The overall summed benchmark runtime of the original function is 0, couldn't run tests.")
            success = False
        if not total_timing:
            logger.warning("Failed to run the tests for the original function, skipping optimization")
            success = False
        if not success:
            return Err("Failed to establish a baseline for the original code.")

        loop_count = benchmarking_results.effective_loop_count()
        logger.info(
            "h3|⌚ Original code summed runtime measured over '%s' loop%s: '%s' per full loop",
            loop_count,
            "s" if loop_count > 1 else "",
            humanize_runtime(total_timing),
        )
        logger.debug("Total original code runtime (ns): %s", total_timing)

        async_throughput, concurrency_metrics = self.collect_async_metrics(
            benchmarking_results, code_context, original_helper_code, test_env
        )

        assert self.args is not None
        if self.args.benchmark:
            assert self.replay_tests_dir is not None
            replay_benchmarking_test_results = benchmarking_results.group_by_benchmarks(
                list(self.total_benchmark_timings.keys()), self.replay_tests_dir, self.project_root
            )
        return Ok(
            (
                OriginalCodeBaseline(
                    behavior_test_results=behavioral_results,
                    benchmarking_test_results=benchmarking_results,
                    replay_benchmarking_test_results=replay_benchmarking_test_results if self.args.benchmark else None,
                    runtime=total_timing,
                    coverage_results=coverage_results,
                    line_profile_results=line_profile_results,
                    async_throughput=async_throughput,
                    concurrency_metrics=concurrency_metrics,
                ),
                functions_to_remove,
            )
        )

    def setup_and_establish_baseline(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        function_to_concolic_tests: dict[str, set[FunctionCalledInTest]],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
        instrumented_unittests_created_for_function: set[Path],
        original_conftest_content: dict[Path, str] | None,
        precomputed_behavioral: tuple[TestResults, CoverageData | None] | None = None,
    ) -> Result[
        tuple[str, dict[str, set[FunctionCalledInTest]], OriginalCodeBaseline, list[str], dict[Path, set[str]]], str
    ]:
        """Set up baseline context and establish original code baseline."""
        from codeflash_python.code_utils.code_utils import restore_conftest

        function_to_optimize_qualified_name = self.function_to_optimize.qualified_name
        function_to_all_tests = {
            key: self.function_to_tests.get(key, set()) | function_to_concolic_tests.get(key, set())
            for key in set(self.function_to_tests) | set(function_to_concolic_tests)
        }

        file_path_to_helper_classes = self.build_helper_classes_map(code_context)

        baseline_result = self.establish_original_code_baseline(
            code_context=code_context,
            original_helper_code=original_helper_code,
            file_path_to_helper_classes=file_path_to_helper_classes,
            precomputed_behavioral=precomputed_behavioral,
        )

        paths_to_cleanup = (
            generated_test_paths + generated_perf_test_paths + list(instrumented_unittests_created_for_function)
        )

        if not baseline_result.is_ok():
            assert self.args is not None
            if self.args.override_fixtures and original_conftest_content is not None:
                restore_conftest(original_conftest_content)
            cleanup_paths(paths_to_cleanup)
            self.cleanup_async_helper_file()
            return Err(cast("Err", baseline_result).error)

        original_code_baseline, test_functions_to_remove = baseline_result.unwrap()
        # Check test quantity for all languages
        quantity_ok = quantity_of_tests_critic(original_code_baseline)
        coverage_ok = coverage_critic(original_code_baseline.coverage_results) if self.should_check_coverage() else True
        if isinstance(original_code_baseline, OriginalCodeBaseline) and (not coverage_ok or not quantity_ok):
            assert self.args is not None
            if self.args.override_fixtures and original_conftest_content is not None:
                restore_conftest(original_conftest_content)
            cleanup_paths(paths_to_cleanup)
            self.cleanup_async_helper_file()
            return Err("The threshold for test confidence was not met.")

        return Ok(
            (
                function_to_optimize_qualified_name,
                function_to_all_tests,
                original_code_baseline,
                test_functions_to_remove,
                file_path_to_helper_classes,
            )
        )

    def build_helper_classes_map(self, code_context: CodeOptimizationContext) -> dict[Path, set[str]]:
        """Build a mapping of file paths to helper class names from code context."""
        file_path_to_helper_classes: dict[Path, set[str]] = defaultdict(set)
        for function_source in code_context.helper_functions:
            if (
                function_source.qualified_name != self.function_to_optimize.qualified_name
                and "." in function_source.qualified_name
            ):
                file_path_to_helper_classes[function_source.file_path].add(function_source.qualified_name.split(".")[0])
        return file_path_to_helper_classes
