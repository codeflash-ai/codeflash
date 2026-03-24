from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_python.code_utils.config_consts import INDIVIDUAL_TESTCASE_TIMEOUT, TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash_python.models.models import TestingMode, TestResults, TestType
from codeflash_python.verification.instrument_existing_tests import inject_profiling_into_existing_test
from codeflash_python.verification.parse_test_output import parse_test_results
from codeflash_python.verification.test_output_utils import parse_concurrency_metrics
from codeflash_python.verification.test_runner import (
    run_behavioral_tests,
    run_benchmarking_tests,
    run_line_profile_tests,
)

if TYPE_CHECKING:
    from codeflash_python.models.models import (
        CodeOptimizationContext,
        ConcurrencyMetrics,
        CoverageData,
        FunctionCalledInTest,
        TestFiles,
    )
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class TestExecutionMixin(_Base):
    def run_and_parse_tests(
        self,
        testing_type: TestingMode,
        test_env: dict[str, str],
        test_files: TestFiles,
        optimization_iteration: int,
        testing_time: float = TOTAL_LOOPING_TIME_EFFECTIVE,
        *,
        enable_coverage: bool = False,
        pytest_min_loops: int = 5,
        pytest_max_loops: int = 250,
        code_context: CodeOptimizationContext | None = None,
        line_profiler_output_file: Path | None = None,
    ) -> tuple[TestResults | dict, CoverageData | None]:
        assert self.project_root is not None
        coverage_database_file = None
        coverage_config_file = None
        try:
            if testing_type == TestingMode.BEHAVIOR:
                result_file_path, run_result, coverage_database_file, coverage_config_file = run_behavioral_tests(
                    test_paths=test_files,
                    test_env=test_env,
                    cwd=self.project_root,
                    timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    enable_coverage=enable_coverage,
                    candidate_index=optimization_iteration,
                )
            elif testing_type == TestingMode.LINE_PROFILE:
                result_file_path, run_result = run_line_profile_tests(
                    test_paths=test_files,
                    test_env=test_env,
                    cwd=self.project_root,
                    timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    line_profile_output_file=line_profiler_output_file,
                )
            elif testing_type == TestingMode.PERFORMANCE:
                result_file_path, run_result = run_benchmarking_tests(
                    test_paths=test_files,
                    test_env=test_env,
                    cwd=self.project_root,
                    timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    min_loops=pytest_min_loops,
                    max_loops=pytest_max_loops,
                    target_duration_seconds=testing_time,
                )
            else:
                msg = f"Unexpected testing type: {testing_type}"
                raise ValueError(msg)
        except subprocess.TimeoutExpired:
            logger.exception(
                "Error running tests in %s.\nTimeout Error", ", ".join(str(f) for f in test_files.test_files)
            )
            return TestResults(), None
        if testing_type in {TestingMode.BEHAVIOR, TestingMode.PERFORMANCE}:
            assert self.test_cfg is not None
            results, coverage_results = parse_test_results(
                test_xml_path=result_file_path,
                test_files=test_files,
                test_config=self.test_cfg,
                optimization_iteration=optimization_iteration,
                run_result=run_result,
                function_name=self.function_to_optimize.qualified_name,
                source_file=self.function_to_optimize.file_path,
                code_context=code_context,
                coverage_database_file=coverage_database_file,
                coverage_config_file=coverage_config_file,
            )
            if testing_type == TestingMode.PERFORMANCE:
                results.perf_stdout = run_result.stdout
            return results, coverage_results
        return self.parse_line_profile_test_results(line_profiler_output_file)

    def run_behavioral_validation(
        self,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
        file_path_to_helper_classes: dict[Path, set[str]],
    ) -> tuple[TestResults, CoverageData | None] | None:
        """Run behavioral tests only. Returns (results, coverage) or None if no tests ran."""
        test_env = self.get_test_env(codeflash_loop_index=0, codeflash_test_iteration=0, codeflash_tracer_disable=1)
        if self.function_to_optimize.is_async:
            self.instrument_async_for_mode(TestingMode.BEHAVIOR)
        try:
            self.instrument_capture(file_path_to_helper_classes)
            behavioral_results, coverage_results = self.run_and_parse_tests(
                testing_type=TestingMode.BEHAVIOR,
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=TOTAL_LOOPING_TIME_EFFECTIVE,
                enable_coverage=True,
                code_context=code_context,
            )
        finally:
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )
        if isinstance(behavioral_results, TestResults) and behavioral_results:
            return behavioral_results, coverage_results
        return None

    def instrument_existing_tests(self, function_to_all_tests: dict[str, set[FunctionCalledInTest]]) -> set[Path]:
        from codeflash_python.models.function_types import qualified_name_with_modules_from_root
        from codeflash_python.models.models import TestFile

        assert self.project_root is not None
        existing_test_files_count = 0
        replay_test_files_count = 0
        concolic_coverage_test_files_count = 0
        unique_instrumented_test_files = set()

        func_qualname = qualified_name_with_modules_from_root(self.function_to_optimize, self.project_root)
        if func_qualname not in function_to_all_tests:
            logger.info("Did not find any pre-existing tests for '%s', will only use generated tests.", func_qualname)
            return unique_instrumented_test_files

        test_file_invocation_positions = defaultdict(list)
        tests_in_file_set = function_to_all_tests.get(func_qualname)
        if tests_in_file_set is None:
            return unique_instrumented_test_files
        for tests_in_file in tests_in_file_set:
            test_file_invocation_positions[
                (tests_in_file.tests_in_file.test_file, tests_in_file.tests_in_file.test_type)
            ].append(tests_in_file)

        for test_file, test_type in test_file_invocation_positions:
            path_obj_test_file = Path(test_file)
            if test_type == TestType.EXISTING_UNIT_TEST:
                existing_test_files_count += 1
            elif test_type == TestType.REPLAY_TEST:
                replay_test_files_count += 1
            elif test_type == TestType.CONCOLIC_COVERAGE_TEST:
                concolic_coverage_test_files_count += 1
            else:
                msg = f"Unexpected test type: {test_type}"
                raise ValueError(msg)

        if existing_test_files_count > 0 or replay_test_files_count > 0 or concolic_coverage_test_files_count > 0:
            logger.info(
                "Discovered %s existing unit test file%s, %s replay test file%s, and "
                "%s concolic coverage test file%s for %s",
                existing_test_files_count,
                "s" if existing_test_files_count != 1 else "",
                replay_test_files_count,
                "s" if replay_test_files_count != 1 else "",
                concolic_coverage_test_files_count,
                "s" if concolic_coverage_test_files_count != 1 else "",
                func_qualname,
            )

        assert self.test_cfg is not None
        for (test_file, test_type), tests_in_file_list in test_file_invocation_positions.items():
            path_obj_test_file = Path(test_file)
            # Use language-specific instrumentation
            success, injected_behavior_test = inject_profiling_into_existing_test(
                test_path=path_obj_test_file,
                call_positions=[test.position for test in tests_in_file_list],
                function_to_optimize=self.function_to_optimize,
                tests_project_root=self.test_cfg.tests_project_rootdir,
                mode=TestingMode.BEHAVIOR,
            )
            if not success:
                logger.debug("Failed to instrument test file %s for behavior testing", test_file)
                continue

            success, injected_perf_test = inject_profiling_into_existing_test(
                test_path=path_obj_test_file,
                call_positions=[test.position for test in tests_in_file_list],
                function_to_optimize=self.function_to_optimize,
                tests_project_root=self.test_cfg.tests_project_rootdir,
                mode=TestingMode.PERFORMANCE,
            )
            if not success:
                logger.debug("Failed to instrument test file %s for performance testing", test_file)
                continue

            def get_instrumented_path(original_path: str, suffix: str) -> Path:
                path_obj = Path(original_path)
                return path_obj.parent / f"{path_obj.stem}{suffix}{path_obj.suffix}"

            new_behavioral_test_path = get_instrumented_path(test_file, "__perfinstrumented")
            new_perf_test_path = get_instrumented_path(test_file, "__perfonlyinstrumented")

            if injected_behavior_test is not None:
                with new_behavioral_test_path.open("w", encoding="utf8") as _f:
                    _f.write(injected_behavior_test)
                logger.debug("[PIPELINE] Wrote instrumented behavior test to %s", new_behavioral_test_path)
            else:
                msg = "injected_behavior_test is None"
                raise ValueError(msg)

            if injected_perf_test is not None:
                with new_perf_test_path.open("w", encoding="utf8") as _f:
                    _f.write(injected_perf_test)
                logger.debug("[PIPELINE] Wrote instrumented perf test to %s", new_perf_test_path)

            unique_instrumented_test_files.add(new_behavioral_test_path)
            unique_instrumented_test_files.add(new_perf_test_path)

            if not self.test_files.get_by_original_file_path(path_obj_test_file):
                self.test_files.add(
                    TestFile(
                        instrumented_behavior_file_path=new_behavioral_test_path,
                        benchmarking_file_path=new_perf_test_path,
                        original_source=None,
                        original_file_path=Path(test_file),
                        test_type=test_type,
                        tests_in_file=[t.tests_in_file for t in tests_in_file_list],
                    )
                )

        instrumented_count = len(unique_instrumented_test_files) // 2  # each test produces behavior + perf files
        if instrumented_count > 0:
            logger.info(
                "Instrumented %s existing unit test file%s for %s",
                instrumented_count,
                "s" if instrumented_count != 1 else "",
                func_qualname,
            )
        return unique_instrumented_test_files

    def run_concurrency_benchmark(
        self, code_context: CodeOptimizationContext, original_helper_code: dict[Path, str], test_env: dict[str, str]
    ) -> ConcurrencyMetrics | None:
        """Run concurrency benchmark to measure sequential vs concurrent execution for async functions.

        This benchmark detects blocking vs non-blocking async code by comparing:
        - Sequential execution time (running N iterations one after another)
        - Concurrent execution time (running N iterations in parallel with asyncio.gather)

        Blocking code (like time.sleep) will have similar sequential and concurrent times.
        Non-blocking code (like asyncio.sleep) will be much faster when run concurrently.

        Returns:
            ConcurrencyMetrics if benchmark ran successfully, None otherwise.

        """
        if not self.function_to_optimize.is_async:
            return None

        from codeflash_python.verification.async_instrumentation import add_async_decorator_to_function

        assert self.project_root is not None
        try:
            # Add concurrency decorator to the source function
            add_async_decorator_to_function(
                self.function_to_optimize.file_path,
                self.function_to_optimize,
                TestingMode.CONCURRENCY,
                project_root=self.project_root,
            )

            # Run the concurrency benchmark tests
            concurrency_results, _ = self.run_and_parse_tests(
                testing_type=TestingMode.PERFORMANCE,  # Use performance mode for running
                test_env=test_env,
                test_files=self.test_files,
                optimization_iteration=0,
                testing_time=5.0,  # Short benchmark time
                enable_coverage=False,
                code_context=code_context,
                pytest_min_loops=1,
                pytest_max_loops=3,
            )
        except Exception as e:
            logger.debug("Concurrency benchmark failed: %s", e)
            return None
        finally:
            # Restore original code
            self.write_code_and_helpers(
                self.function_to_optimize_source_code, original_helper_code, self.function_to_optimize.file_path
            )

        # Parse concurrency metrics from stdout
        from codeflash_python.models.models import TestResults as TestResultsInternal

        if (
            concurrency_results
            and isinstance(concurrency_results, TestResultsInternal)
            and concurrency_results.perf_stdout
        ):
            return parse_concurrency_metrics(concurrency_results, self.function_to_optimize.function_name)

        return None
