from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.models.models import GeneratedTests, GeneratedTestsList
from codeflash_core.danom import Err, Ok
from codeflash_python.code_utils.config_consts import INDIVIDUAL_TESTCASE_TIMEOUT, EffortKeys, get_effort_value
from codeflash_python.verification.verifier import generate_tests

if TYPE_CHECKING:
    from codeflash.models.models import (
        CodeOptimizationContext,
        CodeStringsMarkdown,
        FunctionCalledInTest,
        FunctionSource,
    )
    from codeflash_core.danom import Result
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

from codeflash_python.verification.concolic import generate_concolic_tests

logger = logging.getLogger("codeflash_python")


class TestGenerationMixin(_Base):
    def generate_tests(
        self,
        testgen_context: CodeStringsMarkdown,
        helper_functions: list[FunctionSource],
        testgen_helper_fqns: list[str],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> Result[tuple[int, GeneratedTestsList, dict[str, set[FunctionCalledInTest]], str], str]:
        """Generate unit tests and concolic tests for the function."""
        assert self.args is not None
        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        assert len(generated_test_paths) == n_tests

        if not self.args.no_gen_tests:
            helper_fqns = testgen_helper_fqns or [definition.fully_qualified_name for definition in helper_functions]
            future_tests = self.submit_test_generation_tasks(
                self.executor, testgen_context.markdown, helper_fqns, generated_test_paths, generated_perf_test_paths
            )

        future_concolic_tests = self.executor.submit(
            generate_concolic_tests,
            self.test_cfg,
            self.args.project_root,
            self.function_to_optimize,
            self.function_to_optimize_ast,
        )

        if not self.args.no_gen_tests:
            # Wait for test futures to complete
            futures_to_wait = [*future_tests]
            if future_concolic_tests is not None:
                futures_to_wait.append(future_concolic_tests)
            concurrent.futures.wait(futures_to_wait)
        elif future_concolic_tests is not None:
            concurrent.futures.wait([future_concolic_tests])
        # Process test generation results
        tests: list[GeneratedTests] = []
        if not self.args.no_gen_tests:
            for future in future_tests:
                res = future.result()
                if res:
                    (
                        generated_test_source,
                        instrumented_behavior_test_source,
                        instrumented_perf_test_source,
                        raw_generated_test_source,
                        test_behavior_path,
                        test_perf_path,
                    ) = res
                    tests.append(
                        GeneratedTests(
                            generated_original_test_source=generated_test_source,
                            instrumented_behavior_test_source=instrumented_behavior_test_source,
                            instrumented_perf_test_source=instrumented_perf_test_source,
                            raw_generated_test_source=raw_generated_test_source,
                            behavior_file_path=test_behavior_path,
                            perf_file_path=test_perf_path,
                        )
                    )

            if not tests:
                logger.warning(
                    "Failed to generate and instrument tests for %s", self.function_to_optimize.function_name
                )
                return Err(f"/!\\ NO TESTS GENERATED for {self.function_to_optimize.function_name}")

        if future_concolic_tests is not None:
            function_to_concolic_tests, concolic_test_str = future_concolic_tests.result()
        else:
            function_to_concolic_tests, concolic_test_str = {}, None
        count_tests = len(tests)
        if concolic_test_str:
            count_tests += 1

        generated_tests = GeneratedTestsList(generated_tests=tests)
        return Ok((count_tests, generated_tests, function_to_concolic_tests, concolic_test_str))

    def submit_test_generation_tasks(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        source_code_being_tested: str,
        helper_function_names: list[str],
        generated_test_paths: list[Path],
        generated_perf_test_paths: list[Path],
    ) -> list[concurrent.futures.Future]:
        assert self.aiservice_client is not None
        assert self.test_cfg is not None
        return [
            executor.submit(
                generate_tests,
                self.aiservice_client,
                source_code_being_tested,
                self.function_to_optimize,
                helper_function_names,
                Path(self.original_module_path),
                self.test_cfg,
                INDIVIDUAL_TESTCASE_TIMEOUT,
                self.function_trace_id,
                test_index,
                test_path,
                test_perf_path,
                self.is_numerical_code,
            )
            for test_index, (test_path, test_perf_path) in enumerate(
                zip(generated_test_paths, generated_perf_test_paths)
            )
        ]

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
    ]:
        """Generate and instrument tests for the function."""
        from codeflash.models.models import TestFile, TestType
        from codeflash_python.code_utils.code_utils import get_run_tmp_file
        from codeflash_python.verification.verification_utils import get_test_file_path

        n_tests = get_effort_value(EffortKeys.N_GENERATED_TESTS, self.effort)
        source_file = Path(self.function_to_optimize.file_path)
        generated_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root,
                self.function_to_optimize.function_name,
                test_index,
                test_type="unit",
                source_file_path=source_file,
            )
            for test_index in range(n_tests)
        ]
        generated_perf_test_paths = [
            get_test_file_path(
                self.test_cfg.tests_root,
                self.function_to_optimize.function_name,
                test_index,
                test_type="perf",
                source_file_path=source_file,
            )
            for test_index in range(n_tests)
        ]

        test_results = self.generate_tests(
            testgen_context=code_context.testgen_context,
            helper_functions=code_context.helper_functions,
            testgen_helper_fqns=code_context.testgen_helper_fqns,
            generated_test_paths=generated_test_paths,
            generated_perf_test_paths=generated_perf_test_paths,
        )

        if not test_results.is_ok():
            # Result type doesn't have unwrap_err, manually get error
            return test_results  # type: ignore[return-value]

        count_tests, generated_tests, function_to_concolic_tests, concolic_test_str = test_results.unwrap()

        generated_tests = self.fixup_generated_tests(generated_tests)

        logger.debug("[PIPELINE] Processing %s generated tests", count_tests)
        for i, generated_test in enumerate(generated_tests.generated_tests):
            logger.debug(
                "[PIPELINE] Test %s: behavior_path=%s, perf_path=%s",
                i + 1,
                generated_test.behavior_file_path,
                generated_test.perf_file_path,
            )

            with generated_test.behavior_file_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_behavior_test_source)
            logger.debug("[PIPELINE] Wrote behavioral test to %s", generated_test.behavior_file_path)

            debug_file_path = get_run_tmp_file(Path("perf_test_debug.py"))
            with debug_file_path.open("w", encoding="utf-8") as debug_f:
                debug_f.write(generated_test.instrumented_perf_test_source)

            with generated_test.perf_file_path.open("w", encoding="utf8") as f:
                f.write(generated_test.instrumented_perf_test_source)
            logger.debug("[PIPELINE] Wrote perf test to %s", generated_test.perf_file_path)

            # File paths are expected to be absolute - resolved at their source (CLI, TestConfig, etc.)
            test_file_obj = TestFile(
                instrumented_behavior_file_path=generated_test.behavior_file_path,
                benchmarking_file_path=generated_test.perf_file_path,
                original_file_path=None,
                original_source=generated_test.generated_original_test_source,
                test_type=TestType.GENERATED_REGRESSION,
                tests_in_file=None,  # This is currently unused. We can discover the tests in the file if needed.
            )
            self.test_files.add(test_file_obj)
            logger.debug(
                "[PIPELINE] Added test file to collection: behavior=%s, perf=%s",
                test_file_obj.instrumented_behavior_file_path,
                test_file_obj.benchmarking_file_path,
            )

            logger.info("Generated test %s/%s", i + 1, count_tests)
        if concolic_test_str:
            logger.info("Generated test %s/%s", count_tests, count_tests)

        function_to_all_tests = {
            key: self.function_to_tests.get(key, set()) | function_to_concolic_tests.get(key, set())
            for key in set(self.function_to_tests) | set(function_to_concolic_tests)
        }
        instrumented_unittests_created_for_function = self.instrument_existing_tests(function_to_all_tests)

        assert self.args is not None
        original_conftest_content = None
        if self.args.override_fixtures:
            original_conftest_content = self.instrument_test_fixtures(generated_test_paths + generated_perf_test_paths)

        return Ok(
            (
                generated_tests,
                function_to_concolic_tests,
                concolic_test_str,
                generated_test_paths,
                generated_perf_test_paths,
                instrumented_unittests_created_for_function,
                original_conftest_content,
            )
        )
