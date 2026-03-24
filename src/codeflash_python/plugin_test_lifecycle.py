"""Mixin: test generation, review, and repair."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_core.models import (
    GeneratedTestFile,
    GeneratedTestSuite,
    TestOutcomeStatus,
    TestRepairInfo,
    TestReviewResult,
)
from codeflash_python.plugin_helpers import coverage_data_to_details_dict
from codeflash_python.verification.test_runner import process_generated_test_strings

if TYPE_CHECKING:
    from codeflash_core.config import TestConfig
    from codeflash_core.models import CodeContext, CoverageData, FunctionToOptimize, TestResults
    from codeflash_python.plugin import PythonPlugin as _Base
else:
    _Base = object

logger = logging.getLogger(__name__)


class PluginTestLifecycleMixin(_Base):  # type: ignore[cyclic-class-definition]
    def generate_tests(
        self, function: FunctionToOptimize, context: CodeContext, test_config: TestConfig, trace_id: str = ""
    ) -> GeneratedTestSuite | None:
        from codeflash_python.code_utils.code_utils import module_name_from_file_path
        from codeflash_python.verification.verification_utils import get_test_file_path
        from codeflash_python.verification.verifier import generate_tests as _generate_tests

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for test generation")
            return None

        assert trace_id, "trace_id must be provided"
        internal_fn = function

        internal_ctx = self.last_internal_context
        source_code = internal_ctx.read_writable_code.markdown if internal_ctx else context.target_code

        # Compute is_numerical_code matching original analyze_code_characteristics
        flat_code = internal_ctx.read_writable_code.flat if internal_ctx else context.target_code
        try:
            from codeflash_python.static_analysis.numerical_detection import is_numerical_code as _is_numerical_code

            numerical = _is_numerical_code(code_string=flat_code)
        except Exception:
            numerical = None

        self.is_numerical_code = numerical

        # Cache tests_project_rootdir for use in repair_generated_tests
        tests_project_rootdir = test_config.tests_project_rootdir or test_config.project_root
        self.tests_project_rootdir = tests_project_rootdir

        module_path = Path(module_name_from_file_path(function.file_path, test_config.project_root))
        helper_names = [h.qualified_name for h in context.helper_functions]

        test_dir = test_config.tests_root
        test_dir.mkdir(parents=True, exist_ok=True)

        test_files: list[GeneratedTestFile] = []
        num_tests = 2

        for i in range(num_tests):
            behavior_path = get_test_file_path(test_dir, function.function_name, iteration=i, test_type="unit")
            perf_path = get_test_file_path(test_dir, function.function_name, iteration=i, test_type="perf")

            try:
                result = _generate_tests(
                    aiservice_client=client,
                    source_code_being_tested=source_code,
                    function_to_optimize=internal_fn,
                    helper_function_names=helper_names,
                    module_path=module_path,
                    test_cfg_project_root=tests_project_rootdir,
                    test_timeout=int(test_config.timeout),
                    function_trace_id=trace_id,
                    test_index=i,
                    test_path=behavior_path,
                    test_perf_path=perf_path,
                    is_numerical_code=numerical,
                )
            except Exception:
                logger.exception("Failed to generate test %d for %s", i, function.qualified_name)
                continue

            if result is None:
                continue

            gen_source, behavior_source, perf_source, _raw, _, _ = result

            # Write test files to disk
            behavior_path.parent.mkdir(parents=True, exist_ok=True)
            behavior_path.write_text(behavior_source, encoding="utf-8")
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            perf_path.write_text(perf_source, encoding="utf-8")

            test_files.append(
                GeneratedTestFile(
                    behavior_test_path=behavior_path,
                    perf_test_path=perf_path,
                    behavior_test_source=behavior_source,
                    perf_test_source=perf_source,
                    original_test_source=gen_source,
                )
            )

        if not test_files:
            return None

        return GeneratedTestSuite(test_files=test_files)

    def review_generated_tests(
        self, suite: GeneratedTestSuite, context: CodeContext, test_results: TestResults, trace_id: str = ""
    ) -> list[TestReviewResult]:
        assert trace_id, "trace_id must be provided"

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for test review")
            return []

        # Collect failing test function names and error messages from test results
        failed_test_functions: list[str] = []
        failure_messages: dict[str, str] = {}
        for outcome in test_results.outcomes:
            if outcome.status != TestOutcomeStatus.PASSED:
                failed_test_functions.append(outcome.test_id)
                if outcome.error_message:
                    failure_messages[outcome.test_id] = outcome.error_message

        tests_data = [
            {
                "test_index": i,
                "test_source": tf.original_test_source,
                "failed_test_functions": failed_test_functions,
                "failure_messages": failure_messages,
            }
            for i, tf in enumerate(suite.test_files)
        ]

        try:
            reviews = client.review_generated_tests(
                tests=tests_data,
                function_source_code=context.target_code,
                function_name=context.target_function.function_name,
                trace_id=trace_id,
                language="python",
            )
        except Exception:
            logger.exception("Test review API call failed")
            return []

        return [
            TestReviewResult(
                test_index=r.test_index,
                functions_to_repair=[
                    TestRepairInfo(function_name=f.function_name, reason=f.reason) for f in r.functions_to_repair
                ],
            )
            for r in reviews
        ]

    def repair_generated_tests(
        self,
        suite: GeneratedTestSuite,
        reviews: list[TestReviewResult],
        context: CodeContext,
        trace_id: str = "",
        previous_repair_errors: dict[str, str] | None = None,
        coverage_data: CoverageData | None = None,
    ) -> GeneratedTestSuite | None:
        from codeflash_python.api.types import FunctionRepairInfo
        from codeflash_python.code_utils.code_utils import module_name_from_file_path

        try:
            client = self.get_ai_client()
        except Exception:
            logger.exception("Failed to create AI client for test repair")
            return None

        coverage_details = coverage_data_to_details_dict(coverage_data) if coverage_data is not None else None
        internal_fn = context.target_function
        assert trace_id, "trace_id must be provided"

        new_test_files = list(suite.test_files)

        for review in reviews:
            if not review.functions_to_repair:
                continue

            idx = review.test_index
            if idx >= len(suite.test_files):
                continue

            tf = suite.test_files[idx]

            repair_infos = [
                FunctionRepairInfo(function_name=f.function_name, reason=f.reason) for f in review.functions_to_repair
            ]

            tests_project_rootdir = self.tests_project_rootdir or self.project_root
            module_path = Path(module_name_from_file_path(context.target_file, self.project_root))
            test_module_path = Path(module_name_from_file_path(tf.behavior_test_path, tests_project_rootdir))

            helper_names = [h.qualified_name for h in context.helper_functions]

            try:
                result = client.repair_generated_tests(
                    test_source=tf.original_test_source,
                    functions_to_repair=repair_infos,
                    function_source_code=context.target_code,
                    function_to_optimize=internal_fn,
                    helper_function_names=helper_names,
                    module_path=module_path,
                    test_module_path=test_module_path,
                    test_framework="pytest",
                    test_timeout=60,
                    trace_id=trace_id,
                    language="python",
                    previous_repair_errors=previous_repair_errors,
                    module_source_code=context.target_code,
                    coverage_details=coverage_details,
                )
            except Exception:
                logger.exception("Test repair API call failed for test %d", idx)
                continue

            if result is None:
                continue

            gen_source, behavior_source, perf_source = result

            # Process (replace temp dir placeholders)
            gen_source, behavior_source, perf_source = process_generated_test_strings(
                generated_test_source=gen_source,
                instrumented_behavior_test_source=behavior_source,
                instrumented_perf_test_source=perf_source,
                function_to_optimize=internal_fn,
                test_path=tf.behavior_test_path,
                test_cfg=None,
                project_module_system=None,
            )

            # Write repaired tests
            tf.behavior_test_path.write_text(behavior_source, encoding="utf-8")
            tf.perf_test_path.write_text(perf_source, encoding="utf-8")

            new_test_files[idx] = GeneratedTestFile(
                behavior_test_path=tf.behavior_test_path,
                perf_test_path=tf.perf_test_path,
                behavior_test_source=behavior_source,
                perf_test_source=perf_source,
                original_test_source=gen_source,
            )

        return GeneratedTestSuite(test_files=new_test_files)
