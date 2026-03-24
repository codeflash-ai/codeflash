from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_core.danom import Err, Ok
from codeflash_python.code_utils.code_utils import encoded_tokens_len, module_name_from_file_path
from codeflash_python.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    INDIVIDUAL_TESTCASE_TIMEOUT,
    MAX_TEST_REPAIR_CYCLES,
    OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
)
from codeflash_python.models.models import TestType
from codeflash_python.telemetry.posthog_cf import ph
from codeflash_python.verification.edit_generated_tests import remove_test_functions
from codeflash_python.verification.test_runner import process_generated_test_strings

if TYPE_CHECKING:
    from typing import Any

    from codeflash_core.danom import Result
    from codeflash_python.models.models import CodeOptimizationContext, CoverageData, GeneratedTestsList, TestResults
    from codeflash_python.optimizer_mixins._protocol import FunctionOptimizerProtocol as _Base
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class TestReviewMixin(_Base):
    def review_and_repair_tests(
        self,
        generated_tests: GeneratedTestsList,
        code_context: CodeOptimizationContext,
        original_helper_code: dict[Path, str],
    ) -> Result[tuple[GeneratedTestsList, TestResults | None, CoverageData | None], str]:
        """Run behavioral tests, review quality per-function, repair flagged functions.

        Flow (up to MAX_TEST_REPAIR_CYCLES):
          behavioral -> collect failures -> AI review passing functions -> repair flagged -> loop
        No benchmarking runs here -- only behavioral validation.

        Returns (generated_tests, behavioral_results, coverage) where behavioral/coverage are
        non-None when the last cycle passed with no repairs (results can be reused by baseline).
        """
        file_path_to_helper_classes = self.build_helper_classes_map(code_context)
        behavioral_results: TestResults | None = None
        coverage_results: CoverageData | None = None
        previous_repair_errors: dict[int, dict[str, str]] = {}
        # Apply token limit to function source (same progressive fallback as optimization/testgen context)
        function_source_for_prompt = self.function_to_optimize_source_code
        if encoded_tokens_len(function_source_for_prompt) > OPTIMIZATION_CONTEXT_TOKEN_LIMIT:
            logger.debug("Function source exceeds token limit for review, extracting function only")
            func = self.function_to_optimize
            source_lines = self.function_to_optimize_source_code.splitlines(keepends=True)
            func_start = (func.doc_start_line or func.starting_line or 1) - 1
            func_end = func.ending_line or len(source_lines)
            function_source_for_prompt = "".join(source_lines[func_start:func_end])
        max_cycles = getattr(self.args, "testgen_review_turns", None) or MAX_TEST_REPAIR_CYCLES
        for cycle in range(max_cycles):
            validation = self.run_behavioral_validation(code_context, original_helper_code, file_path_to_helper_classes)
            if validation is None:
                return Err("Generated tests failed behavioral validation.")
            behavioral_results, coverage_results = validation

            failed_by_file: dict[Path, list[str]] = defaultdict(list)
            for result in behavioral_results.test_results:
                if (
                    result.test_type == TestType.GENERATED_REGRESSION
                    and not result.did_pass
                    and result.id.test_function_name
                ):
                    failed_by_file[result.file_name].append(result.id.test_fn_qualified_name())

            test_failure_messages = behavioral_results.test_failures or {}

            tests_for_review = []
            for i, gt in enumerate(generated_tests.generated_tests):
                failed_fns = failed_by_file.get(gt.behavior_file_path, [])
                failure_details = {fn: test_failure_messages[fn] for fn in failed_fns if fn in test_failure_messages}
                tests_for_review.append(
                    {
                        "test_source": gt.raw_generated_test_source or gt.generated_original_test_source,
                        "test_index": i,
                        "failed_test_functions": failed_fns,
                        "failure_messages": failure_details,
                    }
                )

            coverage_summary = ""
            coverage_details: dict[str, Any] | None = None
            if coverage_results and coverage_results.coverage is not None:
                coverage_summary = f"{coverage_results.coverage:.1f}%"
                mc = coverage_results.main_func_coverage
                coverage_details = {
                    "coverage_percentage": coverage_results.coverage,
                    "threshold_percentage": COVERAGE_THRESHOLD,
                    "function_start_line": self.function_to_optimize.starting_line,
                    "main_function": {
                        "name": mc.name,
                        "coverage": mc.coverage,
                        "executed_lines": sorted(mc.executed_lines),
                        "unexecuted_lines": sorted(mc.unexecuted_lines),
                        "executed_branches": mc.executed_branches,
                        "unexecuted_branches": mc.unexecuted_branches,
                    },
                }
                dc = coverage_results.dependent_func_coverage
                if dc:
                    coverage_details["dependent_function"] = {
                        "name": dc.name,
                        "coverage": dc.coverage,
                        "executed_lines": sorted(dc.executed_lines),
                        "unexecuted_lines": sorted(dc.unexecuted_lines),
                        "executed_branches": dc.executed_branches,
                        "unexecuted_branches": dc.unexecuted_branches,
                    }

            assert self.aiservice_client is not None
            review_results = self.aiservice_client.review_generated_tests(
                tests=tests_for_review,
                function_source_code=function_source_for_prompt,
                function_name=self.function_to_optimize.function_name,
                trace_id=self.function_trace_id,
                coverage_summary=coverage_summary,
                coverage_details=coverage_details,
                language=self.function_to_optimize.language,
            )

            all_to_repair = [r for r in review_results if r.functions_to_repair]

            if not all_to_repair:
                return Ok((generated_tests, behavioral_results, coverage_results))

            total_issues = 0
            for review in all_to_repair:
                for _f in review.functions_to_repair:
                    total_issues += 1

            any_repaired = False
            repaired_files = 0
            # Snapshot all sources before repair so we can show diffs and revert on failure
            original_sources: dict[int, str] = {
                r.test_index: generated_tests.generated_tests[r.test_index].generated_original_test_source
                for r in all_to_repair
            }
            pre_repair_snapshots: dict[int, tuple[str, str, str, str | None]] = {
                r.test_index: (
                    generated_tests.generated_tests[r.test_index].generated_original_test_source,
                    generated_tests.generated_tests[r.test_index].instrumented_behavior_test_source,
                    generated_tests.generated_tests[r.test_index].instrumented_perf_test_source,
                    generated_tests.generated_tests[r.test_index].raw_generated_test_source,
                )
                for r in all_to_repair
            }
            repaired_indices: set[int] = set()
            for review in all_to_repair:
                gt = generated_tests.generated_tests[review.test_index]
                ph(
                    "cli-testgen-repair",
                    {
                        "test_index": review.test_index,
                        "cycle": cycle + 1,
                        "functions": [f.function_name for f in review.functions_to_repair],
                    },
                )

                test_module_path = Path(
                    module_name_from_file_path(gt.behavior_file_path, self.test_cfg.tests_project_rootdir)
                )
                assert self.aiservice_client is not None
                repair_result = self.aiservice_client.repair_generated_tests(
                    test_source=gt.generated_original_test_source,
                    functions_to_repair=review.functions_to_repair,
                    function_source_code=function_source_for_prompt,
                    module_source_code=self.function_to_optimize_source_code,
                    function_to_optimize=self.function_to_optimize,
                    helper_function_names=[f.fully_qualified_name for f in code_context.helper_functions],
                    module_path=Path(self.original_module_path),
                    test_module_path=test_module_path,
                    test_framework=self.test_cfg.test_framework,
                    test_timeout=INDIVIDUAL_TESTCASE_TIMEOUT,
                    trace_id=self.function_trace_id,
                    language=self.function_to_optimize.language,
                    coverage_details=coverage_details,
                    previous_repair_errors=previous_repair_errors.get(review.test_index),
                )

                if repair_result is None:
                    logger.debug("Repair failed for test %s, keeping original", review.test_index)
                    continue

                repaired_source, behavior_source, perf_source = repair_result
                raw_repaired_source = repaired_source
                repaired_source, behavior_source, perf_source = process_generated_test_strings(
                    generated_test_source=repaired_source,
                    instrumented_behavior_test_source=behavior_source,
                    instrumented_perf_test_source=perf_source,
                    function_to_optimize=self.function_to_optimize,
                    test_path=gt.behavior_file_path,
                    test_cfg=self.test_cfg,
                    project_module_system=None,
                )

                gt.generated_original_test_source = repaired_source
                gt.instrumented_behavior_test_source = behavior_source
                gt.instrumented_perf_test_source = perf_source
                gt.raw_generated_test_source = raw_repaired_source

                gt.behavior_file_path.write_text(behavior_source, encoding="utf8")
                gt.perf_file_path.write_text(perf_source, encoding="utf8")
                any_repaired = True
                repaired_files += 1
                repaired_indices.add(review.test_index)

            if not any_repaired:
                logger.warning("All repair API calls failed; proceeding with unrepaired tests")
                break

            validation = self.run_behavioral_validation(code_context, original_helper_code, file_path_to_helper_classes)
            if validation is None:
                for idx in repaired_indices:
                    gt = generated_tests.generated_tests[idx]
                    orig_source, orig_behavior, orig_perf, orig_raw = pre_repair_snapshots[idx]
                    gt.generated_original_test_source = orig_source
                    gt.instrumented_behavior_test_source = orig_behavior
                    gt.instrumented_perf_test_source = orig_perf
                    gt.raw_generated_test_source = orig_raw
                    gt.behavior_file_path.write_text(orig_behavior, encoding="utf8")
                    gt.perf_file_path.write_text(orig_perf, encoding="utf8")
                return Err("Repaired tests failed behavioral validation.")
            behavioral_results, coverage_results = validation

            # Collect failing and all test function names per file
            still_failing_by_file: dict[Path, set[str]] = defaultdict(set)
            all_fns_by_file: dict[Path, set[str]] = defaultdict(set)
            for result in behavioral_results.test_results:
                if result.test_type == TestType.GENERATED_REGRESSION and result.id.test_function_name:
                    fn_name = result.id.test_fn_qualified_name()
                    all_fns_by_file[result.file_name].add(fn_name)
                    if not result.did_pass:
                        still_failing_by_file[result.file_name].add(fn_name)

            reverted_indices = set()
            partially_fixed_indices = set()
            removed_fns_by_index: dict[int, set[str]] = {}
            for idx in repaired_indices:
                gt = generated_tests.generated_tests[idx]
                failing_fns = still_failing_by_file.get(gt.behavior_file_path)
                if not failing_fns:
                    continue

                all_fns_in_file = all_fns_by_file.get(gt.behavior_file_path, set())
                if failing_fns >= all_fns_in_file and all_fns_in_file:
                    # ALL functions fail -> full revert to pre-repair state
                    orig_source, orig_behavior, orig_perf, orig_raw = pre_repair_snapshots[idx]
                    gt.generated_original_test_source = orig_source
                    gt.instrumented_behavior_test_source = orig_behavior
                    gt.instrumented_perf_test_source = orig_perf
                    gt.raw_generated_test_source = orig_raw
                    gt.behavior_file_path.write_text(orig_behavior, encoding="utf8")
                    gt.perf_file_path.write_text(orig_perf, encoding="utf8")
                    reverted_indices.add(idx)
                else:
                    # Partial failure -> remove only failing functions, keep passing ones
                    fns_to_remove = list(failing_fns)
                    removed_fns_by_index[idx] = set(fns_to_remove)
                    gt.generated_original_test_source = remove_test_functions(
                        gt.generated_original_test_source, fns_to_remove
                    )
                    gt.instrumented_behavior_test_source = remove_test_functions(
                        gt.instrumented_behavior_test_source, fns_to_remove
                    )
                    gt.instrumented_perf_test_source = remove_test_functions(
                        gt.instrumented_perf_test_source, fns_to_remove
                    )
                    if gt.raw_generated_test_source is not None:
                        gt.raw_generated_test_source = remove_test_functions(
                            gt.raw_generated_test_source, fns_to_remove
                        )
                    gt.behavior_file_path.write_text(gt.instrumented_behavior_test_source, encoding="utf8")
                    gt.perf_file_path.write_text(gt.instrumented_perf_test_source, encoding="utf8")
                    partially_fixed_indices.add(idx)

            # Show diffs only for repairs that survived re-validation
            successful_repairs = [r for r in all_to_repair if r.test_index not in reverted_indices]
            if successful_repairs:
                self.display_repaired_functions(generated_tests, successful_repairs, original_sources)

            modified_indices = reverted_indices | partially_fixed_indices
            if modified_indices:
                messages = []
                if reverted_indices:
                    messages.append(f"reverted {len(reverted_indices)} test file(s)")
                if partially_fixed_indices:
                    messages.append(f"removed failing functions from {len(partially_fixed_indices)} test file(s)")
                # Collect error messages from failed functions so the next cycle can learn
                revalidation_failures = behavioral_results.test_failures or {}
                for idx in modified_indices:
                    gt = generated_tests.generated_tests[idx]
                    removed_fns = removed_fns_by_index.get(idx, set())
                    errors_for_file: dict[str, str] = {}
                    for result in behavioral_results.test_results:
                        if (
                            result.file_name == gt.behavior_file_path
                            and result.test_type == TestType.GENERATED_REGRESSION
                            and not result.did_pass
                            and result.id.test_function_name
                        ):
                            fn_name = result.id.test_fn_qualified_name()
                            if fn_name not in removed_fns:
                                errors_for_file[fn_name] = revalidation_failures.get(fn_name, "Test failed")
                    if errors_for_file:
                        previous_repair_errors[idx] = errors_for_file
                # Invalidate behavioral results since files were modified
                behavioral_results = None
                coverage_results = None

        return Ok((generated_tests, behavioral_results, coverage_results))
