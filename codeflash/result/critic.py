from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    MIN_IMPROVEMENT_THRESHOLD,
    MIN_TESTCASE_PASSED_THRESHOLD,
)
from codeflash.models.models import TestType

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CoverageData, OptimizedCandidateResult, OriginalCodeBaseline, TestResults


def performance_gain(*, original_runtime_ns: int, optimized_runtime_ns: int) -> float:
    """Calculate the performance gain of an optimized code over the original code.

    This value multiplied by 100 gives the percentage improvement in runtime.
    """
    if optimized_runtime_ns == 0:
        return 0.0
    return (original_runtime_ns - optimized_runtime_ns) / optimized_runtime_ns


def speedup_critic(
    candidate_result: OptimizedCandidateResult,
    original_code_runtime: int,
    best_runtime_until_now: int | None,
    function_to_optimize: FunctionToOptimize,
    *,
    disable_gh_action_noise: bool = False,
    original_baseline_results: OriginalCodeBaseline | None = None,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    For async functions, dispatches to async_speedup_critic for specialized evaluation.
    For sync functions, uses traditional runtime-only evaluation.

    Ensure that the optimization is actually faster than the original code, above the noise floor.
    The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
    when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance, also we want to be more confident there.
    """
    if function_to_optimize.is_async and original_baseline_results:
        return async_speedup_critic(
            candidate_result=candidate_result,
            original_baseline_results=original_baseline_results,
            best_runtime_until_now=best_runtime_until_now,
            disable_gh_action_noise=disable_gh_action_noise,
        )

    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_code_runtime < 10000 else MIN_IMPROVEMENT_THRESHOLD
    if not disable_gh_action_noise and env_utils.is_ci():
        noise_floor = noise_floor * 2  # Increase the noise floor in GitHub Actions mode

    perf_gain = performance_gain(
        original_runtime_ns=original_code_runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )
    if best_runtime_until_now is None:
        return bool(perf_gain > noise_floor)
    return bool(perf_gain > noise_floor and candidate_result.best_test_runtime < best_runtime_until_now)


def async_speedup_critic(
    candidate_result: OptimizedCandidateResult,
    original_baseline_results: OriginalCodeBaseline,
    best_runtime_until_now: int | None,
    *,
    disable_gh_action_noise: bool = False,
) -> bool:
    """Simplified speedup evaluation for async functions with throughput-first approach.

    For async functions:
    1. If throughput data exists and shows improvement, accept the optimization
    2. Otherwise, fall back to traditional runtime evaluation
    """
    # Calculate noise floor with same logic as sync functions
    noise_floor = (
        3 * MIN_IMPROVEMENT_THRESHOLD if original_baseline_results.runtime < 10000 else MIN_IMPROVEMENT_THRESHOLD
    )
    if not disable_gh_action_noise and env_utils.is_ci():
        noise_floor = noise_floor * 2  # Increase the noise floor in GitHub Actions mode

    # Check for throughput improvement first
    candidate_throughput = _calculate_average_throughput(candidate_result.benchmarking_test_results)
    original_throughput = _calculate_average_throughput(original_baseline_results.benchmarking_test_results)

    if original_throughput and original_throughput > 0 and candidate_throughput:
        throughput_gain = (candidate_throughput - original_throughput) / original_throughput
        if throughput_gain > noise_floor:
            # Throughput improved above noise floor - accept optimization
            return (
                True if best_runtime_until_now is None else candidate_result.best_test_runtime < best_runtime_until_now
            )

    # Fall back to traditional runtime evaluation
    perf_gain = performance_gain(
        original_runtime_ns=original_baseline_results.runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )

    if best_runtime_until_now is None:
        return bool(perf_gain > noise_floor)
    return bool(perf_gain > noise_floor and candidate_result.best_test_runtime < best_runtime_until_now)


def _calculate_average_throughput(test_results: TestResults) -> float | None:
    """Calculate average throughput from test results that have throughput data."""
    throughput_values = [
        result.throughput for result in test_results.test_results if result.throughput is not None and result.did_pass
    ]

    if not throughput_values:
        return None

    return sum(throughput_values) / len(throughput_values)


def quantity_of_tests_critic(candidate_result: OptimizedCandidateResult | OriginalCodeBaseline) -> bool:
    test_results = candidate_result.behavior_test_results
    report = test_results.get_test_pass_fail_report_by_type()

    pass_count = 0
    for test_type in report:
        pass_count += report[test_type]["passed"]

    if pass_count >= MIN_TESTCASE_PASSED_THRESHOLD:
        return True
    # If one or more tests passed, check if least one of them was a successful REPLAY_TEST
    return bool(pass_count >= 1 and report[TestType.REPLAY_TEST]["passed"] >= 1)


def coverage_critic(original_code_coverage: CoverageData | None, test_framework: str) -> bool:
    """Check if the coverage meets the threshold."""
    if test_framework == "unittest":
        logger.debug("Coverage critic is not implemented for unittest yet.")
        return True
    if original_code_coverage:
        return original_code_coverage.coverage >= COVERAGE_THRESHOLD
    return False
