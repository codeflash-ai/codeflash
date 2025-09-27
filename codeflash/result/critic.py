from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    MIN_IMPROVEMENT_THRESHOLD,
    MIN_TESTCASE_PASSED_THRESHOLD,
    MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD,
)
from codeflash.models import models

if TYPE_CHECKING:
    from codeflash.models.models import CoverageData, OptimizedCandidateResult, OriginalCodeBaseline


def performance_gain(*, original_runtime_ns: int, optimized_runtime_ns: int) -> float:
    """Calculate the performance gain of an optimized code over the original code.

    This value multiplied by 100 gives the percentage improvement in runtime.
    """
    if optimized_runtime_ns == 0:
        return 0.0
    return (original_runtime_ns - optimized_runtime_ns) / optimized_runtime_ns


def throughput_gain(*, original_throughput: int, optimized_throughput: int) -> float:
    """Calculate the throughput gain of an optimized code over the original code.

    This value multiplied by 100 gives the percentage improvement in throughput.
    For throughput, higher values are better (more executions per time period).
    """
    if original_throughput == 0:
        return 0.0
    return (optimized_throughput - original_throughput) / original_throughput


def speedup_critic(
    candidate_result: OptimizedCandidateResult,
    original_code_runtime: int,
    best_runtime_until_now: int | None,
    *,
    disable_gh_action_noise: bool = False,
    original_async_throughput: int | None = None,
    best_throughput_until_now: int | None = None,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    Evaluates both runtime performance and async throughput improvements.

    For runtime performance:
    - Ensures the optimization is actually faster than the original code, above the noise floor.
    - The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
      when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    - The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance.

    For async throughput (when available):
    - Evaluates throughput improvements using MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD
    - Throughput improvements complement runtime improvements for async functions
    """
    # Runtime performance evaluation
    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_code_runtime < 10000 else MIN_IMPROVEMENT_THRESHOLD
    if not disable_gh_action_noise and env_utils.is_ci():
        noise_floor = noise_floor * 2  # Increase the noise floor in GitHub Actions mode

    perf_gain = performance_gain(
        original_runtime_ns=original_code_runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )
    runtime_improved = perf_gain > noise_floor

    # Check runtime comparison with best so far
    runtime_is_best = best_runtime_until_now is None or candidate_result.best_test_runtime < best_runtime_until_now

    throughput_improved = True  # Default to True if no throughput data
    throughput_is_best = True  # Default to True if no throughput data

    if original_async_throughput is not None and candidate_result.async_throughput is not None:
        if original_async_throughput > 0:
            throughput_gain_value = throughput_gain(
                original_throughput=original_async_throughput, optimized_throughput=candidate_result.async_throughput
            )
            throughput_improved = throughput_gain_value > MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD

        throughput_is_best = (
            best_throughput_until_now is None or candidate_result.async_throughput > best_throughput_until_now
        )

    if original_async_throughput is not None and candidate_result.async_throughput is not None:
        # When throughput data is available, accept if EITHER throughput OR runtime improves significantly
        throughput_acceptance = throughput_improved and throughput_is_best
        runtime_acceptance = runtime_improved and runtime_is_best
        return throughput_acceptance or runtime_acceptance
    return runtime_improved and runtime_is_best


def quantity_of_tests_critic(candidate_result: OptimizedCandidateResult | OriginalCodeBaseline) -> bool:
    test_results = candidate_result.behavior_test_results
    report = test_results.get_test_pass_fail_report_by_type()

    pass_count = 0
    for test_type in report:
        pass_count += report[test_type]["passed"]

    if pass_count >= MIN_TESTCASE_PASSED_THRESHOLD:
        return True
    # If one or more tests passed, check if least one of them was a successful REPLAY_TEST
    return bool(pass_count >= 1 and report[models.TestType.REPLAY_TEST]["passed"] >= 1)  # type: ignore  # noqa: PGH003


def coverage_critic(original_code_coverage: CoverageData | None, test_framework: str) -> bool:
    """Check if the coverage meets the threshold."""
    if test_framework == "unittest":
        logger.debug("Coverage critic is not implemented for unittest yet.")
        return True
    if original_code_coverage:
        return original_code_coverage.coverage >= COVERAGE_THRESHOLD
    return False
