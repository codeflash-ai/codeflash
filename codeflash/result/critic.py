from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    MIN_IMPROVEMENT_THRESHOLD,
    MIN_TESTCASE_PASSED_THRESHOLD,
)
from codeflash.code_utils.env_utils import get_pr_number as _gh_get_pr_number
from codeflash.models.models import OptimizedCandidateResult, TestType

if TYPE_CHECKING:
    from codeflash.models.models import CoverageData, OptimizedCandidateResult, OriginalCodeBaseline


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
    disable_gh_action_noise: Optional[bool] = None,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    Ensure that the optimization is actually faster than the original code, above the noise floor.
    The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
    when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance, also we want to be more confident there.
    """
    # Set initial noise floor based on the runtime
    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_code_runtime < 10000 else MIN_IMPROVEMENT_THRESHOLD

    # Increase noise floor if running in GitHub Actions unless disabled
    if not disable_gh_action_noise and _gh_get_pr_number() is not None:
        noise_floor = noise_floor * 2

    perf_gain = performance_gain(
        original_runtime_ns=original_code_runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )

    # No previous best runtime, just check if above noise floor
    if best_runtime_until_now is None:
        return perf_gain > noise_floor

    # Must both be above noise floor *and* faster than previous best
    return perf_gain > noise_floor and candidate_result.best_test_runtime < best_runtime_until_now


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
