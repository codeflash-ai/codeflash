from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    MIN_IMPROVEMENT_THRESHOLD,
    MIN_TESTCASE_PASSED_THRESHOLD,
)
from codeflash.models.models import OptimizedCandidateResult, OriginalCodeBaseline, TestType

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
    best_runtime_until_now: int,
    disable_gh_action_noise: Optional[bool] = None,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    Ensure that the optimization is actually faster than the original code, above the noise floor.
    The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
    when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance, also we want to be more confident there.
    """
    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_code_runtime < 10000 else MIN_IMPROVEMENT_THRESHOLD
    if not disable_gh_action_noise:
        in_github_actions_mode = bool(env_utils.get_pr_number())
        if in_github_actions_mode:
            noise_floor = noise_floor * 2  # Increase the noise floor in GitHub Actions mode

    perf_gain = performance_gain(
        original_runtime_ns=original_code_runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )
    return bool(perf_gain > noise_floor and candidate_result.best_test_runtime < best_runtime_until_now)


def quantity_of_tests_critic(candidate_result: OptimizedCandidateResult | OriginalCodeBaseline) -> bool:
    test_results = candidate_result.behavior_test_results
    # Fast path: compute report, pass_count and REPLAY_TEST in a single loop
    report = {}
    pass_count = 0
    replay_pass = 0
    for test_result in test_results.test_results:
        if test_result.loop_index == 1:
            ttype = test_result.test_type
            if ttype not in report:
                report[ttype] = {"passed": 0, "failed": 0}
            if test_result.did_pass:
                report[ttype]["passed"] += 1
                pass_count += 1
                if ttype == TestType.REPLAY_TEST:
                    replay_pass += 1
            else:
                report[ttype]["failed"] += 1

    if pass_count >= MIN_TESTCASE_PASSED_THRESHOLD:
        return True
    # If at least one test passed, check if at least one was a successful REPLAY_TEST
    return bool(pass_count >= 1 and replay_pass >= 1)


def coverage_critic(original_code_coverage: CoverageData | None, test_framework: str) -> bool:
    """Check if the coverage meets the threshold."""
    if test_framework == "unittest":
        logger.debug("Coverage critic is not implemented for unittest yet.")
        return True
    if original_code_coverage:
        return original_code_coverage.coverage >= COVERAGE_THRESHOLD
    return False
