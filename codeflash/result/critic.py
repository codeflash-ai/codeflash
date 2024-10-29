from __future__ import annotations

from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import MIN_IMPROVEMENT_THRESHOLD
from codeflash.models.models import OptimizedCandidateResult
from codeflash.verification.test_results import TestType


def performance_gain(*, original_runtime_ns: int, optimized_runtime_ns: int) -> float:
    """Calculate the performance gain of an optimized code over the original code. This value multiplied by 100
    gives the percentage improvement in runtime.
    """
    return (original_runtime_ns - optimized_runtime_ns) / optimized_runtime_ns


def speedup_critic(
    candidate_result: OptimizedCandidateResult, original_code_runtime: int, best_runtime_until_now: int
) -> bool:
    """Takes in a correct optimized Test Result and decides if the optimization should actually
    be surfaced to the user.
    Ensures that the optimization is actually faster than the original code, above the noise floor.
    The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
    when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance, also we want to be more confident there.
    """
    in_github_actions_mode = bool(env_utils.get_pr_number())
    if original_code_runtime < 10_000:
        noise_floor = 2 * MIN_IMPROVEMENT_THRESHOLD
    else:
        noise_floor = MIN_IMPROVEMENT_THRESHOLD
    if in_github_actions_mode:
        noise_floor = noise_floor * 2  # Increase the noise floor in GitHub Actions mode

    perf_gain = performance_gain(
        original_runtime_ns=original_code_runtime, optimized_runtime_ns=candidate_result.best_test_runtime
    )
    if (perf_gain > noise_floor) and candidate_result.best_test_runtime < best_runtime_until_now:
        return True
    return False


def quantity_of_tests_critic(candidate_result: OptimizedCandidateResult) -> bool:
    test_results = candidate_result.best_test_results
    in_github_actions_mode = bool(env_utils.get_pr_number())

    report = test_results.get_test_pass_fail_report_by_type()

    pass_count = 0
    for test_type in report:
        pass_count += report[test_type]["passed"]

    if in_github_actions_mode:
        if pass_count >= 4:
            return True
    elif pass_count >= 2:
        return True
    # If only one test passed, check if it's a REPLAY_TEST
    if pass_count == 1 and report[TestType.REPLAY_TEST]["passed"] == 1:
        return True

    return False
