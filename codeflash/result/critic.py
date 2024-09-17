from __future__ import annotations

from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import MIN_IMPROVEMENT_THRESHOLD
from codeflash.models.models import OptimizedCandidateResult


def performance_gain(*, original_runtime_ns: int, optimized_runtime_ns: int) -> float:
    """Calculate the performance gain of an optimized code over the original code. This value multiplied by 100
    gives the percentage improvement in runtime.
    """
    return (original_runtime_ns - optimized_runtime_ns) / optimized_runtime_ns


def speedup_critic(
    candidate_result: OptimizedCandidateResult,
    original_code_runtime: int,
    best_runtime_until_now: int,
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
        original_runtime_ns=original_code_runtime,
        optimized_runtime_ns=candidate_result.best_test_runtime,
    )
    if (perf_gain > noise_floor) and candidate_result.best_test_runtime < best_runtime_until_now:
        return True
    return False


def quantity_of_tests_critic(candidate_result: OptimizedCandidateResult) -> bool:
    test_results = candidate_result.best_test_results.test_results
    in_github_actions_mode = bool(env_utils.get_pr_number())

    passed_test = None
    count = 0

    for test_result in test_results:
        if test_result.did_pass:
            count += 1
            if count == 1:
                passed_test = test_result
            if in_github_actions_mode and count >= 4 or count > 1:
                return True

    # If only one test passed, check if it's a REPLAY_TEST
    if count == 1 and passed_test.test_type.name == "REPLAY_TEST":
        return True

    return False
