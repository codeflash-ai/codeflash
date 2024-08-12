import logging

from codeflash.code_utils.config_consts import MIN_IMPROVEMENT_THRESHOLD
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.models import OptimizedCandidateResult


def speedup_critic(
    candidate_result: OptimizedCandidateResult, original_code_runtime: int, best_runtime_until_now: int
) -> bool:
    """Takes in a correct optimized Test Result and decides if the optimization should actually
    be surfaced to the user.
    Ensures that the optimization is actually faster than the original code, above the noise floor.
    The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
    when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    """
    if original_code_runtime < 10_000:
        noise_floor = 2 * MIN_IMPROVEMENT_THRESHOLD
    else:
        noise_floor = MIN_IMPROVEMENT_THRESHOLD
    if (
        ((original_code_runtime - candidate_result.best_test_runtime) / candidate_result.best_test_runtime)
        > noise_floor
    ) and candidate_result.best_test_runtime < best_runtime_until_now:
        logging.info(
            "This candidate is better than the previous best candidate.",
        )

        logging.info(
            f"Original runtime: {humanize_runtime(original_code_runtime)} Best test runtime: "
            f"{humanize_runtime(candidate_result.best_test_runtime)}, ratio = "
            f"{((original_code_runtime - candidate_result.best_test_runtime) / candidate_result.best_test_runtime)}",
        )
        return True
    else:
        return False


def generated_test_critic(candidate_result: OptimizedCandidateResult) -> bool:
    test_results = candidate_result.best_test_results.test_results
    count = len(
        [
            test_result
            for test_result in test_results
            if test_result.test_type.name == "GENERATED_REGRESSION"
            and test_result.did_pass
        ]
    )
    return count > 1
