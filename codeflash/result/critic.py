from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from codeflash.code_utils import env_utils
from codeflash.code_utils.config_consts import (
    COVERAGE_THRESHOLD,
    MIN_CONCURRENCY_IMPROVEMENT_THRESHOLD,
    MIN_IMPROVEMENT_THRESHOLD,
    MIN_TESTCASE_PASSED_THRESHOLD,
    MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD,
)
from codeflash.models import models

if TYPE_CHECKING:
    from codeflash.models.models import ConcurrencyMetrics, CoverageData, OptimizedCandidateResult, OriginalCodeBaseline


class AcceptanceReason(Enum):
    RUNTIME = "runtime"
    THROUGHPUT = "throughput"
    CONCURRENCY = "concurrency"
    NONE = "none"


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


def concurrency_gain(original_metrics: ConcurrencyMetrics, optimized_metrics: ConcurrencyMetrics) -> float:
    """Calculate concurrency ratio improvement.

    Returns the relative improvement in concurrency ratio.
    Higher is better - means the optimized code scales better with concurrent execution.

    concurrency_ratio = sequential_time / concurrent_time
    A ratio of 10 means concurrent execution is 10x faster than sequential.
    """
    if original_metrics.concurrency_ratio == 0:
        return 0.0
    return (
        optimized_metrics.concurrency_ratio - original_metrics.concurrency_ratio
    ) / original_metrics.concurrency_ratio


def speedup_critic(
    candidate_result: OptimizedCandidateResult,
    original_code_runtime: int,
    best_runtime_until_now: int | None,
    *,
    disable_gh_action_noise: bool = False,
    original_async_throughput: int | None = None,
    best_throughput_until_now: int | None = None,
    original_concurrency_metrics: ConcurrencyMetrics | None = None,
    best_concurrency_ratio_until_now: float | None = None,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    Evaluates runtime performance, async throughput, and concurrency improvements.

    For runtime performance:
    - Ensures the optimization is actually faster than the original code, above the noise floor.
    - The noise floor is a function of the original code runtime. Currently, the noise floor is 2xMIN_IMPROVEMENT_THRESHOLD
      when the original runtime is less than 10 microseconds, and becomes MIN_IMPROVEMENT_THRESHOLD for any higher runtime.
    - The noise floor is doubled when benchmarking on a (noisy) GitHub Action virtual instance.

    For async throughput (when available):
    - Evaluates throughput improvements using MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD
    - Throughput improvements complement runtime improvements for async functions

    For concurrency (when available):
    - Evaluates concurrency ratio improvements using MIN_CONCURRENCY_IMPROVEMENT_THRESHOLD
    - Concurrency improvements detect when blocking calls are replaced with non-blocking equivalents
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

    # Concurrency evaluation
    concurrency_improved = False
    concurrency_is_best = True
    if original_concurrency_metrics is not None and candidate_result.concurrency_metrics is not None:
        conc_gain = concurrency_gain(original_concurrency_metrics, candidate_result.concurrency_metrics)
        concurrency_improved = conc_gain > MIN_CONCURRENCY_IMPROVEMENT_THRESHOLD
        concurrency_is_best = (
            best_concurrency_ratio_until_now is None
            or candidate_result.concurrency_metrics.concurrency_ratio > best_concurrency_ratio_until_now
        )

    # Accept if ANY of: runtime, throughput, or concurrency improves significantly
    if original_async_throughput is not None and candidate_result.async_throughput is not None:
        throughput_acceptance = throughput_improved and throughput_is_best
        runtime_acceptance = runtime_improved and runtime_is_best
        concurrency_acceptance = concurrency_improved and concurrency_is_best
        return throughput_acceptance or runtime_acceptance or concurrency_acceptance
    return runtime_improved and runtime_is_best


def get_acceptance_reason(
    original_runtime_ns: int,
    optimized_runtime_ns: int,
    *,
    original_async_throughput: int | None = None,
    optimized_async_throughput: int | None = None,
    original_concurrency_metrics: ConcurrencyMetrics | None = None,
    optimized_concurrency_metrics: ConcurrencyMetrics | None = None,
) -> AcceptanceReason:
    """Determine why an optimization was accepted.

    Returns the primary reason for acceptance, with priority:
    concurrency > throughput > runtime (for async code).
    """
    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_runtime_ns < 10000 else MIN_IMPROVEMENT_THRESHOLD
    if env_utils.is_ci():
        noise_floor = noise_floor * 2

    perf_gain = performance_gain(original_runtime_ns=original_runtime_ns, optimized_runtime_ns=optimized_runtime_ns)
    runtime_improved = perf_gain > noise_floor

    throughput_improved = False
    if (
        original_async_throughput is not None
        and optimized_async_throughput is not None
        and original_async_throughput > 0
    ):
        throughput_gain_value = throughput_gain(
            original_throughput=original_async_throughput, optimized_throughput=optimized_async_throughput
        )
        throughput_improved = throughput_gain_value > MIN_THROUGHPUT_IMPROVEMENT_THRESHOLD

    concurrency_improved = False
    if original_concurrency_metrics is not None and optimized_concurrency_metrics is not None:
        conc_gain = concurrency_gain(original_concurrency_metrics, optimized_concurrency_metrics)
        concurrency_improved = conc_gain > MIN_CONCURRENCY_IMPROVEMENT_THRESHOLD

    # Return reason with priority: concurrency > throughput > runtime
    if original_async_throughput is not None and optimized_async_throughput is not None:
        if concurrency_improved:
            return AcceptanceReason.CONCURRENCY
        if throughput_improved:
            return AcceptanceReason.THROUGHPUT
        if runtime_improved:
            return AcceptanceReason.RUNTIME
        return AcceptanceReason.NONE

    if runtime_improved:
        return AcceptanceReason.RUNTIME
    return AcceptanceReason.NONE


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


def coverage_critic(original_code_coverage: CoverageData | None) -> bool:
    """Check if the coverage meets the threshold."""
    if original_code_coverage:
        return original_code_coverage.coverage >= COVERAGE_THRESHOLD
    return False
