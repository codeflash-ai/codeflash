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
from codeflash.models.test_type import TestType

if TYPE_CHECKING:
    from codeflash.models.models import ConcurrencyMetrics, CoverageData, OptimizedCandidateResult, OriginalCodeBaseline


class AcceptanceReason(Enum):
    RUNTIME = "runtime"
    THROUGHPUT = "throughput"
    CONCURRENCY = "concurrency"
    RENDER_COUNT = "render_count"
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
    original_render_profiles: list | None = None,
    render_count_low_confidence: bool = False,
) -> bool:
    """Take in a correct optimized Test Result and decide if the optimization should actually be surfaced to the user.

    Evaluates runtime performance, async throughput, concurrency improvements, and React render efficiency.

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

    For React render efficiency (when available):
    - Evaluates render count reduction and render duration improvements
    - Accepts if render count reduced by >= 20% or render duration improved significantly
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

    # React render efficiency evaluation
    render_efficiency_improved = False
    if original_render_profiles and candidate_result.render_profiles:
        from codeflash.languages.javascript.frameworks.react.benchmarking import compare_render_benchmarks  # noqa: PLC0415

        benchmark = compare_render_benchmarks(original_render_profiles, candidate_result.render_profiles)
        if benchmark:
            render_efficiency_improved = render_efficiency_critic(
                benchmark.original_render_count,
                benchmark.optimized_render_count,
                benchmark.original_avg_duration_ms,
                benchmark.optimized_avg_duration_ms,
                original_dom_mutations=benchmark.original_dom_mutations,
                optimized_dom_mutations=benchmark.optimized_dom_mutations,
                original_update_render_count=benchmark.original_update_render_count,
                optimized_update_render_count=benchmark.optimized_update_render_count,
                original_update_duration=benchmark.original_update_avg_duration_ms,
                optimized_update_duration=benchmark.optimized_update_avg_duration_ms,
                child_render_reduction=benchmark.child_render_reduction,
                original_interaction_duration_ms=benchmark.original_interaction_duration_ms,
                optimized_interaction_duration_ms=benchmark.optimized_interaction_duration_ms,
                trust_duration=False,
                low_confidence=render_count_low_confidence,
            )

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

    # Accept if ANY of: render efficiency, runtime, throughput, or concurrency improves significantly
    if render_efficiency_improved:
        return True

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
    original_render_count: int | None = None,
    optimized_render_count: int | None = None,
    original_render_duration: float | None = None,
    optimized_render_duration: float | None = None,
    original_dom_mutations: int = 0,
    optimized_dom_mutations: int = 0,
    original_update_render_count: int = 0,
    optimized_update_render_count: int = 0,
    original_update_duration: float = 0.0,
    optimized_update_duration: float = 0.0,
    child_render_reduction: int = 0,
    original_interaction_duration_ms: float = 0.0,
    optimized_interaction_duration_ms: float = 0.0,
) -> AcceptanceReason:
    """Determine why an optimization was accepted.

    Returns the primary reason for acceptance, with priority:
    render_count > concurrency > throughput > runtime.
    """
    noise_floor = 3 * MIN_IMPROVEMENT_THRESHOLD if original_runtime_ns < 10000 else MIN_IMPROVEMENT_THRESHOLD
    if env_utils.is_ci():
        noise_floor = noise_floor * 2

    perf_gain = performance_gain(original_runtime_ns=original_runtime_ns, optimized_runtime_ns=optimized_runtime_ns)
    runtime_improved = perf_gain > noise_floor

    # Check React render efficiency
    render_improved = False
    if (
        original_render_count is not None
        and optimized_render_count is not None
        and original_render_duration is not None
        and optimized_render_duration is not None
    ):
        render_improved = render_efficiency_critic(
            original_render_count,
            optimized_render_count,
            original_render_duration,
            optimized_render_duration,
            original_dom_mutations=original_dom_mutations,
            optimized_dom_mutations=optimized_dom_mutations,
            original_update_render_count=original_update_render_count,
            optimized_update_render_count=optimized_update_render_count,
            original_update_duration=original_update_duration,
            optimized_update_duration=optimized_update_duration,
            child_render_reduction=child_render_reduction,
            original_interaction_duration_ms=original_interaction_duration_ms,
            optimized_interaction_duration_ms=optimized_interaction_duration_ms,
            trust_duration=False,
        )

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

    # Return reason with priority: render_count > concurrency > throughput > runtime
    if render_improved:
        return AcceptanceReason.RENDER_COUNT

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
    return bool(pass_count >= 1 and report[TestType.REPLAY_TEST]["passed"] >= 1)


def coverage_critic(original_code_coverage: CoverageData | None) -> bool:
    """Check if the coverage meets the threshold."""
    if original_code_coverage:
        return original_code_coverage.coverage >= COVERAGE_THRESHOLD
    return False


# Minimum render count reduction percentage to accept a React optimization
MIN_RENDER_COUNT_REDUCTION_PCT = 0.20  # 20%


MIN_DOM_MUTATION_REDUCTION_PCT = 0.20  # 20%


MIN_INTERACTION_DURATION_REDUCTION_PCT = 0.20  # 20%

MIN_CHILD_RENDER_REDUCTION = 2


def render_efficiency_critic(
    original_render_count: int,
    optimized_render_count: int,
    original_render_duration: float,
    optimized_render_duration: float,
    best_render_count_until_now: int | None = None,
    original_dom_mutations: int = 0,
    optimized_dom_mutations: int = 0,
    original_update_render_count: int = 0,
    optimized_update_render_count: int = 0,
    original_update_duration: float = 0.0,
    optimized_update_duration: float = 0.0,
    child_render_reduction: int = 0,
    original_interaction_duration_ms: float = 0.0,
    optimized_interaction_duration_ms: float = 0.0,
    trust_duration: bool = True,
    low_confidence: bool = False,
) -> bool:
    """Evaluate whether a React optimization reduces re-renders, render time, or DOM mutations sufficiently.

    Uses update-phase render counts as primary signal when available (tests that
    trigger interactions produce update-phase markers). Falls back to total
    render count if no update-phase data exists.

    When ``trust_duration`` is False (e.g. jsdom where actualDuration is noise),
    render duration is excluded from the acceptance criteria.

    When ``low_confidence`` is True (render counts varied across validation
    runs), the render count reduction threshold is raised from 20% to 30%
    to reduce false positives from measurement noise.

    Accepts if:
    - Update render count reduced by >= threshold (primary), OR total render count reduced by >= threshold (fallback)
    - OR render duration reduced by >= MIN_IMPROVEMENT_THRESHOLD (when trust_duration=True)
    - OR DOM mutations reduced by >= 20%
    - OR child component render reduction >= MIN_CHILD_RENDER_REDUCTION (captures useCallback/memo optimizations)
    - OR interaction duration reduced by >= 20% (captures debounce/throttle optimizations)
    - AND the candidate is the best seen so far
    """
    if original_render_count == 0 and original_dom_mutations == 0 and child_render_reduction == 0:
        return False

    # Use update-phase counts as primary signal when available.
    # When the ONLY signal is mount-phase render count (no update-phase data, no DOM mutations,
    # no child reduction, no interaction data), we cannot meaningfully evaluate the optimization.
    # Mount count reductions are not a valid React optimization signal — memoization optimizations
    # often *increase* mount cost while reducing update-phase renders.
    # When update-phase data exists, ONLY use it for render count acceptance —
    # total count (which includes mount) dilutes the signal.
    has_update_data = original_update_render_count > 0 or optimized_update_render_count > 0
    has_dom_signal = original_dom_mutations > 0
    has_child_signal = child_render_reduction > 0
    has_interaction_signal = original_interaction_duration_ms > 0

    if not has_update_data and not has_dom_signal and not has_child_signal and not has_interaction_signal:
        return False

    # Check render count reduction (higher threshold when confidence is low)
    render_count_threshold = 0.30 if low_confidence else MIN_RENDER_COUNT_REDUCTION_PCT
    count_improved = False
    if has_update_data:
        # Primary: update-phase only — do NOT fall through to total count
        if original_update_render_count > 0:
            count_reduction = (
                (original_update_render_count - optimized_update_render_count) / original_update_render_count
            )
            count_improved = count_reduction >= render_count_threshold
    elif original_render_count > 0:
        # Fallback: total count when zero update-phase data exists
        count_reduction = (original_render_count - optimized_render_count) / original_render_count
        count_improved = count_reduction >= render_count_threshold

    # Determine effective counts for best-candidate tracking
    effective_opt_count = optimized_update_render_count if has_update_data else optimized_render_count

    # Check render duration reduction (prefer update-phase duration)
    # Skipped when trust_duration=False (jsdom actualDuration is noise)
    duration_improved = False
    if trust_duration:
        effective_orig_duration = original_update_duration if has_update_data else original_render_duration
        effective_opt_duration = optimized_update_duration if has_update_data else optimized_render_duration
        if effective_orig_duration > 0:
            duration_gain = (effective_orig_duration - effective_opt_duration) / effective_orig_duration
            duration_improved = duration_gain > MIN_IMPROVEMENT_THRESHOLD

    # Check DOM mutation reduction
    dom_mutations_improved = False
    if original_dom_mutations > 0:
        dom_reduction = (original_dom_mutations - optimized_dom_mutations) / original_dom_mutations
        dom_mutations_improved = dom_reduction >= MIN_DOM_MUTATION_REDUCTION_PCT

    # Check child render reduction (useCallback/memo optimization signal)
    child_renders_improved = child_render_reduction >= MIN_CHILD_RENDER_REDUCTION

    # Check interaction duration reduction (debounce/throttle optimization signal)
    interaction_duration_improved = False
    if original_interaction_duration_ms > 0:
        interaction_reduction = (
            (original_interaction_duration_ms - optimized_interaction_duration_ms) / original_interaction_duration_ms
        )
        interaction_duration_improved = interaction_reduction >= MIN_INTERACTION_DURATION_REDUCTION_PCT

    # Check if this is the best candidate so far
    is_best = best_render_count_until_now is None or effective_opt_count <= best_render_count_until_now

    return (
        count_improved
        or duration_improved
        or dom_mutations_improved
        or child_renders_improved
        or interaction_duration_improved
    ) and is_best
