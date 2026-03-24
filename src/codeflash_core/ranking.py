from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash_core.models import BenchmarkResults, ScoredCandidate


def compute_speedup(baseline: BenchmarkResults, candidate: BenchmarkResults) -> float:
    """Compute speedup as percentage improvement: (baseline - candidate) / candidate.

    Matches original codeflash performance_gain formula.
    Returns 0.0 if candidate time is zero (no improvement measurable).
    A positive value means the candidate is faster (e.g. 1.0 = 100% faster).
    """
    if candidate.total_time <= 0:
        return 0.0
    return (baseline.total_time - candidate.total_time) / candidate.total_time


def score_candidate(speedup: float) -> float:
    """Score a candidate based on its speedup.

    Currently score == speedup. This is the extension point for adding
    more signals (code complexity, diff size, etc.) in the future.
    """
    return speedup


def select_best(candidates: list[ScoredCandidate]) -> ScoredCandidate | None:
    """Select the best candidate by score. Returns None if list is empty."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)
