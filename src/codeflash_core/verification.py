from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash_core.models import TestOutcome, TestResults

# Type alias for a custom output comparator function.
OutputComparator = Callable[[object, object], bool]


def is_equivalent(baseline: TestResults, candidate: TestResults, comparator: OutputComparator | None = None) -> bool:
    """Check if candidate test results are equivalent to baseline.

    Both must pass, have the same number of outcomes, and each outcome
    must match on test_id, status, and output.

    If *comparator* is provided it is used instead of ``==`` to compare
    captured test outputs.
    """
    if not baseline.passed or not candidate.passed:
        return False

    if len(baseline.outcomes) != len(candidate.outcomes):
        return False

    baseline_by_id = {o.test_id: o for o in baseline.outcomes}
    candidate_by_id = {o.test_id: o for o in candidate.outcomes}

    if baseline_by_id.keys() != candidate_by_id.keys():
        return False

    return all(
        outcomes_match(baseline_by_id[test_id], candidate_by_id[test_id], comparator=comparator)
        for test_id in baseline_by_id
    )


def outcomes_match(baseline: TestOutcome, candidate: TestOutcome, comparator: OutputComparator | None = None) -> bool:
    if baseline.status != candidate.status:
        return False
    if baseline.output is not None and candidate.output is not None:
        return compare_outputs(baseline.output, candidate.output, comparator=comparator)
    return True


def compare_outputs(
    baseline_output: object, candidate_output: object, comparator: OutputComparator | None = None
) -> bool:
    """Compare return-value outputs using *comparator*, falling back to ``==``."""
    if comparator is not None:
        return comparator(baseline_output, candidate_output)
    return baseline_output == candidate_output
