from __future__ import annotations

import difflib


def choose_weights(**importance: float) -> list[float]:
    """Choose normalized weights from relative importance values.

    Example:
        choose_weights(runtime=3, diff=1)
        -> [0.75, 0.25]

    Args:
        **importance: keyword args of metric=importance (relative numbers).

    Returns:
        A list of weights in the same order as the arguments.

    """
    total = sum(importance.values())
    if total == 0:
        raise ValueError("At least one importance value must be > 0")

    return [v / total for v in importance.values()]


def normalize_by_max(values: list[float]) -> list[float]:
    mx = max(values)
    if mx == 0:
        return [0.0] * len(values)
    return [v / mx for v in values]


def create_score_dictionary_from_metrics(weights: list[float], *metrics: list[float]) -> dict[int, float]:
    """Combine multiple metrics into a single weighted score dictionary.

    Each metric is a list of values (smaller = better).
    The total score for each index is the weighted sum of its values
    across all metrics:

        score[index] = Σ (value * weight)

    Args:
        weights: A list of weights, one per metric. Larger weight = more influence.
        *metrics: Lists of values (one list per metric, aligned by index).

    Returns:
        A dictionary mapping each index to its combined weighted score.

    """
    if len(weights) != len(metrics):
        raise ValueError("Number of weights must match number of metrics")

    combined: dict[int, float] = {}

    for weight, metric in zip(weights, metrics):
        for idx, value in enumerate(metric):
            combined[idx] = combined.get(idx, 0) + value * weight

    return combined


def diff_length(a: str, b: str) -> int:
    """Compute the length (in characters) of the unified diff between two strings.

    Args:
        a (str): Original string.
        b (str): Modified string.

    Returns:
        int: Total number of characters in the diff.

    """
    # Split input strings into lines for line-by-line diff
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)

    # Compute unified diff
    diff_lines = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))

    # Join all lines with newline to calculate total diff length
    diff_text = "\n".join(diff_lines)

    return len(diff_text)


def create_rank_dictionary_compact(int_array: list[int]) -> dict[int, int]:
    """Create a dictionary from a list of ints, mapping the original index to its rank.

    This version uses a more compact, "Pythonic" implementation.

    Args:
        int_array: A list of integers.

    Returns:
        A dictionary where keys are original indices and values are the
        rank of the element in ascending order.

    """
    # Sort the indices of the array based on their corresponding values
    sorted_indices = sorted(range(len(int_array)), key=lambda i: int_array[i])

    # Create a dictionary mapping the original index to its rank (its position in the sorted list)
    return {original_index: rank for rank, original_index in enumerate(sorted_indices)}
