import pytest

from code_to_optimize.bubble_sort import sorter


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(5000))))
    assert result == list(range(5000))

# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(5000))))
    assert result == list(range(5000))