import pytest

from code_to_optimize.bubble_sort import sorter


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(500))))
    assert result == list(range(500))

@pytest.mark.benchmark
def test_sort_with_marker(benchmark):
    @benchmark
    def inner_fn():
        return sorter(list(reversed(range(5))))
    assert 1==1

# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(500))))
    assert result == list(range(500))
