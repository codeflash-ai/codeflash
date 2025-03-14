import pytest

from code_to_optimize.bubble_sort_codeflash_trace import sorter, Sorter


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(500))))
    assert result == list(range(500))

# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(500))))
    assert result == list(range(500))

def test_class_sort(benchmark):
    obj = Sorter(list(reversed(range(100))))
    result1 = benchmark(obj.sorter, 2)
    result2 = benchmark(Sorter.sort_class, list(reversed(range(100))))
    result3 = benchmark(Sorter.sort_static, list(reversed(range(100))))
    result4 = benchmark(Sorter, [1,2,3])