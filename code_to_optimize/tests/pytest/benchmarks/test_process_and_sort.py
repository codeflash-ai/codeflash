from code_to_optimize.process_and_bubble_sort import compute_and_sort
from code_to_optimize.bubble_sort2 import sorter
def test_compute_and_sort(benchmark):
    result = benchmark(compute_and_sort, list(reversed(range(5000))))
    assert result == 6247083.5

def test_no_func(benchmark):
    benchmark(sorter, list(reversed(range(5000))))