import pytest
from code_to_optimize.bubble_sort_codeflash_trace import sorter

def test_benchmark_sort(benchmark):
    @benchmark
    def do_sort():
        sorter(list(reversed(range(500))))

@pytest.mark.benchmark(group="benchmark_decorator")
def test_pytest_mark(benchmark):
    benchmark(sorter, list(reversed(range(500))))