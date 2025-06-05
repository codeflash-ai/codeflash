import pytest

from code_to_optimize.bubble_sort import sorter


class DummyBenchmark:
    def __call__(self, func, *args, **kwargs):
        # Mimic calling benchmark(func, *args, **kwargs)
        return func(*args, **kwargs)

    def __getattr__(self, name):
        # Mimic benchmark attributes like .pedantic, .extra_info etc.
        def dummy(*args, **kwargs):
            return None

        return dummy


@pytest.fixture
def benchmark():
    return DummyBenchmark()


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(500))))
    assert result == list(range(500))


# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(500))))
    assert result == list(range(500))
