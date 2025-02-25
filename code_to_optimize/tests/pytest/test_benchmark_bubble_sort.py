from code_to_optimize.bubble_sort import sorter


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(5000))))
    assert result == list(range(5000))
