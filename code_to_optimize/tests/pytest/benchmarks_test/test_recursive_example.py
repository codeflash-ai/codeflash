from code_to_optimize.bubble_sort_codeflash_trace import recursive_bubble_sort


def test_recursive_sort(benchmark):
    result = benchmark(recursive_bubble_sort, list(reversed(range(500))))
    assert result == list(range(500))