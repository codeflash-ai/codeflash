from code_to_optimize.bubble_sort_multithread import multithreaded_sorter

def test_benchmark_sort(benchmark):
    benchmark(multithreaded_sorter, [list(range(1000)) for i in range (10)])