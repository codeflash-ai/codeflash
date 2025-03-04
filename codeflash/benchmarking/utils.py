def print_benchmark_table(function_benchmark_timings, total_benchmark_timings):
    # Print table header
    print(f"{'Benchmark Test':<50} | {'Total Time (s)':<15} | {'Function Time (s)':<15} | {'Percentage (%)':<15}")
    print("-" * 100)

    # Process each function's benchmark data
    for func_path, test_times in function_benchmark_timings.items():
        function_name = func_path.split(":")[-1]
        print(f"\n== Function: {function_name} ==")

        # Sort by percentage (highest first)
        sorted_tests = []
        for test_name, func_time in test_times.items():
            total_time = total_benchmark_timings.get(test_name, 0)
            if total_time > 0:
                percentage = (func_time / total_time) * 100
                sorted_tests.append((test_name, total_time, func_time, percentage))

        sorted_tests.sort(key=lambda x: x[3], reverse=True)

        # Print each test's data
        for test_name, total_time, func_time, percentage in sorted_tests:
            print(f"{test_name:<50} | {total_time:<15.3f} | {func_time:<15.3f} | {percentage:<15.2f}")

# Usage

