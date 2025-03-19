def print_benchmark_table(function_benchmark_timings: dict[str,dict[str,int]], total_benchmark_timings: dict[str,int]):
    # Define column widths
    benchmark_col_width = 50
    time_col_width = 15

    # Print table header
    header = f"{'Benchmark Test':{benchmark_col_width}} | {'Total Time (ms)':{time_col_width}} | {'Function Time (ms)':{time_col_width}} | {'Percentage (%)':{time_col_width}}"
    print(header)
    print("-" * len(header))

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
                # Convert nanoseconds to milliseconds
                func_time_ms = func_time / 1_000_000
                total_time_ms = total_time / 1_000_000
                sorted_tests.append((test_name, total_time_ms, func_time_ms, percentage))

        sorted_tests.sort(key=lambda x: x[3], reverse=True)

        # Print each test's data
        for test_name, total_time, func_time, percentage in sorted_tests:
            benchmark_file, benchmark_func, benchmark_line = test_name.split("::")
            benchmark_name = f"{benchmark_file}::{benchmark_func}"
            print(f"{benchmark_name:{benchmark_col_width}} | {total_time:{time_col_width}.3f} | {func_time:{time_col_width}.3f} | {percentage:{time_col_width}.2f}")
    print()
