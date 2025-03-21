from rich.console import Console
from rich.table import Table


def print_benchmark_table(function_benchmark_timings: dict[str, dict[str, int]],
                          total_benchmark_timings: dict[str, int]):
    console = Console()

    # Process each function's benchmark data
    for func_path, test_times in function_benchmark_timings.items():
        function_name = func_path.split(":")[-1]

        # Create a table for this function
        table = Table(title=f"Function: {function_name}", border_style="blue")

        # Add columns
        table.add_column("Benchmark Test", style="cyan", no_wrap=True)
        table.add_column("Total Time (ms)", justify="right", style="green")
        table.add_column("Function Time (ms)", justify="right", style="yellow")
        table.add_column("Percentage (%)", justify="right", style="red")

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

        # Add rows to the table
        for test_name, total_time, func_time, percentage in sorted_tests:
            benchmark_file, benchmark_func, benchmark_line = test_name.split("::")
            benchmark_name = f"{benchmark_file}::{benchmark_func}"
            table.add_row(
                benchmark_name,
                f"{total_time:.3f}",
                f"{func_time:.3f}",
                f"{percentage:.2f}"
            )

        # Print the table
        console.print(table)