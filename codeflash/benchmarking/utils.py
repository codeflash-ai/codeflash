from rich.console import Console
from rich.table import Table

from codeflash.cli_cmds.console import logger


def validate_and_format_benchmark_table(function_benchmark_timings: dict[str, dict[str, int]],
                          total_benchmark_timings: dict[str, int]) -> dict[str, list[tuple[str, float, float, float]]]:
    function_to_result = {}
    # Process each function's benchmark data
    for func_path, test_times in function_benchmark_timings.items():
        # Sort by percentage (highest first)
        sorted_tests = []
        for test_name, func_time in test_times.items():
            total_time = total_benchmark_timings.get(test_name, 0)
            if func_time > total_time:
                logger.debug(f"Skipping test {test_name} due to func_time {func_time} > total_time {total_time}")
                # If the function time is greater than total time, likely to have multithreading / multiprocessing issues.
                # Do not try to project the optimization impact for this function.
                sorted_tests.append((test_name, 0.0, 0.0, 0.0))
            if total_time > 0:
                percentage = (func_time / total_time) * 100
                # Convert nanoseconds to milliseconds
                func_time_ms = func_time / 1_000_000
                total_time_ms = total_time / 1_000_000
                sorted_tests.append((test_name, total_time_ms, func_time_ms, percentage))
        sorted_tests.sort(key=lambda x: x[3], reverse=True)
        function_to_result[func_path] = sorted_tests
    return function_to_result

def print_benchmark_table(function_to_results: dict[str, list[tuple[str, float, float, float]]]) -> None:
    console = Console()
    for func_path, sorted_tests in function_to_results.items():
        function_name = func_path.split(":")[-1]

        # Create a table for this function
        table = Table(title=f"Function: {function_name}", border_style="blue")

        # Add columns
        table.add_column("Benchmark Test", style="cyan", no_wrap=True)
        table.add_column("Total Time (ms)", justify="right", style="green")
        table.add_column("Function Time (ms)", justify="right", style="yellow")
        table.add_column("Percentage (%)", justify="right", style="red")

        for test_name, total_time, func_time, percentage in sorted_tests:
            benchmark_file, benchmark_func, benchmark_line = test_name.split("::")
            benchmark_name = f"{benchmark_file}::{benchmark_func}"
            if total_time == 0.0:
                table.add_row(
                    benchmark_name,
                    "N/A",
                    "N/A",
                    "N/A"
                )
            else:
                table.add_row(
                    benchmark_name,
                    f"{total_time:.3f}",
                    f"{func_time:.3f}",
                    f"{percentage:.2f}"
                )

        # Print the table
        console.print(table)