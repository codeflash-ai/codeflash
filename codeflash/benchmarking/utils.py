from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.table import Table

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.models import BenchmarkDetail, ProcessedBenchmarkInfo
from codeflash.result.critic import performance_gain

if TYPE_CHECKING:
    from codeflash.models.models import BenchmarkKey


def validate_and_format_benchmark_table(
    function_benchmark_timings: dict[str, dict[BenchmarkKey, int]], total_benchmark_timings: dict[BenchmarkKey, int]
) -> dict[str, list[tuple[BenchmarkKey, float, float, float]]]:
    function_to_result = {}
    # Process each function's benchmark data
    for func_path, test_times in function_benchmark_timings.items():
        # Sort by percentage (highest first)
        sorted_tests = []
        for benchmark_key, func_time in test_times.items():
            total_time = total_benchmark_timings.get(benchmark_key, 0)
            if func_time > total_time:
                logger.debug(f"Skipping test {benchmark_key} due to func_time {func_time} > total_time {total_time}")
                # If the function time is greater than total time, likely to have multithreading / multiprocessing issues.
                # Do not try to project the optimization impact for this function.
                sorted_tests.append((benchmark_key, 0.0, 0.0, 0.0))
            elif total_time > 0:
                percentage = (func_time / total_time) * 100
                # Convert nanoseconds to milliseconds
                func_time_ms = func_time / 1_000_000
                total_time_ms = total_time / 1_000_000
                sorted_tests.append((benchmark_key, total_time_ms, func_time_ms, percentage))
        sorted_tests.sort(key=lambda x: x[3], reverse=True)
        function_to_result[func_path] = sorted_tests
    return function_to_result


def print_benchmark_table(function_to_results: dict[str, list[tuple[BenchmarkKey, float, float, float]]]) -> None:
    try:
        terminal_width = int(shutil.get_terminal_size().columns * 0.9)
    except Exception:
        terminal_width = 120  # Fallback width
    console = Console(width=terminal_width)
    for func_path, sorted_tests in function_to_results.items():
        console.print()
        function_name = func_path.split(":")[-1]

        # Create a table for this function
        table = Table(title=f"Function: {function_name}", width=terminal_width, border_style="blue", show_lines=True)
        benchmark_col_width = max(int(terminal_width * 0.4), 40)
        # Add columns - split the benchmark test into two columns
        table.add_column("Benchmark Module Path", width=benchmark_col_width, style="cyan", overflow="fold")
        table.add_column("Test Function", style="magenta", overflow="fold")
        table.add_column("Total Time (ms)", justify="right", style="green")
        table.add_column("Function Time (ms)", justify="right", style="yellow")
        table.add_column("Percentage (%)", justify="right", style="red")

        for benchmark_key, total_time, func_time, percentage in sorted_tests:
            # Split the benchmark test into module path and function name
            module_path = benchmark_key.module_path
            test_function = benchmark_key.function_name

            if total_time == 0.0:
                table.add_row(module_path, test_function, "N/A", "N/A", "N/A")
            else:
                table.add_row(module_path, test_function, f"{total_time:.3f}", f"{func_time:.3f}", f"{percentage:.2f}")

        # Print the table
        console.print(table)


def process_benchmark_data(
    replay_performance_gain: dict[BenchmarkKey, float],
    fto_benchmark_timings: dict[BenchmarkKey, int],
    total_benchmark_timings: dict[BenchmarkKey, int],
) -> Optional[ProcessedBenchmarkInfo]:
    """Process benchmark data and generate detailed benchmark information.

    Args:
    ----
        replay_performance_gain: The performance gain from replay
        fto_benchmark_timings: Function to optimize benchmark timings
        total_benchmark_timings: Total benchmark timings

    Returns:
    -------
        ProcessedBenchmarkInfo containing processed benchmark details

    """
    if not replay_performance_gain or not fto_benchmark_timings or not total_benchmark_timings:
        return None

    benchmark_details = []

    for benchmark_key, og_benchmark_timing in fto_benchmark_timings.items():
        total_benchmark_timing = total_benchmark_timings.get(benchmark_key, 0)

        if total_benchmark_timing == 0:
            continue  # Skip benchmarks with zero timing

        # Calculate expected new benchmark timing
        expected_new_benchmark_timing = (
            total_benchmark_timing
            - og_benchmark_timing
            + (1 / (replay_performance_gain[benchmark_key] + 1)) * og_benchmark_timing
        )

        # Calculate speedup
        benchmark_speedup_percent = (
            performance_gain(
                original_runtime_ns=total_benchmark_timing, optimized_runtime_ns=int(expected_new_benchmark_timing)
            )
            * 100
        )

        benchmark_details.append(
            BenchmarkDetail(
                benchmark_name=benchmark_key.module_path,
                test_function=benchmark_key.function_name,
                original_timing=humanize_runtime(int(total_benchmark_timing)),
                expected_new_timing=humanize_runtime(int(expected_new_benchmark_timing)),
                speedup_percent=benchmark_speedup_percent,
            )
        )

    return ProcessedBenchmarkInfo(benchmark_details=benchmark_details)
