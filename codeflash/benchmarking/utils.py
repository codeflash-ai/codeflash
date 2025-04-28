from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rich.table import Table

from codeflash.cli_cmds.console import console, logger
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
    for func_path, sorted_tests in function_to_results.items():
        console.print()
        function_name = func_path.split(":")[-1]

        table = Table(title=f"Function: {function_name}", border_style="blue", show_lines=True)
        table.add_column("Benchmark Module Path", style="cyan", overflow="fold")
        table.add_column("Test Function", style="magenta", overflow="fold")
        table.add_column("Total Time (ms)", justify="right", style="green")
        table.add_column("Function Time (ms)", justify="right", style="yellow")
        table.add_column("Percentage (%)", justify="right", style="red")

        multi_call_bases = set()
        call_1_tests = []

        for i, (benchmark_key, _, _, _) in enumerate(sorted_tests):
            test_function = benchmark_key.function_name
            module_path = benchmark_key.module_path
            if "::call_" in test_function:
                try:
                    base_name, call_part = test_function.rsplit("::call_", 1)
                    call_num = int(call_part)
                    if call_num == 1:
                        call_1_tests.append((i, base_name, module_path))
                    elif call_num > 1:
                        multi_call_bases.add((base_name, module_path))
                except ValueError:
                    pass

        tests_to_modify = {
            index: base_name
            for index, base_name, module_path in call_1_tests
            if (base_name, module_path) not in multi_call_bases
        }

        for i, (benchmark_key, total_time, func_time, percentage) in enumerate(sorted_tests):
            module_path = benchmark_key.module_path
            test_function_display = tests_to_modify.get(i, benchmark_key.function_name)

            if total_time == 0.0:
                table.add_row(module_path, test_function_display, "N/A", "N/A", "N/A")
            else:
                table.add_row(
                    module_path, test_function_display, f"{total_time:.3f}", f"{func_time:.3f}", f"{percentage:.2f}"
                )

        console.print(table)


def process_benchmark_data(
    replay_performance_gain: dict[BenchmarkKey, float],
    fto_benchmark_timings: dict[BenchmarkKey, int],
    total_benchmark_timings: dict[BenchmarkKey, int],
) -> Optional[ProcessedBenchmarkInfo]:
    """Process benchmark data and generate detailed benchmark information.

    Args:
        replay_performance_gain: The performance gain from replay
        fto_benchmark_timings: Function to optimize benchmark timings
        total_benchmark_timings: Total benchmark timings

    Returns:
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
