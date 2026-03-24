from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash.models.models import BenchmarkDetail, ProcessedBenchmarkInfo
from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.result.critic import performance_gain

if TYPE_CHECKING:
    from codeflash.models.models import BenchmarkKey


logger = logging.getLogger("codeflash_python")


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
                logger.debug(
                    "Skipping test %s due to func_time %s > total_time %s", benchmark_key, func_time, total_time
                )
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
    headers = ["Benchmark Module Path", "Test Function", "Total Time (ms)", "Function Time (ms)", "Percentage (%)"]
    for func_path, sorted_tests in function_to_results.items():
        function_name = func_path.split(":")[-1]

        rows = []
        for benchmark_key, total_time, func_time, percentage in sorted_tests:
            module_path = benchmark_key.module_path
            test_function = benchmark_key.function_name
            if total_time == 0.0:
                rows.append([module_path, test_function, "N/A", "N/A", "N/A"])
            else:
                rows.append([module_path, test_function, f"{total_time:.3f}", f"{func_time:.3f}", f"{percentage:.2f}"])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        lines = [f"Function: {function_name}", fmt.format(*headers), "-" * sum([*col_widths, 2 * (len(headers) - 1)])]
        for row in rows:
            lines.append(fmt.format(*row))
        logger.info("\n".join(lines))


def process_benchmark_data(
    replay_performance_gain: dict[BenchmarkKey, float],
    fto_benchmark_timings: dict[BenchmarkKey, int],
    total_benchmark_timings: dict[BenchmarkKey, int],
) -> ProcessedBenchmarkInfo | None:
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
