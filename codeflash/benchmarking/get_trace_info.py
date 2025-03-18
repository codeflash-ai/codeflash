import sqlite3
from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def get_function_benchmark_timings(trace_path: Path) -> dict[str, dict[str, float]]:
    """Process the trace file and extract timing data for all functions.

    Args:
        trace_path: Path to the trace file
        all_functions_to_optimize: List of FunctionToOptimize objects (not used directly,
                                  but kept for backward compatibility)

    Returns:
        A nested dictionary where:
        - Outer keys are module_name.qualified_name (module.class.function)
        - Inner keys are benchmark filename :: benchmark test function :: line number
        - Values are function timing in milliseconds

    """
    # Initialize the result dictionary
    result = {}

    # Connect to the SQLite database
    connection = sqlite3.connect(trace_path)
    cursor = connection.cursor()

    try:
        # Query the function_calls table for all function calls
        cursor.execute(
            "SELECT module_name, class_name, function_name, "
            "benchmark_file_name, benchmark_function_name, benchmark_line_number, "
            "(time_ns - overhead_time_ns) as actual_time_ns "
            "FROM function_calls"
        )

        # Process each row
        for row in cursor.fetchall():
            module_name, class_name, function_name, benchmark_file, benchmark_func, benchmark_line, time_ns = row

            # Create the function key (module_name.class_name.function_name)
            if class_name:
                qualified_name = f"{module_name}.{class_name}.{function_name}"
            else:
                qualified_name = f"{module_name}.{function_name}"

            # Create the benchmark key (file::function::line)
            benchmark_key = f"{benchmark_file}::{benchmark_func}::{benchmark_line}"

            # Initialize the inner dictionary if needed
            if qualified_name not in result:
                result[qualified_name] = {}

            # If multiple calls to the same function in the same benchmark,
            # add the times together
            if benchmark_key in result[qualified_name]:
                result[qualified_name][benchmark_key] += time_ns
            else:
                result[qualified_name][benchmark_key] = time_ns

    finally:
        # Close the connection
        connection.close()

    return result


def get_benchmark_timings(trace_path: Path) -> dict[str, float]:
    """Extract total benchmark timings from trace files.

    Args:
        trace_path: Path to the trace file

    Returns:
        A dictionary mapping where:
        - Keys are benchmark filename :: benchmark test function :: line number
        - Values are total benchmark timing in milliseconds

    """
    # Initialize the result dictionary
    result = {}

    # Connect to the SQLite database
    connection = sqlite3.connect(trace_path)
    cursor = connection.cursor()

    try:
        # Query the benchmark_timings table
        cursor.execute(
            "SELECT benchmark_file_name, benchmark_function_name, benchmark_line_number, time_ns "
            "FROM benchmark_timings"
        )

        # Process each row
        for row in cursor.fetchall():
            benchmark_file, benchmark_func, benchmark_line, time_ns = row

            # Create the benchmark key (file::function::line)
            benchmark_key = f"{benchmark_file}::{benchmark_func}::{benchmark_line}"

            # Store the timing
            result[benchmark_key] = time_ns

    finally:
        # Close the connection
        connection.close()

    return result
