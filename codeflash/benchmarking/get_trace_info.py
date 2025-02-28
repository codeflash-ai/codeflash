import sqlite3
from pathlib import Path
from typing import Dict, Set

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def get_function_benchmark_timings(trace_dir: Path, all_functions_to_optimize: list[FunctionToOptimize]) -> dict[str, dict[str, float]]:
    """Process all trace files in the given directory and extract timing data for the specified functions.

    Args:
        trace_dir: Path to the directory containing .trace files
        all_functions_to_optimize: Set of FunctionToOptimize objects representing functions to include

    Returns:
        A nested dictionary where:
        - Outer keys are function qualified names with file name
        - Inner keys are benchmark names (trace filename without .trace extension)
        - Values are function timing in milliseconds

    """
    # Create a mapping of (filename, function_name, class_name) -> qualified_name for efficient lookups
    function_lookup = {}
    function_benchmark_timings = {}

    for func in all_functions_to_optimize:
        qualified_name = func.qualified_name_with_file_name

        # Extract components (assumes Path.name gives only filename without directory)
        filename = func.file_path
        function_name = func.function_name

        # Get class name if there's a parent
        class_name = func.parents[0].name if func.parents else None

        # Store in lookup dictionary
        key = (filename, function_name, class_name)
        function_lookup[key] = qualified_name
        function_benchmark_timings[qualified_name] = {}

    # Find all .trace files in the directory
    trace_files = list(trace_dir.glob("*.trace"))

    for trace_file in trace_files:
        # Extract benchmark name from filename (without .trace)
        benchmark_name = trace_file.stem

        # Connect to the trace database
        conn = sqlite3.connect(trace_file)
        cursor = conn.cursor()

        # For each function we're interested in, query the database directly
        for (filename, function_name, class_name), qualified_name in function_lookup.items():
            # Adjust query based on whether we have a class name
            if class_name:
                cursor.execute(
                    "SELECT total_time_ns FROM pstats WHERE filename LIKE ? AND function = ? AND class_name = ?",
                    (f"%{filename}", function_name, class_name)
                )
            else:
                cursor.execute(
                    "SELECT total_time_ns FROM pstats WHERE filename LIKE ? AND function = ? AND (class_name IS NULL OR class_name = '')",
                    (f"%{filename}", function_name)
                )

            result = cursor.fetchone()
            if result:
                time_ns = result[0]
                function_benchmark_timings[qualified_name][benchmark_name] = time_ns / 1e6  # Convert to milliseconds

        conn.close()

    return function_benchmark_timings


def get_benchmark_timings(trace_dir: Path) -> dict[str, float]:
    """Extract total benchmark timings from trace files.

    Args:
        trace_dir: Path to the directory containing .trace files

    Returns:
        A dictionary mapping benchmark names to their total execution time in milliseconds.
    """
    benchmark_timings = {}

    # Find all .trace files in the directory
    trace_files = list(trace_dir.glob("*.trace"))

    for trace_file in trace_files:
        # Extract benchmark name from filename (without .trace extension)
        benchmark_name = trace_file.stem

        # Connect to the trace database
        conn = sqlite3.connect(trace_file)
        cursor = conn.cursor()

        # Query the total_time table for the benchmark's total execution time
        try:
            cursor.execute("SELECT time_ns FROM total_time")
            result = cursor.fetchone()
            if result:
                time_ns = result[0]
                # Convert nanoseconds to milliseconds
                benchmark_timings[benchmark_name] = time_ns / 1e6
        except sqlite3.OperationalError:
            # Handle case where total_time table might not exist
            print(f"Warning: Could not get total time for benchmark {benchmark_name}")

        conn.close()

    return benchmark_timings
