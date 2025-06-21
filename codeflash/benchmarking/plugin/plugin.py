from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.models.models import BenchmarkKey


class CodeFlashBenchmarkPlugin:
    def __init__(self) -> None:
        self._trace_path = None
        self._connection = None
        self.project_root = None
        self.benchmark_timings = []

    def setup(self, trace_path: str, project_root: str) -> None:
        try:
            # Open connection
            self.project_root = project_root
            self._trace_path = trace_path
            self._connection = sqlite3.connect(self._trace_path)
            cur = self._connection.cursor()
            cur.execute("PRAGMA synchronous = OFF")
            cur.execute("PRAGMA journal_mode = MEMORY")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_timings("
                "benchmark_module_path TEXT, benchmark_function_name TEXT, benchmark_line_number INTEGER,"
                "benchmark_time_ns INTEGER)"
            )
            self._connection.commit()
            self.close()  # Reopen only at the end of pytest session
        except Exception as e:
            print(f"Database setup error: {e}")
            if self._connection:
                self._connection.close()
                self._connection = None
            raise

    def write_benchmark_timings(self) -> None:
        if not self.benchmark_timings:
            return  # No data to write

        if self._connection is None:
            self._connection = sqlite3.connect(self._trace_path)

        try:
            cur = self._connection.cursor()
            # Insert data into the benchmark_timings table
            cur.executemany(
                "INSERT INTO benchmark_timings (benchmark_module_path, benchmark_function_name, benchmark_line_number, benchmark_time_ns) VALUES (?, ?, ?, ?)",
                self.benchmark_timings,
            )
            self._connection.commit()
            self.benchmark_timings = []  # Clear the benchmark timings list
        except Exception as e:
            print(f"Error writing to benchmark timings database: {e}")
            self._connection.rollback()
            raise

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    @staticmethod
    def get_function_benchmark_timings(trace_path: Path) -> dict[str, dict[BenchmarkKey, int]]:
        """Process the trace file and extract timing data for all functions.

        Args:
        ----
            trace_path: Path to the trace file

        Returns:
        -------
            A nested dictionary where:
            - Outer keys are module_name.qualified_name (module.class.function)
            - Inner keys are of type BenchmarkKey
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
                "benchmark_module_path, benchmark_function_name, benchmark_line_number, function_time_ns "
                "FROM benchmark_function_timings"
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
                benchmark_key = BenchmarkKey(module_path=benchmark_file, function_name=benchmark_func)
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

    @staticmethod
    def get_benchmark_timings(trace_path: Path) -> dict[BenchmarkKey, int]:
        """Extract total benchmark timings from trace files.

        Args:
        ----
            trace_path: Path to the trace file

        Returns:
        -------
            A dictionary mapping where:
            - Keys are of type BenchmarkKey
            - Values are total benchmark timing in milliseconds (with overhead subtracted)

        """
        # Initialize the result dictionary
        result = {}
        overhead_by_benchmark = {}

        # Connect to the SQLite database
        connection = sqlite3.connect(trace_path)
        cursor = connection.cursor()

        try:
            # Query the benchmark_function_timings table to get total overhead for each benchmark
            cursor.execute(
                "SELECT benchmark_module_path, benchmark_function_name, benchmark_line_number, SUM(overhead_time_ns) "
                "FROM benchmark_function_timings "
                "GROUP BY benchmark_module_path, benchmark_function_name, benchmark_line_number"
            )

            # Process overhead information
            for row in cursor.fetchall():
                benchmark_file, benchmark_func, benchmark_line, total_overhead_ns = row
                benchmark_key = BenchmarkKey(module_path=benchmark_file, function_name=benchmark_func)
                overhead_by_benchmark[benchmark_key] = total_overhead_ns or 0  # Handle NULL sum case

            # Query the benchmark_timings table for total times
            cursor.execute(
                "SELECT benchmark_module_path, benchmark_function_name, benchmark_line_number, benchmark_time_ns "
                "FROM benchmark_timings"
            )

            # Process each row and subtract overhead
            for row in cursor.fetchall():
                benchmark_file, benchmark_func, benchmark_line, time_ns = row

                # Create the benchmark key (file::function::line)
                benchmark_key = BenchmarkKey(module_path=benchmark_file, function_name=benchmark_func)
                # Subtract overhead from total time
                overhead = overhead_by_benchmark.get(benchmark_key, 0)
                result[benchmark_key] = time_ns - overhead

        finally:
            # Close the connection
            connection.close()

        return result

    # Pytest hooks
    @pytest.hookimpl
    def pytest_sessionfinish(self, session, exitstatus) -> None:  # noqa: ANN001, ARG002
        """Execute after whole test run is completed."""
        # Write any remaining benchmark timings to the database
        codeflash_trace.close()
        if self.benchmark_timings:
            self.write_benchmark_timings()
        # Close the database connection
        self.close()

    @staticmethod
    def pytest_addoption(parser: pytest.Parser) -> None:
        parser.addoption("--codeflash-trace", action="store_true", default=False, help="Enable CodeFlash tracing")

    @staticmethod
    def pytest_plugin_registered(plugin, manager) -> None:  # noqa: ANN001
        # Not necessary since run with -p no:benchmark, but just in case
        if hasattr(plugin, "name") and plugin.name == "pytest-benchmark":
            manager.unregister(plugin)

    @staticmethod
    def pytest_configure(config: pytest.Config) -> None:
        """Register the benchmark marker."""
        config.addinivalue_line(
            "markers", "benchmark: mark test as a benchmark that should be run with codeflash tracing"
        )

    @staticmethod
    def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
        # Skip tests that don't have the benchmark fixture
        if not config.getoption("--codeflash-trace"):
            return

        skip_no_benchmark = pytest.mark.skip(reason="Test requires benchmark fixture")
        for item in items:
            # Check for direct benchmark fixture usage
            has_fixture = hasattr(item, "fixturenames") and "benchmark" in item.fixturenames

            # Check for @pytest.mark.benchmark marker
            has_marker = False
            if hasattr(item, "get_closest_marker"):
                marker = item.get_closest_marker("benchmark")
                if marker is not None:
                    has_marker = True

            # Skip if neither fixture nor marker is present
            if not (has_fixture or has_marker):
                item.add_marker(skip_no_benchmark)

    # Benchmark fixture
    class Benchmark:  # noqa: D106
        def __init__(self, request: pytest.FixtureRequest) -> None:
            self.request = request

        def __call__(self, func, *args, **kwargs):  # type: ignore  # noqa: ANN001, ANN002, ANN003, ANN204, PGH003
            """Handle both direct function calls and decorator usage."""
            if args or kwargs:
                # Used as benchmark(func, *args, **kwargs)
                return self._run_benchmark(func, *args, **kwargs)

            # Used as @benchmark decorator
            def wrapped_func(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                return func(*args, **kwargs)

            self._run_benchmark(func)
            return wrapped_func

        def _run_benchmark(self, func, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            """Actual benchmark implementation."""
            benchmark_module_path = module_name_from_file_path(
                Path(str(self.request.node.fspath)), Path(codeflash_benchmark_plugin.project_root)
            )
            benchmark_function_name = self.request.node.name
            line_number = int(str(sys._getframe(2).f_lineno))  # 2 frames up in the call stack  # noqa: SLF001
            # Set env vars
            os.environ["CODEFLASH_BENCHMARK_FUNCTION_NAME"] = benchmark_function_name
            os.environ["CODEFLASH_BENCHMARK_MODULE_PATH"] = benchmark_module_path
            os.environ["CODEFLASH_BENCHMARK_LINE_NUMBER"] = str(line_number)
            os.environ["CODEFLASH_BENCHMARKING"] = "True"
            # Run the function
            start = time.time_ns()
            result = func(*args, **kwargs)
            end = time.time_ns()
            # Reset the environment variable
            os.environ["CODEFLASH_BENCHMARKING"] = "False"

            # Write function calls
            codeflash_trace.write_function_timings()
            # Reset function call count
            codeflash_trace.function_call_count = 0
            # Add to the benchmark timings buffer
            codeflash_benchmark_plugin.benchmark_timings.append(
                (benchmark_module_path, benchmark_function_name, line_number, end - start)
            )

            return result

    @staticmethod
    @pytest.fixture
    def benchmark(request: pytest.FixtureRequest) -> object:
        if not request.config.getoption("--codeflash-trace"):
            return None

        return CodeFlashBenchmarkPlugin.Benchmark(request)


codeflash_benchmark_plugin = CodeFlashBenchmarkPlugin()
