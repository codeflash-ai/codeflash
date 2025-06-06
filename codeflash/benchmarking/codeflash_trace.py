import functools
import os
import pickle
import sqlite3
import threading
import time
from typing import Any, Callable

from codeflash.picklepatch.pickle_patcher import PicklePatcher


class CodeflashTrace:
    """Decorator class that traces and profiles function execution."""

    def __init__(self) -> None:
        self.function_calls_data = []
        self.function_call_count = 0
        self.pickle_count_limit = 1000
        self._connection = None
        self._trace_path = None
        self._thread_local = threading.local()
        self._thread_local.active_functions = set()

    def setup(self, trace_path: str) -> None:
        """Set up the database connection for direct writing.

        Args:
        ----
            trace_path: Path to the trace database file

        """
        try:
            self._trace_path = trace_path
            self._connection = sqlite3.connect(self._trace_path)
            cur = self._connection.cursor()
            cur.execute("PRAGMA synchronous = OFF")
            cur.execute("PRAGMA journal_mode = MEMORY")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_function_timings("
                "function_name TEXT, class_name TEXT, module_name TEXT, file_path TEXT,"
                "benchmark_function_name TEXT, benchmark_module_path TEXT, benchmark_line_number INTEGER,"
                "function_time_ns INTEGER, overhead_time_ns INTEGER, args BLOB, kwargs BLOB)"
            )
            self._connection.commit()
        except Exception as e:
            print(f"Database setup error: {e}")
            if self._connection:
                self._connection.close()
                self._connection = None
            raise

    def write_function_timings(self) -> None:
        """Write function call data directly to the database.

        Args:
        ----
            data: List of function call data tuples to write

        """
        if not self.function_calls_data:
            return  # No data to write

        if self._connection is None and self._trace_path is not None:
            self._connection = sqlite3.connect(self._trace_path)

        try:
            cur = self._connection.cursor()
            # Insert data into the benchmark_function_timings table
            cur.executemany(
                "INSERT INTO benchmark_function_timings"
                "(function_name, class_name, module_name, file_path, benchmark_function_name, "
                "benchmark_module_path, benchmark_line_number, function_time_ns, overhead_time_ns, args, kwargs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self.function_calls_data,
            )
            self._connection.commit()
            self.function_calls_data = []
        except Exception as e:
            print(f"Error writing to function timings database: {e}")
            if self._connection:
                self._connection.rollback()
            raise

    def open(self) -> None:
        """Open the database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self._trace_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator to trace function execution.

        Args:
        ----
            func: The function to be decorated

        Returns:
        -------
            The wrapped function

        """
        func_id = (func.__module__, func.__name__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:  # noqa: ANN002, ANN003, ANN401
            # Initialize thread-local active functions set if it doesn't exist
            if not hasattr(self._thread_local, "active_functions"):
                self._thread_local.active_functions = set()
            # If it's in a recursive function, just return the result
            if func_id in self._thread_local.active_functions:
                return func(*args, **kwargs)
            # Track active functions so we can detect recursive functions
            self._thread_local.active_functions.add(func_id)
            # Measure execution time
            start_time = time.thread_time_ns()
            result = func(*args, **kwargs)
            end_time = time.thread_time_ns()
            # Calculate execution time
            execution_time = end_time - start_time
            self.function_call_count += 1

            # Check if currently in pytest benchmark fixture
            if os.environ.get("CODEFLASH_BENCHMARKING", "False") == "False":
                self._thread_local.active_functions.remove(func_id)
                return result
            # Get benchmark info from environment
            benchmark_function_name = os.environ.get("CODEFLASH_BENCHMARK_FUNCTION_NAME", "")
            benchmark_module_path = os.environ.get("CODEFLASH_BENCHMARK_MODULE_PATH", "")
            benchmark_line_number = os.environ.get("CODEFLASH_BENCHMARK_LINE_NUMBER", "")
            # Get class name
            class_name = ""
            qualname = func.__qualname__
            if "." in qualname:
                class_name = qualname.split(".")[0]

            # Limit pickle count so memory does not explode
            if self.function_call_count > self.pickle_count_limit:
                print("Pickle limit reached")
                self._thread_local.active_functions.remove(func_id)
                overhead_time = time.thread_time_ns() - end_time
                self.function_calls_data.append(
                    (
                        func.__name__,
                        class_name,
                        func.__module__,
                        func.__code__.co_filename,
                        benchmark_function_name,
                        benchmark_module_path,
                        benchmark_line_number,
                        execution_time,
                        overhead_time,
                        None,
                        None,
                    )
                )
                return result

            try:
                # Pickle the arguments
                pickled_args = PicklePatcher.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_kwargs = PicklePatcher.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"Error pickling arguments for function {func.__name__}: {e}")
                # Add to the list of function calls without pickled args. Used for timing info only
                self._thread_local.active_functions.remove(func_id)
                overhead_time = time.thread_time_ns() - end_time
                self.function_calls_data.append(
                    (
                        func.__name__,
                        class_name,
                        func.__module__,
                        func.__code__.co_filename,
                        benchmark_function_name,
                        benchmark_module_path,
                        benchmark_line_number,
                        execution_time,
                        overhead_time,
                        None,
                        None,
                    )
                )
                return result
            # Flush to database every 100 calls
            if len(self.function_calls_data) > 100:
                self.write_function_timings()

            # Add to the list of function calls with pickled args, to be used for replay tests
            self._thread_local.active_functions.remove(func_id)
            overhead_time = time.thread_time_ns() - end_time
            self.function_calls_data.append(
                (
                    func.__name__,
                    class_name,
                    func.__module__,
                    func.__code__.co_filename,
                    benchmark_function_name,
                    benchmark_module_path,
                    benchmark_line_number,
                    execution_time,
                    overhead_time,
                    pickled_args,
                    pickled_kwargs,
                )
            )
            return result

        return wrapper


# Create a singleton instance
codeflash_trace = CodeflashTrace()
