import functools
import os
import pickle
import sqlite3
import sys
import time
from typing import Callable

import dill


class CodeflashTrace:
    """Decorator class that traces and profiles function execution."""

    def __init__(self, trace_path: str = None) -> None:
        self.function_calls_data = []
        self.function_call_count = 0
        self.pickle_count_limit = 1000
        self._connection = None
        self._trace_path = trace_path
        if self._trace_path:
            self._initialize_db_connection()
        self.cur = None
        if self._connection:
            self.cur = self._connection.cursor()

    def setup(self, trace_path: str) -> None:
        """Set up the database connection for direct writing.

        Args:
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
        """Write function call data directly to the database."""
        if self._connection is None or self.cur is None:
            return  # No connection to write data

        self._write_batch_and_clear()

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
            func: The function to be decorated

        Returns:
            The wrapped function

        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Measure execution time
            start_time = time.thread_time_ns()
            result = func(*args, **kwargs)
            end_time = time.thread_time_ns()
            # Calculate execution time
            execution_time = end_time - start_time

            self.function_call_count += 1

            # Measure overhead
            original_recursion_limit = sys.getrecursionlimit()
            # Check if currently in pytest benchmark fixture
            if os.environ.get("CODEFLASH_BENCHMARKING", "False") == "False":
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

            if self.function_call_count <= self.pickle_count_limit:
                try:
                    sys.setrecursionlimit(1000000)
                    args = dict(args.items())
                    if class_name and func.__name__ == "__init__" and "self" in args:
                        del args["self"]
                    # Pickle the arguments
                    pickled_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                    pickled_kwargs = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
                    sys.setrecursionlimit(original_recursion_limit)
                except (TypeError, pickle.PicklingError, AttributeError, RecursionError, OSError):
                    # we retry with dill if pickle fails. It's slower but more comprehensive
                    try:
                        pickled_args = dill.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                        pickled_kwargs = dill.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
                        sys.setrecursionlimit(original_recursion_limit)

                    except (TypeError, dill.PicklingError, AttributeError, RecursionError, OSError) as e:
                        print(f"Error pickling arguments for function {func.__name__}: {e}")
                        return None

            if len(self.function_calls_data) > 1000:
                self.write_function_timings()
            # Calculate overhead time
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

    def _initialize_db_connection(self):
        if self._connection is None and self._trace_path is not None:
            self._connection = sqlite3.connect(self._trace_path)

    def _pickle_args_kwargs(self, args, kwargs):
        try:
            pickled_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
            pickled_kwargs = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
        except (TypeError, pickle.PicklingError, AttributeError, RecursionError, OSError):
            try:
                pickled_args = dill.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_kwargs = dill.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
            except (TypeError, dill.PicklingError, AttributeError, RecursionError, OSError) as e:
                print(f"Error pickling arguments: {e}")
                return None, None
        return pickled_args, pickled_kwargs

    def _write_batch_and_clear(self):
        if not self.function_calls_data:
            return  # No data to write
        try:
            self.cur.executemany(
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


# Create a singleton instance
codeflash_trace = CodeflashTrace()
