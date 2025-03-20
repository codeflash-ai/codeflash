import functools
import os
import pickle
import time
from typing import Callable




class CodeflashTrace:
    """A class that provides both a decorator for tracing function calls
    and a context manager for managing the tracing data lifecycle.
    """

    def __init__(self) -> None:
        self.function_calls_data = []

    # def __enter__(self) -> None:
    #     # Initialize for context manager use
    #     self.function_calls_data = []
    #     return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Cleanup is optional here
        pass

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
            start_time = time.perf_counter_ns()
            result = func(*args, **kwargs)
            end_time = time.perf_counter_ns()

            # Calculate execution time
            execution_time = end_time - start_time

            # Measure overhead
            overhead_start_time = time.perf_counter_ns()

            try:
                # Check if currently in pytest benchmark fixture
                if os.environ.get("CODEFLASH_BENCHMARKING", "False") == "False":
                    return result

                # Pickle the arguments
                pickled_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_kwargs = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)

                # Get benchmark info from environment
                benchmark_function_name = os.environ.get("CODEFLASH_BENCHMARK_FUNCTION_NAME", "")
                benchmark_file_name = os.environ.get("CODEFLASH_BENCHMARK_FILE_NAME", "")
                benchmark_line_number = os.environ.get("CODEFLASH_BENCHMARK_LINE_NUMBER", "")
                # Get class name
                class_name = ""
                qualname = func.__qualname__
                if "." in qualname:
                    class_name = qualname.split(".")[0]
                # Calculate overhead time
                overhead_end_time = time.perf_counter_ns()
                overhead_time = overhead_end_time - overhead_start_time


                self.function_calls_data.append(
                    (func.__name__, class_name, func.__module__, func.__code__.co_filename,
                     benchmark_function_name, benchmark_file_name, benchmark_line_number, execution_time,
                     overhead_time, pickled_args, pickled_kwargs)
                )

            except Exception as e:
                print(f"Error in codeflash_trace: {e}")

            return result
        return wrapper

# Create a singleton instance
codeflash_trace = CodeflashTrace()
