import functools
import pickle
import sqlite3
import time
import os

def codeflash_trace(output_file: str):
    """A decorator factory that returns a decorator that measures the execution time
    of a function and pickles its arguments using the highest protocol available.

    Args:
        output_file: Path to the SQLite database file where results will be stored

    Returns:
        The decorator function

    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Measure execution time
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # Calculate execution time
            execution_time = end_time - start_time

            # Measure overhead
            overhead_start_time = time.time()

            try:
                # Connect to the database
                con = sqlite3.connect(output_file)
                cur = con.cursor()
                cur.execute("PRAGMA synchronous = OFF")

                # Check if table exists and create it if it doesn't
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS function_calls(function_name TEXT, class_name TEXT, file_name TEXT, benchmark_function_name TEXT, benchmark_file_name TEXT,"
                    "time_ns INTEGER, args BLOB, kwargs BLOB)"
                )

                # Pickle the arguments
                pickled_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_kwargs = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)

                # Get benchmark info from environment
                benchmark_function_name = os.environ.get("CODEFLASH_BENCHMARK_FUNCTION_NAME")
                benchmark_file_name = os.environ.get("CODEFLASH_BENCHMARK_FILE_NAME")
                # Insert the data
                cur.execute(
                    "INSERT INTO function_calls (function_name, classname, filename, benchmark_function_name, benchmark_file_name, time_ns, args, kwargs) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (func.__name__, func.__module__, func.__code__.co_filename,
                     execution_time, pickled_args, pickled_kwargs)
                )

                # Commit and close
                con.commit()
                con.close()

                overhead_end_time = time.time()

                print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute")
                print(f"Function '{func.__name__}' overhead took {overhead_end_time - overhead_start_time:.6f} seconds to execute")

            except Exception as e:
                print(f"Error in codeflash_trace: {e}")

            return result
        return wrapper
    return decorator
