import functools
import os
import pickle
import sqlite3
import time
from pathlib import Path
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
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            # Calculate execution time
            execution_time = end_time - start_time

            # Measure overhead
            overhead_start_time = time.time()
            overhead_time = 0

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

                # Calculate overhead time
                overhead_end_time = time.time()
                overhead_time = overhead_end_time - overhead_start_time

                class_name = ""
                qualname = func.__qualname__
                if "." in qualname:
                    class_name = qualname.split(".")[0]
                self.function_calls_data.append(
                    (func.__name__, class_name, func.__module__, func.__code__.co_filename,
                     benchmark_function_name, benchmark_file_name, benchmark_line_number, execution_time,
                     overhead_time, pickled_args, pickled_kwargs)
                )

            except Exception as e:
                print(f"Error in codeflash_trace: {e}")

            return result
        return wrapper

    def write_to_db(self, output_file: str) -> None:
        """Write all collected function call data to the SQLite database.

        Args:
            output_file: Path to the SQLite database file where results will be stored

        """
        if not self.function_calls_data:
            print("No function call data to write")
            return
        self.db_path = output_file
        try:
            # Connect to the database
            con = sqlite3.connect(output_file)
            cur = con.cursor()
            cur.execute("PRAGMA synchronous = OFF")

            # Check if table exists and create it if it doesn't
            cur.execute(
                "CREATE TABLE IF NOT EXISTS function_calls("
                "function_name TEXT, class_name TEXT,  module_name TEXT, file_name TEXT,"
                "benchmark_function_name TEXT, benchmark_file_name TEXT, benchmark_line_number INTEGER,"
                "time_ns INTEGER, overhead_time_ns INTEGER, args BLOB, kwargs BLOB)"
            )

            # Insert all data at once
            cur.executemany(
                "INSERT INTO function_calls "
                "(function_name, class_name, module_name, file_name, benchmark_function_name, "
                "benchmark_file_name, benchmark_line_number, time_ns, overhead_time_ns, args, kwargs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self.function_calls_data
            )

            # Commit and close
            con.commit()
            con.close()

            print(f"Successfully wrote {len(self.function_calls_data)} function call records to {output_file}")

            # Clear the data after writing
            self.function_calls_data.clear()

        except Exception as e:
            print(f"Error writing function calls to database: {e}")

    def print_codeflash_db(self, limit: int = None) -> None:
        """
        Print the contents of a CodeflashTrace SQLite database.

        Args:
            db_path: Path to the SQLite database file
            limit: Maximum number of records to print (None for all)
        """
        try:
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get the count of records
            cursor.execute("SELECT COUNT(*) FROM function_calls")
            total_records = cursor.fetchone()[0]
            print(f"Found {total_records} function call records in {self.db_path}")

            # Build the query with optional limit
            query = "SELECT * FROM function_calls"
            if limit:
                query += f" LIMIT {limit}"

            # Execute the query
            cursor.execute(query)

            # Print column names
            columns = [desc[0] for desc in cursor.description]
            print("\nColumns:", columns)
            print("\n" + "=" * 80 + "\n")

            # Print each row
            for i, row in enumerate(cursor.fetchall()):
                print(f"Record #{i + 1}:")
                print(f"  Function: {row[0]}")
                print(f"  Class: {row[1]}")
                print(f"  Module: {row[2]}")
                print(f"  File: {row[3]}")
                print(f"  Benchmark Function: {row[4] or 'N/A'}")
                print(f"  Benchmark File: {row[5] or 'N/A'}")
                print(f"  Benchmark Line: {row[6] or 'N/A'}")
                print(f"  Execution Time: {row[7]:.6f} seconds")
                print(f"  Overhead Time: {row[8]:.6f} seconds")

                # Unpickle and print args and kwargs
                try:
                    args = pickle.loads(row[9])
                    kwargs = pickle.loads(row[10])

                    print(f"  Args: {args}")
                    print(f"  Kwargs: {kwargs}")
                except Exception as e:
                    print(f"  Error unpickling args/kwargs: {e}")
                    print(f"  Raw args: {row[8]}")
                    print(f"  Raw kwargs: {row[9]}")

                print("\n" + "-" * 40 + "\n")

            conn.close()

        except Exception as e:
            print(f"Error reading database: {e}")


# Create a singleton instance
codeflash_trace = CodeflashTrace()
