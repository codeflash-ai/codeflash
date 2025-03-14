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
                # Pickle the arguments
                pickled_args = pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL)
                pickled_kwargs = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)

                # Get benchmark info from environment
                benchmark_function_name = os.environ.get("CODEFLASH_BENCHMARK_FUNCTION_NAME", "")
                benchmark_file_name = os.environ.get("CODEFLASH_BENCHMARK_FILE_NAME", "")

                # Calculate overhead time
                overhead_end_time = time.time()
                overhead_time = overhead_end_time - overhead_start_time

                class_name = ""
                qualname = func.__qualname__
                if "." in qualname:
                    class_name = qualname.split(".")[0]
                self.function_calls_data.append(
                    (func.__name__, class_name, func.__module__, func.__code__.co_filename,
                     benchmark_function_name, benchmark_file_name, execution_time,
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
                "benchmark_function_name TEXT, benchmark_file_name TEXT, "
                "time_ns INTEGER, overhead_time_ns INTEGER, args BLOB, kwargs BLOB)"
            )

            # Insert all data at once
            cur.executemany(
                "INSERT INTO function_calls "
                "(function_name, class_name, module_name, file_name, benchmark_function_name, "
                "benchmark_file_name, time_ns, overhead_time_ns, args, kwargs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                print(f"  Execution Time: {row[6]:.6f} seconds")
                print(f"  Overhead Time: {row[7]:.6f} seconds")

                # Unpickle and print args and kwargs
                try:
                    args = pickle.loads(row[8])
                    kwargs = pickle.loads(row[9])

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

    def generate_replay_test(self, output_dir: str = None, project_root: str = "", test_framework: str = "pytest",
                             max_run_count: int = 100) -> None:
        """
        Generate multiple replay tests from the traced function calls, grouping by benchmark name.

        Args:
            output_dir: Directory to write the generated tests (if None, only returns the code)
            project_root: Root directory of the project for module imports
            test_framework: 'pytest' or 'unittest'
            max_run_count: Maximum number of runs to include per function

        Returns:
            Dictionary mapping benchmark names to generated test code
        """
        import isort
        from codeflash.verification.verification_utils import get_test_file_path

        if not self.db_path:
            print("No database path set. Call write_to_db first or set db_path manually.")
            return {}

        try:
            # Import the function here to avoid circular imports
            from codeflash.benchmarking.replay_test import create_trace_replay_test

            print("connecting to: ", self.db_path)
            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get distinct benchmark names
            cursor.execute(
                "SELECT DISTINCT benchmark_function_name, benchmark_file_name FROM function_calls"
            )
            benchmarks = cursor.fetchall()

            # Generate a test for each benchmark
            for benchmark in benchmarks:
                benchmark_function_name, benchmark_file_name = benchmark
                # Get functions associated with this benchmark
                cursor.execute(
                    "SELECT DISTINCT function_name, class_name, module_name, file_name FROM function_calls "
                    "WHERE benchmark_function_name = ? AND benchmark_file_name = ?",
                    (benchmark_function_name, benchmark_file_name)
                )

                functions_data = []
                for func_row in cursor.fetchall():
                    function_name, class_name, module_name, file_name = func_row

                    # Add this function to our list
                    functions_data.append({
                        "function_name": function_name,
                        "class_name": class_name,
                        "file_name": file_name,
                        "module_name": module_name
                    })

                if not functions_data:
                    print(f"No functions found for benchmark {benchmark_function_name} in {benchmark_file_name}")
                    continue

                # Generate the test code for this benchmark
                test_code = create_trace_replay_test(
                    trace_file=self.db_path,
                    functions_data=functions_data,
                    test_framework=test_framework,
                    max_run_count=max_run_count,
                )
                test_code = isort.code(test_code)

                # Write to file if requested
                if output_dir:
                    output_file = get_test_file_path(
                        test_dir=Path(output_dir), function_name=f"{benchmark_file_name[5:]}_{benchmark_function_name}", test_type="replay"
                    )
                    with open(output_file, 'w') as f:
                        f.write(test_code)
                    print(f"Replay test for benchmark `{benchmark_function_name}` in {benchmark_file_name} written to {output_file}")

            conn.close()

        except Exception as e:
            print(f"Error generating replay tests: {e}")

# Create a singleton instance
codeflash_trace = CodeflashTrace()
