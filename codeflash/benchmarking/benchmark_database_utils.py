import sqlite3
from pathlib import Path

import pickle


class BenchmarkDatabaseUtils:
    def __init__(self, trace_path :Path) -> None:
        self.trace_path = trace_path
        self.connection = None

    def setup(self) -> None:
        try:
            # Open connection
            self.connection = sqlite3.connect(self.trace_path)
            cur = self.connection.cursor()
            cur.execute("PRAGMA synchronous = OFF")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS function_calls("
                "function_name TEXT, class_name TEXT, module_name TEXT, file_name TEXT,"
                "benchmark_function_name TEXT, benchmark_file_name TEXT, benchmark_line_number INTEGER,"
                "time_ns INTEGER, overhead_time_ns INTEGER, args BLOB, kwargs BLOB)"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS benchmark_timings("
                "benchmark_file_name TEXT, benchmark_function_name TEXT, benchmark_line_number INTEGER,"
                "time_ns INTEGER)"  # Added closing parenthesis
            )
            self.connection.commit()
            # Don't close the connection here
        except Exception as e:
            print(f"Database setup error: {e}")
            if self.connection:
                self.connection.close()
                self.connection = None
            raise

    def write_function_timings(self, data: list[tuple]) -> None:
        if not self.connection:
            self.connection = sqlite3.connect(self.trace_path)

        try:
            cur = self.connection.cursor()
            # Insert data into the function_calls table
            cur.executemany(
                "INSERT INTO function_calls "
                "(function_name, class_name, module_name, file_name, benchmark_function_name, "
                "benchmark_file_name, benchmark_line_number, time_ns, overhead_time_ns, args, kwargs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data
            )
            self.connection.commit()
        except Exception as e:
            print(f"Error writing to function timings database: {e}")
            self.connection.rollback()
            raise

    def write_benchmark_timings(self, data: list[tuple]) -> None:
        if not self.connection:
            self.connection = sqlite3.connect(self.trace_path)

        try:
            cur = self.connection.cursor()
            # Insert data into the benchmark_timings table
            cur.executemany(
                "INSERT INTO benchmark_timings (benchmark_file_name, benchmark_function_name, benchmark_line_number, time_ns) VALUES (?, ?, ?, ?)",
                data
            )
            self.connection.commit()
        except Exception as e:
            print(f"Error writing to benchmark timings database: {e}")
            self.connection.rollback()
            raise

    def print_function_timings(self, limit: int = None) -> None:
        """Print the contents of a CodeflashTrace SQLite database.

        Args:
            limit: Maximum number of records to print (None for all)
        """
        if not self.connection:
            self.connection = sqlite3.connect(self.trace_path)
        try:
            cur = self.connection.cursor()

            # Get the count of records
            cur.execute("SELECT COUNT(*) FROM function_calls")
            total_records = cur.fetchone()[0]
            print(f"Found {total_records} function call records in {self.trace_path}")

            # Build the query with optional limit
            query = "SELECT * FROM function_calls"
            if limit:
                query += f" LIMIT {limit}"

            # Execute the query
            cur.execute(query)

            # Print column names
            columns = [desc[0] for desc in cur.description]
            print("\nColumns:", columns)
            print("\n" + "=" * 80 + "\n")

            # Print each row
            for i, row in enumerate(cur.fetchall()):
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
                    print(f"  Raw args: {row[9]}")
                    print(f"  Raw kwargs: {row[10]}")

                print("\n" + "-" * 40 + "\n")

        except Exception as e:
            print(f"Error reading database: {e}")

    def print_benchmark_timings(self, limit: int = None) -> None:
        """Print the contents of a CodeflashTrace SQLite database.
        Args:
            limit: Maximum number of records to print (None for all)
            """
        if not self.connection:
            self.connection = sqlite3.connect(self.trace_path)
        try:
            cur = self.connection.cursor()

            # Get the count of records
            cur.execute("SELECT COUNT(*) FROM benchmark_timings")
            total_records = cur.fetchone()[0]
            print(f"Found {total_records} benchmark timing records in {self.trace_path}")

            # Build the query with optional limit
            query = "SELECT * FROM benchmark_timings"
            if limit:
                query += f" LIMIT {limit}"

            # Execute the query
            cur.execute(query)

            # Print column names
            columns = [desc[0] for desc in cur.description]
            print("\nColumns:", columns)
            print("\n" + "=" * 80 + "\n")

            # Print each row
            for i, row in enumerate(cur.fetchall()):
                print(f"Record #{i + 1}:")
                print(f"  Benchmark File: {row[0] or 'N/A'}")
                print(f"  Benchmark Function: {row[1] or 'N/A'}")
                print(f"  Benchmark Line: {row[2] or 'N/A'}")
                print(f"  Execution Time: {row[3] / 1e9:.6f} seconds")  # Convert nanoseconds to seconds
                print("\n" + "-" * 40 + "\n")

        except Exception as e:
            print(f"Error reading benchmark timings database: {e}")


    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None


    @staticmethod
    def get_function_benchmark_timings(trace_path: Path) -> dict[str, dict[str, int]]:
        """Process the trace file and extract timing data for all functions.

        Args:
            trace_path: Path to the trace file

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
                "benchmark_file_name, benchmark_function_name, benchmark_line_number, time_ns "
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

    @staticmethod
    def get_benchmark_timings(trace_path: Path) -> dict[str, int]:
        """Extract total benchmark timings from trace files.

        Args:
            trace_path: Path to the trace file

        Returns:
            A dictionary mapping where:
            - Keys are benchmark filename :: benchmark test function :: line number
            - Values are total benchmark timing in milliseconds (with overhead subtracted)

        """
        # Initialize the result dictionary
        result = {}
        overhead_by_benchmark = {}

        # Connect to the SQLite database
        connection = sqlite3.connect(trace_path)
        cursor = connection.cursor()

        try:
            # Query the function_calls table to get total overhead for each benchmark
            cursor.execute(
                "SELECT benchmark_file_name, benchmark_function_name, benchmark_line_number, SUM(overhead_time_ns) "
                "FROM function_calls "
                "GROUP BY benchmark_file_name, benchmark_function_name, benchmark_line_number"
            )

            # Process overhead information
            for row in cursor.fetchall():
                benchmark_file, benchmark_func, benchmark_line, total_overhead_ns = row
                benchmark_key = f"{benchmark_file}::{benchmark_func}::{benchmark_line}"
                overhead_by_benchmark[benchmark_key] = total_overhead_ns or 0  # Handle NULL sum case

            # Query the benchmark_timings table for total times
            cursor.execute(
                "SELECT benchmark_file_name, benchmark_function_name, benchmark_line_number, time_ns "
                "FROM benchmark_timings"
            )

            # Process each row and subtract overhead
            for row in cursor.fetchall():
                benchmark_file, benchmark_func, benchmark_line, time_ns = row

                # Create the benchmark key (file::function::line)
                benchmark_key = f"{benchmark_file}::{benchmark_func}::{benchmark_line}"

                # Subtract overhead from total time
                overhead = overhead_by_benchmark.get(benchmark_key, 0)
                result[benchmark_key] = time_ns - overhead

        finally:
            # Close the connection
            connection.close()

        return result
