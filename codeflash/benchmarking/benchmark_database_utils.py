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

