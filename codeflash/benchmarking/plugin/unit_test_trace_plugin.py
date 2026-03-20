from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from codeflash.benchmarking.codeflash_trace import codeflash_trace
from codeflash.code_utils.code_utils import module_name_from_file_path


class UnitTestTracePlugin:
    def __init__(self) -> None:
        self._trace_path: str | None = None
        self._connection: sqlite3.Connection | None = None
        self.project_root: str | None = None
        self.benchmark_timings: list[tuple[str, str, int, int]] = []

    def setup(self, trace_path: str, project_root: str) -> None:
        try:
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
            self.close()
        except Exception as e:
            print(f"Database setup error: {e}")
            if self._connection:
                self._connection.close()
                self._connection = None
            raise

    def write_benchmark_timings(self) -> None:
        if not self.benchmark_timings:
            return

        if self._connection is None:
            self._connection = sqlite3.connect(self._trace_path)

        try:
            cur = self._connection.cursor()
            cur.executemany(
                "INSERT INTO benchmark_timings (benchmark_module_path, benchmark_function_name, benchmark_line_number, benchmark_time_ns) VALUES (?, ?, ?, ?)",
                self.benchmark_timings,
            )
            self._connection.commit()
            self.benchmark_timings = []
        except Exception as e:
            print(f"Error writing to benchmark timings database: {e}")
            self._connection.rollback()
            raise

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> None:
        node_path = getattr(item, "path", None) or getattr(item, "fspath", None)
        if node_path is None:
            yield
            return

        benchmark_module_path = module_name_from_file_path(
            Path(str(node_path)), Path(self.project_root), traverse_up=True
        )
        benchmark_function_name = item.name
        line_number = item.reportinfo()[1] or 0

        os.environ["CODEFLASH_BENCHMARK_FUNCTION_NAME"] = benchmark_function_name
        os.environ["CODEFLASH_BENCHMARK_MODULE_PATH"] = benchmark_module_path
        os.environ["CODEFLASH_BENCHMARK_LINE_NUMBER"] = str(line_number)
        os.environ["CODEFLASH_BENCHMARKING"] = "True"

        start = time.perf_counter_ns()
        yield
        end = time.perf_counter_ns()

        os.environ["CODEFLASH_BENCHMARKING"] = "False"

        codeflash_trace.write_function_timings()
        codeflash_trace.function_call_count = 0

        self.benchmark_timings.append((benchmark_module_path, benchmark_function_name, line_number, end - start))

    @pytest.hookimpl
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        codeflash_trace.close()
        if self.benchmark_timings:
            self.write_benchmark_timings()
        self.close()


unit_test_trace_plugin = UnitTestTracePlugin()
