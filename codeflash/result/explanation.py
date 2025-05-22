from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path
from typing import Optional, cast

from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.models import BenchmarkDetail, TestResults


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Explanation:
    raw_explanation_message: str
    winning_behavioral_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    original_runtime_ns: int
    best_runtime_ns: int
    function_name: str
    file_path: Path
    benchmark_details: Optional[list[BenchmarkDetail]] = None

    @property
    def perf_improvement_line(self) -> str:
        return f"{self.speedup_pct} improvement ({self.speedup_x} faster)."

    @property
    def speedup(self) -> float:
        return (self.original_runtime_ns / self.best_runtime_ns) - 1

    @property
    def speedup_x(self) -> str:
        return f"{self.speedup:,.2f}x"

    @property
    def speedup_pct(self) -> str:
        return f"{self.speedup * 100:,.0f}%"

    def to_console_string(self) -> str:
        # TODO: After doing the best optimization, remove the test cases that errored on the new code, because they might be failing because of syntax errors and such.
        # TODO: Sometimes the explanation says something similar to "This is the code that was optimized", remove such parts
        original_runtime_human = humanize_runtime(self.original_runtime_ns)
        best_runtime_human = humanize_runtime(self.best_runtime_ns)
        benchmark_info = ""

        if self.benchmark_details:
            # Get terminal width (or use a reasonable default if detection fails)
            try:
                terminal_width = int(shutil.get_terminal_size().columns * 0.9)
            except Exception:
                terminal_width = 200  # Fallback width

            # Create a rich table for better formatting
            table = Table(title="Benchmark Performance Details", width=terminal_width, show_lines=True)

            # Add columns - split Benchmark File and Function into separate columns
            # Using proportional width for benchmark file column (40% of terminal width)
            benchmark_col_width = max(int(terminal_width * 0.4), 40)
            table.add_column("Benchmark Module Path", style="cyan", width=benchmark_col_width, overflow="fold")
            table.add_column("Test Function", style="cyan", overflow="fold")
            table.add_column("Original Runtime", style="magenta", justify="right")
            table.add_column("Expected New Runtime", style="green", justify="right")
            table.add_column("Speedup", style="red", justify="right")

            # Add rows with split data
            for detail in self.benchmark_details:
                # Split the benchmark name and test function
                benchmark_name = detail.benchmark_name
                test_function = detail.test_function

                table.add_row(
                    benchmark_name,
                    test_function,
                    f"{detail.original_timing}",
                    f"{detail.expected_new_timing}",
                    f"{detail.speedup_percent:.2f}%",
                )
            # Convert table to string
            string_buffer = StringIO()
            console = Console(file=string_buffer, width=terminal_width)
            console.print(table)
            benchmark_info = cast("StringIO", console.file).getvalue() + "\n"  # Cast for mypy

        return (
            f"Optimized {self.function_name} in {self.file_path}\n"
            f"{self.perf_improvement_line}\n"
            f"Runtime went down from {original_runtime_human} to {best_runtime_human} \n\n"
            + (benchmark_info if benchmark_info else "")
            + self.raw_explanation_message
            + " \n\n"
            + "The new optimized code was tested for correctness. The results are listed below.\n"
            + f"{TestResults.report_to_string(self.winning_behavioral_test_results.get_test_pass_fail_report_by_type())}\n"
        )

    def explanation_message(self) -> str:
        return self.raw_explanation_message
