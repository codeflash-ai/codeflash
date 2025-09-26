from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path
from typing import Optional, cast

from pydantic.dataclasses import dataclass
from rich.console import Console
from rich.table import Table

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.models.models import BenchmarkDetail, TestResults
from codeflash.result.critic import throughput_gain


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Explanation:
    raw_explanation_message: str
    winning_behavior_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    original_runtime_ns: int
    best_runtime_ns: int
    function_name: str
    file_path: Path
    benchmark_details: Optional[list[BenchmarkDetail]] = None
    original_async_throughput: Optional[int] = None
    best_async_throughput: Optional[int] = None

    @property
    def perf_improvement_line(self) -> str:
        runtime_improvement = self.speedup

        if (
            self.original_async_throughput is not None
            and self.best_async_throughput is not None
            and self.original_async_throughput > 0
        ):
            throughput_improvement = throughput_gain(
                original_throughput=self.original_async_throughput, optimized_throughput=self.best_async_throughput
            )

            # Use throughput metrics if throughput improvement is better or runtime got worse
            if throughput_improvement > runtime_improvement or runtime_improvement <= 0:
                throughput_pct = f"{throughput_improvement * 100:,.0f}%"
                throughput_x = f"{throughput_improvement + 1:,.2f}x"
                return f"{throughput_pct} improvement ({throughput_x} faster)."

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

    def __str__(self) -> str:
        # TODO: After doing the best optimization, remove the test cases that errored on the new code, because they might be failing because of syntax errors and such.
        # TODO: Sometimes the explanation says something similar to "This is the code that was optimized", remove such parts
        original_runtime_human = humanize_runtime(self.original_runtime_ns)
        best_runtime_human = humanize_runtime(self.best_runtime_ns)

        # Determine if we're showing throughput or runtime improvements
        runtime_improvement = self.speedup
        is_using_throughput_metric = False

        if (
            self.original_async_throughput is not None
            and self.best_async_throughput is not None
            and self.original_async_throughput > 0
        ):
            throughput_improvement = throughput_gain(
                original_throughput=self.original_async_throughput, optimized_throughput=self.best_async_throughput
            )

            if throughput_improvement > runtime_improvement or runtime_improvement <= 0:
                is_using_throughput_metric = True

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

        if is_using_throughput_metric:
            performance_description = (
                f"Throughput improved from {self.original_async_throughput} to {self.best_async_throughput} operations/second "
                f"(runtime: {original_runtime_human} → {best_runtime_human})\n\n"
            )
        else:
            performance_description = f"Runtime went down from {original_runtime_human} to {best_runtime_human} \n\n"

        return (
            f"Optimized {self.function_name} in {self.file_path}\n"
            f"{self.perf_improvement_line}\n"
            + performance_description
            + (benchmark_info if benchmark_info else "")
            + self.raw_explanation_message
            + " \n\n"
            + (
                # in the lsp (extension) we display the test results before the optimization summary
                ""
                if is_LSP_enabled()
                else "The new optimized code was tested for correctness. The results are listed below.\n"
                f"{TestResults.report_to_string(self.winning_behavior_test_results.get_test_pass_fail_report_by_type())}\n"
            )
        )

    def explanation_message(self) -> str:
        return self.raw_explanation_message
