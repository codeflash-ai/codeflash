from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

from pydantic.dataclasses import dataclass

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.models import BenchmarkDetail, TestResults
from codeflash.verification.test_results import TestResults


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
            benchmark_info += "Benchmark Performance Details:\n"
            for detail in self.benchmark_details:
                benchmark_info += f"Original timing for {detail.benchmark_name}::{detail.test_function}: {detail.original_timing}\n"
                benchmark_info += f"Expected new timing for {detail.benchmark_name}::{detail.test_function}: {detail.expected_new_timing}\n"
                benchmark_info += f"Benchmark speedup for {detail.benchmark_name}::{detail.test_function}: {detail.speedup_percent:.2f}%\n\n"

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
