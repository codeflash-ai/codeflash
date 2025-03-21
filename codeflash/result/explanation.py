from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

from pydantic.dataclasses import dataclass

from codeflash.code_utils.time_utils import humanize_runtime
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
    replay_performance_gain: Optional[float]
    fto_benchmark_timings: Optional[dict[str, int]]
    total_benchmark_timings: Optional[dict[str, int]]

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
        if self.replay_performance_gain and self.fto_benchmark_timings and self.total_benchmark_timings:
            benchmark_info += "Benchmark Performance Details:\n"
            for benchmark_key, og_benchmark_timing in self.fto_benchmark_timings.items():
                # benchmark key is benchmark filename :: benchmark test function :: line number
                try:
                    benchmark_file_name, benchmark_test_function, line_number = benchmark_key.split("::")
                except ValueError:
                    benchmark_info += f"Benchmark key {benchmark_key} is not in the expected format.\n"
                    continue

                total_benchmark_timing = self.total_benchmark_timings[benchmark_key]
                # find out expected new benchmark timing, then calculate how much total benchmark was sped up. print out intermediate values
                benchmark_info += f"Original timing for {benchmark_file_name}::{benchmark_test_function}: {humanize_runtime(total_benchmark_timing)}\n"
                replay_speedup = self.replay_performance_gain
                expected_new_benchmark_timing = total_benchmark_timing - og_benchmark_timing + 1 / (
                        replay_speedup + 1) * og_benchmark_timing
                benchmark_info += f"Expected new timing for {benchmark_file_name}::{benchmark_test_function}: {humanize_runtime(int(expected_new_benchmark_timing))}\n"

                benchmark_speedup_ratio = total_benchmark_timing / expected_new_benchmark_timing
                benchmark_speedup_percent = (benchmark_speedup_ratio - 1) * 100
                benchmark_info += f"Benchmark speedup for {benchmark_file_name}::{benchmark_test_function}: {benchmark_speedup_percent:.2f}%\n\n"

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
