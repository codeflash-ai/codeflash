from __future__ import annotations

from pathlib import Path

from pydantic.dataclasses import dataclass

from codeflash.models.models import BenchmarkDetail, ConcurrencyMetrics, TestResults
from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.result.critic import AcceptanceReason, concurrency_gain, throughput_gain


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Explanation:
    raw_explanation_message: str
    winning_behavior_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    original_runtime_ns: int
    best_runtime_ns: int
    function_name: str
    file_path: Path
    benchmark_details: list[BenchmarkDetail] | None = None
    original_async_throughput: int | None = None
    best_async_throughput: int | None = None
    original_concurrency_metrics: ConcurrencyMetrics | None = None
    best_concurrency_metrics: ConcurrencyMetrics | None = None
    acceptance_reason: AcceptanceReason = AcceptanceReason.RUNTIME

    @property
    def perf_improvement_line(self) -> str:
        improvement_type = {
            AcceptanceReason.RUNTIME: "runtime",
            AcceptanceReason.THROUGHPUT: "throughput",
            AcceptanceReason.CONCURRENCY: "concurrency",
            AcceptanceReason.NONE: "",
        }.get(self.acceptance_reason, "")

        if improvement_type:
            return f"{self.speedup_pct} {improvement_type} improvement ({self.speedup_x} faster)."
        return f"{self.speedup_pct} improvement ({self.speedup_x} faster)."

    @property
    def speedup(self) -> float:
        """Returns the improvement value for the metric that caused acceptance."""
        if (
            self.acceptance_reason == AcceptanceReason.CONCURRENCY
            and self.original_concurrency_metrics
            and self.best_concurrency_metrics
        ):
            return concurrency_gain(self.original_concurrency_metrics, self.best_concurrency_metrics)

        if (
            self.acceptance_reason == AcceptanceReason.THROUGHPUT
            and self.original_async_throughput is not None
            and self.best_async_throughput is not None
            and self.original_async_throughput > 0
        ):
            return throughput_gain(
                original_throughput=self.original_async_throughput, optimized_throughput=self.best_async_throughput
            )

        return (self.original_runtime_ns / self.best_runtime_ns) - 1

    @property
    def speedup_x(self) -> str:
        return f"{self.speedup:,.2f}x"

    @property
    def speedup_pct(self) -> str:
        return f"{self.speedup * 100:,.0f}%"

    def __str__(self) -> str:
        original_runtime_human = humanize_runtime(self.original_runtime_ns)
        best_runtime_human = humanize_runtime(self.best_runtime_ns)

        # Determine if we're showing throughput or runtime improvements
        benchmark_info = ""

        if self.benchmark_details:
            headers = ["Benchmark Module Path", "Test Function", "Original Runtime", "Expected New Runtime", "Speedup"]
            rows = []
            for detail in self.benchmark_details:
                rows.append(
                    [
                        detail.benchmark_name,
                        detail.test_function,
                        detail.original_timing,
                        detail.expected_new_timing,
                        f"{detail.speedup_percent:.2f}%",
                    ]
                )
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(cell))
            fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
            lines = [
                "Benchmark Performance Details",
                fmt.format(*headers),
                "-" * sum([*col_widths, 2 * (len(headers) - 1)]),
            ]
            for row in rows:
                lines.append(fmt.format(*row))
            benchmark_info = "\n".join(lines) + "\n\n"

        if (
            self.acceptance_reason == AcceptanceReason.CONCURRENCY
            and self.original_concurrency_metrics
            and self.best_concurrency_metrics
        ):
            orig_ratio = self.original_concurrency_metrics.concurrency_ratio
            best_ratio = self.best_concurrency_metrics.concurrency_ratio
            performance_description = (
                f"Concurrency ratio improved from {orig_ratio:.2f}x to {best_ratio:.2f}x "
                f"(concurrent execution now {best_ratio:.2f}x faster than sequential)\n\n"
            )
        elif (
            self.acceptance_reason == AcceptanceReason.THROUGHPUT
            and self.original_async_throughput is not None
            and self.best_async_throughput is not None
        ):
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
            + "The new optimized code was tested for correctness. The results are listed below.\n"
            f"{TestResults.report_to_string(self.winning_behavior_test_results.get_test_pass_fail_report_by_type())}\n"
        )

    def explanation_message(self) -> str:
        return self.raw_explanation_message
