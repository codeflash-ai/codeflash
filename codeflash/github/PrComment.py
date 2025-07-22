from __future__ import annotations  # noqa: N999

from typing import Optional, Union

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.models import BenchmarkDetail, TestResults


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class PrComment:
    optimization_explanation: str
    best_runtime: int
    original_runtime: int
    function_name: str
    relative_file_path: str
    speedup_x: str
    speedup_pct: str
    winning_behavior_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    benchmark_details: Optional[list[BenchmarkDetail]] = None

    def to_json(self) -> dict[str, Union[dict[str, dict[str, int]], int, str, Optional[list[BenchmarkDetail]]]]:
        report_table = {
            test_type.to_name(): result
            for test_type, result in self.winning_behavior_test_results.get_test_pass_fail_report_by_type().items()
            if test_type.to_name()
        }

        return {
            "optimization_explanation": self.optimization_explanation,
            "best_runtime": humanize_runtime(self.best_runtime),
            "original_runtime": humanize_runtime(self.original_runtime),
            "function_name": self.function_name,
            "file_path": self.relative_file_path,
            "speedup_x": self.speedup_x,
            "speedup_pct": self.speedup_pct,
            "loop_count": self.winning_benchmarking_test_results.number_of_loops(),
            "report_table": report_table,
            "benchmark_details": self.benchmark_details if self.benchmark_details else None,
        }


class FileDiffContent(BaseModel):
    oldContent: str  # noqa: N815
    newContent: str  # noqa: N815
