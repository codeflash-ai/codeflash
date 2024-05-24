from typing import Union

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.verification.test_results import TestResults


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class PrComment:
    optimization_explanation: str
    best_runtime: int
    original_runtime: int
    function_name: str
    relative_file_path: str
    speedup_x: str
    speedup_pct: str
    winning_test_results: TestResults

    def to_json(self) -> dict[str, Union[str, dict[str, dict[str, int]]]]:
        return {
            "optimization_explanation": self.optimization_explanation,
            "best_runtime": humanize_runtime(self.best_runtime),
            "original_runtime": humanize_runtime(self.original_runtime),
            "function_name": self.function_name,
            "file_path": self.relative_file_path,
            "speedup_x": self.speedup_x,
            "speedup_pct": self.speedup_pct,
            "report_table": {
                test_type.to_name(): result
                for test_type, result in self.winning_test_results.get_test_pass_fail_report_by_type().items()
            },
        }


class FileDiffContent(BaseModel):
    oldContent: str
    newContent: str
