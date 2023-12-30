from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.verification.test_results import TestResults


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class PrComment:
    optimization_explanation: str
    best_runtime: int
    function_name: str
    original_runtime: int
    file_path: str
    speedup: float
    winning_test_results: TestResults

    def to_json(self) -> dict[str, str | dict[str, dict[str, int]]]:
        return {
            "optimization_explanation": self.optimization_explanation,
            "best_runtime": f"{(self.best_runtime / 1000):.2f}",
            "original_runtime": f"{(self.original_runtime / 1000):.2f}",
            "function_name": self.function_name,
            "file_path": self.file_path,
            "speedup_x": f"{self.speedup:.2f}",
            "speedup_pct": f"{self.speedup * 100:.2f}",
            "report_table": {
                test_type.to_name(): result
                for test_type, result in self.winning_test_results.get_test_pass_fail_report_by_type().items()
            },
        }


class FileDiffContent(BaseModel):
    old_content: str
    new_content: str
