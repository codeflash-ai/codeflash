from pydantic.dataclasses import dataclass

from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.verification.test_results import TestResults


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Explanation:
    raw_explanation_message: str
    winning_test_results: TestResults
    original_runtime_ns: int
    best_runtime_ns: int
    function_name: str
    file_path: str

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

        explanation = (
            f"Optimized {self.function_name} in {self.file_path}\n"
            f"{self.perf_improvement_line}\n"
            f"Runtime went down from {original_runtime_human} to {best_runtime_human} \n\n"
            + "Explanation:\n"
            + self.raw_explanation_message
            + " \n\n"
            + "The new optimized code was tested for correctness. The results are listed below.\n"
            + f"{TestResults.report_to_string(self.winning_test_results.get_test_pass_fail_report_by_type())}\n"
        )

        return explanation

    def explanation_message(self) -> str:
        return self.raw_explanation_message
