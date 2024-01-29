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
    path: str

    @property
    def speedup(self) -> float:
        return (self.original_runtime_ns / self.best_runtime_ns) - 1

    def to_console_string(self) -> str:
        # TODO: After doing the best optimization, remove the test cases that errored on the new code, because they might be failing because of syntax errors and such.
        # TODO: Sometimes the explanation says something similar to "This is the code that was optimized", remove such parts
        original_runtime_human = humanize_runtime(self.original_runtime_ns)
        best_runtime_human = humanize_runtime(self.best_runtime_ns)

        explanation = (
            f"Function {self.function_name} in file {self.path}:\n"
            f"Performance went up by {self.speedup:.2f}x ({self.speedup * 100:.2f}%). Runtime went down from {original_runtime_human} to {best_runtime_human} \n\n"
            + "Optimization explanation:\n"
            + self.raw_explanation_message
            + " \n\n"
            + "The code has been tested for correctness.\n"
            + f"Test Results for the best optimized code:- {TestResults.report_to_string(self.winning_test_results.get_test_pass_fail_report_by_type())}\n"
        )

        return explanation
