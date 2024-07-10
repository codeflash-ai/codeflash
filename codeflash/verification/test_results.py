from __future__ import annotations

import logging
import sys
from enum import Enum
from typing import Iterator, Optional

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.verification.comparator import comparator


class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4

    def to_name(self) -> str:
        names = {
            TestType.EXISTING_UNIT_TEST: "⚙️ Existing Unit Tests",
            TestType.INSPIRED_REGRESSION: "🎨 Inspired Regression Tests",
            TestType.GENERATED_REGRESSION: "🌀 Generated Regression Tests",
            TestType.REPLAY_TEST: "⏪ Replay Tests",
        }
        return names[self]


@dataclass(frozen=True)
class InvocationId:
    test_module_path: str  # The fully qualified name of the test module
    test_class_name: Optional[str]  # The name of the class where the test is defined
    test_function_name: str  # The name of the test_function. Does not include the components of the file_name
    function_getting_tested: str
    iteration_id: Optional[str]

    # test_module_path:TestSuiteClass.test_function_name:function_tested:iteration_id
    def id(self):
        return f"{self.test_module_path}:{(self.test_class_name + '.' if self.test_class_name else '')}{self.test_function_name}:{self.function_getting_tested}:{self.iteration_id}"

    @staticmethod
    def from_str_id(string_id: str, iteration_id: Optional[str] = None) -> InvocationId:
        components = string_id.split(":")
        assert len(components) == 4
        second_components = components[1].split(".")
        if len(second_components) == 1:
            test_class_name = None
            test_function_name = second_components[0]
        else:
            test_class_name = second_components[0]
            test_function_name = second_components[1]
        return InvocationId(
            test_module_path=components[0],
            test_class_name=test_class_name,
            test_function_name=test_function_name,
            function_getting_tested=components[2],
            iteration_id=iteration_id if iteration_id else components[3],
        )


@dataclass(frozen=True)
class FunctionTestInvocation:
    id: InvocationId  # The fully qualified name of the function invocation (id)
    file_name: str  # The file where the test is defined
    did_pass: bool  # Whether the test this function invocation was part of, passed or failed
    runtime: Optional[int]  # Time in nanoseconds
    test_framework: str  # unittest or pytest
    test_type: TestType
    return_value: Optional[object]  # The return value of the function invocation
    timed_out: Optional[bool]

    def test_executed(self) -> bool:
        return self.test_type != TestType.EXISTING_UNIT_TEST or self.id.function_getting_tested


class TestResults(BaseModel):
    test_results: list[FunctionTestInvocation] = []

    def add(self, function_test_invocation: FunctionTestInvocation) -> None:
        self.test_results.append(function_test_invocation)

    def merge(self, other: TestResults) -> None:
        self.test_results.extend(other.test_results)

    def get_by_id(
        self,
        invocation_id: InvocationId,
    ) -> FunctionTestInvocation | None:
        return next((r for r in self.test_results if r.id == invocation_id), None)

    def get_all_ids(self) -> list[InvocationId]:
        return [test_result.id for test_result in self.test_results]

    def get_test_pass_fail_report(self) -> str:
        passed = 0
        failed = 0
        for test_result in self.test_results:
            if test_result.test_executed():
                if test_result.did_pass:
                    passed += 1
                else:
                    logging.info(f"Failed test: {test_result.id}")
                    failed += 1
        return f"Passed: {passed}, Failed: {failed}"

    def get_test_pass_fail_report_by_type(self) -> dict[TestType, dict[str, int]]:
        report = {}
        for test_type in TestType:
            report[test_type] = {"passed": 0, "failed": 0}
        for test_result in self.test_results:
            if test_result.test_executed():
                if test_result.did_pass:
                    report[test_result.test_type]["passed"] += 1
                else:
                    report[test_result.test_type]["failed"] += 1
        return report

    @staticmethod
    def report_to_string(report: dict[TestType, dict[str, int]]) -> str:
        return " ".join(
            [
                f"{test_type.to_name()}- (Passed: {report[test_type]['passed']}, Failed: {report[test_type]['failed']})"
                for test_type in TestType
            ],
        )

    def total_passed_runtime(self) -> int:
        for result in self.test_results:
            if result.did_pass and result.runtime is None:
                logging.debug(
                    f"Ignoring test case that passed but had no runtime -> {result.id}",
                )
        timing = sum(
            [
                result.runtime
                for result in self.test_results
                if (result.did_pass and result.runtime is not None)
            ],
        )
        return timing

    def __iter__(self) -> Iterator[FunctionTestInvocation]:
        return iter(self.test_results)

    def __len__(self) -> int:
        return len(self.test_results)

    def __getitem__(self, index: int) -> FunctionTestInvocation:
        return self.test_results[index]

    def __setitem__(self, index: int, value: FunctionTestInvocation) -> None:
        self.test_results[index] = value

    def __delitem__(self, index: int) -> None:
        del self.test_results[index]

    def __contains__(self, value: FunctionTestInvocation) -> bool:
        return value in self.test_results

    def __bool__(self) -> bool:
        return bool(self.test_results)

    def __eq__(self, other: object) -> bool:
        # Unordered comparison
        if type(self) != type(other):
            return False
        if len(self) != len(other):
            return False
        original_recursion_limit = sys.getrecursionlimit()
        for test_result in self:
            other_test_result = other.get_by_id(test_result.id)
            if other_test_result is None:
                return False

            if original_recursion_limit < 5000:
                sys.setrecursionlimit(5000)
            if (
                test_result.file_name != other_test_result.file_name
                or test_result.did_pass != other_test_result.did_pass
                or test_result.runtime != other_test_result.runtime
                or test_result.test_framework != other_test_result.test_framework
                or test_result.test_type != other_test_result.test_type
                or not comparator(
                    test_result.return_value,
                    other_test_result.return_value,
                )
            ):
                sys.setrecursionlimit(original_recursion_limit)
                return False
        sys.setrecursionlimit(original_recursion_limit)
        return True
