from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, cast

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from rich.tree import Tree

from codeflash.cli_cmds.console import logger
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
    def id(self) -> str:
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
    loop_index: int  # The loop index of the function invocation, starts at 1
    id: InvocationId  # The fully qualified name of the function invocation (id)
    file_name: Path  # The file where the test is defined
    did_pass: bool  # Whether the test this function invocation was part of, passed or failed
    runtime: Optional[int]  # Time in nanoseconds
    test_framework: str  # unittest or pytest
    test_type: TestType
    return_value: Optional[object]  # The return value of the function invocation
    timed_out: Optional[bool]

    @property
    def unique_invocation_loop_id(self) -> str:
        return f"{self.loop_index}:{self.id.id()}"


class TestResults(BaseModel):
    # don't modify these directly, use the add method
    # also we don't support deletion of test results elements - caution is advised
    test_results: list[FunctionTestInvocation] = []
    test_result_idx: dict[str, int] = {}

    def add(self, function_test_invocation: FunctionTestInvocation) -> None:
        if function_test_invocation.unique_invocation_loop_id in self.test_result_idx:
            logger.warning(
                f"Test result with id {function_test_invocation.unique_invocation_loop_id} already exists. SKIPPING"
            )
            return
        self.test_result_idx[function_test_invocation.unique_invocation_loop_id] = len(self.test_results)
        self.test_results.append(function_test_invocation)

    def merge(self, other: TestResults) -> None:
        original_len = len(self.test_results)
        self.test_results.extend(other.test_results)
        for k, v in other.test_result_idx.items():
            if k in self.test_result_idx:
                raise ValueError(f"Test result with id {k} already exists.")
            self.test_result_idx[k] = v + original_len

    def get_by_unique_invocation_loop_id(self, unique_invocation_loop_id: str) -> FunctionTestInvocation | None:
        try:
            return self.test_results[self.test_result_idx[unique_invocation_loop_id]]
        except (IndexError, KeyError):
            return None

    def get_all_ids(self) -> set[InvocationId]:
        return {test_result.id for test_result in self.test_results}

    def get_all_unique_invocation_loop_ids(self) -> set[str]:
        return {test_result.unique_invocation_loop_id for test_result in self.test_results}

    def number_of_loops(self) -> int:
        if not self.test_results:
            return 0
        return max(test_result.loop_index for test_result in self.test_results)

    def get_test_pass_fail_report_by_type(self) -> dict[TestType, dict[str, int]]:
        report = {}
        for test_type in TestType:
            report[test_type] = {"passed": 0, "failed": 0}
        for test_result in self.test_results:
            if test_result.loop_index == 1:
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
            ]
        )

    @staticmethod
    def report_to_tree(report: dict[TestType, dict[str, int]], title: str) -> Tree:
        tree = Tree(title)
        for test_type in TestType:
            tree.add(
                f"{test_type.to_name()} - Passed: {report[test_type]['passed']}, Failed: {report[test_type]['failed']}"
            )
        return tree

    def total_passed_runtime(self) -> int:
        """Calculate the sum of runtimes of all test cases that passed, where a testcase runtime
        is the minimum value of all looped execution runtimes.

        :return: The runtime in nanoseconds.
        """
        for result in self.test_results:
            if result.did_pass and not result.runtime:
                logger.debug(
                    f"Ignoring test case that passed but had no runtime -> {result.id}, Loop # {result.loop_index}"
                )
        usable_results = [result for result in self.test_results if result.did_pass and result.runtime]
        return sum(
            [
                min([result.runtime for result in usable_results if result.id == invocation_id])
                for invocation_id in {result.id for result in usable_results}
            ]
        )

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
        if type(self) is not type(other):
            return False
        if len(self) != len(other):
            return False
        original_recursion_limit = sys.getrecursionlimit()
        cast(TestResults, other)
        for test_result in self:
            other_test_result = other.get_by_unique_invocation_loop_id(test_result.unique_invocation_loop_id)
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
                or not comparator(test_result.return_value, other_test_result.return_value)
            ):
                sys.setrecursionlimit(original_recursion_limit)
                return False
        sys.setrecursionlimit(original_recursion_limit)
        return True
