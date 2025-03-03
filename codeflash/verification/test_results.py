from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from rich.tree import Tree

from codeflash.cli_cmds.console import DEBUG_MODE, logger
from codeflash.verification.comparator import comparator

if TYPE_CHECKING:
    from collections.abc import Iterator


class VerificationType(str, Enum):
    FUNCTION_CALL = (
        "function_call"  # Correctness verification for a test function, checks input values and output values)
    )
    INIT_STATE_FTO = "init_state_fto"  # Correctness verification for fto class instance attributes after init
    INIT_STATE_HELPER = "init_state_helper"  # Correctness verification for helper class instance attributes after init


class TestType(Enum):
    EXISTING_UNIT_TEST = 1
    INSPIRED_REGRESSION = 2
    GENERATED_REGRESSION = 3
    REPLAY_TEST = 4
    CONCOLIC_COVERAGE_TEST = 5
    INIT_STATE_TEST = 6

    def to_name(self) -> str:
        if self == TestType.INIT_STATE_TEST:
            return ""
        names = {
            TestType.EXISTING_UNIT_TEST: "⚙️ Existing Unit Tests",
            TestType.INSPIRED_REGRESSION: "🎨 Inspired Regression Tests",
            TestType.GENERATED_REGRESSION: "🌀 Generated Regression Tests",
            TestType.REPLAY_TEST: "⏪ Replay Tests",
            TestType.CONCOLIC_COVERAGE_TEST: "🔎 Concolic Coverage Tests",
        }
        return names[self]


@dataclass(frozen=True)
class InvocationId:
    test_module_path: str  # The fully qualified name of the test module
    test_class_name: Optional[str]  # The name of the class where the test is defined
    test_function_name: Optional[str]  # The name of the test_function. Does not include the components of the file_name
    function_getting_tested: str
    iteration_id: Optional[str]

    # test_module_path:TestSuiteClass.test_function_name:function_tested:iteration_id
    def id(self) -> str:
        class_prefix = f"{self.test_class_name}." if self.test_class_name else ""
        return (
            f"{self.test_module_path}:{class_prefix}{self.test_function_name}:"
            f"{self.function_getting_tested}:{self.iteration_id}"
        )

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
    verification_type: Optional[str] = VerificationType.FUNCTION_CALL

    @property
    def unique_invocation_loop_id(self) -> str:
        return f"{self.loop_index}:{self.id.id()}"


class TestResults(BaseModel):
    # don't modify these directly, use the add method
    # also we don't support deletion of test results elements - caution is advised
    test_results: list[FunctionTestInvocation] = []
    test_result_idx: dict[str, int] = {}

    def add(self, function_test_invocation: FunctionTestInvocation) -> None:
        unique_id = function_test_invocation.unique_invocation_loop_id
        if unique_id in self.test_result_idx:
            if DEBUG_MODE:
                logger.warning(f"Test result with id {unique_id} already exists. SKIPPING")
            return
        self.test_result_idx[unique_id] = len(self.test_results)
        self.test_results.append(function_test_invocation)

    def merge(self, other: TestResults) -> None:
        original_len = len(self.test_results)
        self.test_results.extend(other.test_results)
        for k, v in other.test_result_idx.items():
            if k in self.test_result_idx:
                msg = f"Test result with id {k} already exists."
                raise ValueError(msg)
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

    def usable_runtime_data_by_test_case(self) -> dict[InvocationId, list[int]]:
        for result in self.test_results:
            if result.did_pass and not result.runtime:
                msg = (
                    f"Ignoring test case that passed but had no runtime -> {result.id}, "
                    f"Loop # {result.loop_index}, Test Type: {result.test_type}, "
                    f"Verification Type: {result.verification_type}"
                )
                logger.debug(msg)

        usable_runtimes = [
            (result.id, result.runtime) for result in self.test_results if result.did_pass and result.runtime
        ]
        return {
            usable_id: [runtime[1] for runtime in usable_runtimes if runtime[0] == usable_id]
            for usable_id in {runtime[0] for runtime in usable_runtimes}
        }

    def total_passed_runtime(self) -> int:
        """Calculate the sum of runtimes of all test cases that passed.

        A testcase runtime is the minimum value of all looped execution runtimes.

        :return: The runtime in nanoseconds.
        """
        return sum(
            [min(usable_runtime_data) for _, usable_runtime_data in self.usable_runtime_data_by_test_case().items()]
        )

    def __iter__(self) -> Iterator[FunctionTestInvocation]:
        return iter(self.test_results)

    def __len__(self) -> int:
        return len(self.test_results)

    def __getitem__(self, index: int) -> FunctionTestInvocation:
        return self.test_results[index]

    def __setitem__(self, index: int, value: FunctionTestInvocation) -> None:
        self.test_results[index] = value

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
