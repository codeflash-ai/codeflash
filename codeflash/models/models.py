from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich.tree import Tree

from codeflash.cli_cmds.console import DEBUG_MODE

if TYPE_CHECKING:
    from collections.abc import Iterator
import enum
import re
import sys
from collections.abc import Collection
from enum import Enum, IntEnum
from pathlib import Path
from re import Pattern
from typing import Annotated, Optional, cast

from jedi.api.classes import Name
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import module_name_from_file_path, validate_python_code
from codeflash.code_utils.env_utils import is_end_to_end
from codeflash.verification.comparator import comparator

# If the method spam is in the class Ham, which is at the top level of the module eggs in the package foo, the fully
# qualified name of the method is foo.eggs.Ham.spam, its qualified name is Ham.spam, and its name is spam. The full name
# of the module is foo.eggs.


class ValidCode(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_code: str
    normalized_code: str


# TODO COVER FIX
class CoverReturnCode(IntEnum):
    DID_NOT_RUN = -1
    NO_DIFFERENCES = 0
    COUNTER_EXAMPLES = 1
    ERROR = 2


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FunctionSource:
    file_path: Path
    qualified_name: str
    fully_qualified_name: str
    only_function_name: str
    source_code: str
    jedi_definition: Name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionSource):
            return False
        return (
            self.file_path == other.file_path
            and self.qualified_name == other.qualified_name
            and self.fully_qualified_name == other.fully_qualified_name
            and self.only_function_name == other.only_function_name
            and self.source_code == other.source_code
        )

    def __hash__(self) -> int:
        return hash(
            (self.file_path, self.qualified_name, self.fully_qualified_name, self.only_function_name, self.source_code)
        )


class BestOptimization(BaseModel):
    candidate: OptimizedCandidate
    helper_functions: list[FunctionSource]
    runtime: int
    replay_performance_gain: Optional[dict[BenchmarkKey, float]] = None
    winning_behavior_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    winning_replay_benchmarking_test_results: Optional[TestResults] = None


@dataclass(frozen=True)
class BenchmarkKey:
    module_path: str
    function_name: str

    def __str__(self) -> str:
        return f"{self.module_path}::{self.function_name}"


@dataclass
class BenchmarkDetail:
    benchmark_name: str
    test_function: str
    original_timing: str
    expected_new_timing: str
    speedup_percent: float

    def to_string(self) -> str:
        return (
            f"Original timing for {self.benchmark_name}::{self.test_function}: {self.original_timing}\n"
            f"Expected new timing for {self.benchmark_name}::{self.test_function}: {self.expected_new_timing}\n"
            f"Benchmark speedup for {self.benchmark_name}::{self.test_function}: {self.speedup_percent:.2f}%\n"
        )

    def to_dict(self) -> dict[str, any]:
        return {
            "benchmark_name": self.benchmark_name,
            "test_function": self.test_function,
            "original_timing": self.original_timing,
            "expected_new_timing": self.expected_new_timing,
            "speedup_percent": self.speedup_percent,
        }


@dataclass
class ProcessedBenchmarkInfo:
    benchmark_details: list[BenchmarkDetail]

    def to_string(self) -> str:
        if not self.benchmark_details:
            return ""

        result = "Benchmark Performance Details:\n"
        for detail in self.benchmark_details:
            result += detail.to_string() + "\n"
        return result

    def to_dict(self) -> dict[str, list[dict[str, any]]]:
        return {"benchmark_details": [detail.to_dict() for detail in self.benchmark_details]}


class CodeString(BaseModel):
    code: Annotated[str, AfterValidator(validate_python_code)]
    file_path: Optional[Path] = None


class CodeStringsMarkdown(BaseModel):
    code_strings: list[CodeString] = []

    @property
    def markdown(self) -> str:
        """Returns the markdown representation of the code, including the file path where possible."""
        return "\n".join(
            [
                f"```python{':' + str(code_string.file_path) if code_string.file_path else ''}\n{code_string.code.strip()}\n```"
                for code_string in self.code_strings
            ]
        )


class CodeOptimizationContext(BaseModel):
    testgen_context_code: str = ""
    read_writable_code: str = Field(min_length=1)
    read_only_context_code: str = ""
    hashing_code_context: str = ""
    hashing_code_context_hash: str = ""
    helper_functions: list[FunctionSource]
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]]


class CodeContextType(str, Enum):
    READ_WRITABLE = "READ_WRITABLE"
    READ_ONLY = "READ_ONLY"
    TESTGEN = "TESTGEN"
    HASHING = "HASHING"


class OptimizedCandidateResult(BaseModel):
    max_loop_count: int
    best_test_runtime: int
    behavior_test_results: TestResults
    benchmarking_test_results: TestResults
    replay_benchmarking_test_results: Optional[dict[BenchmarkKey, TestResults]] = None
    optimization_candidate_index: int
    total_candidate_timing: int


class GeneratedTests(BaseModel):
    generated_original_test_source: str
    instrumented_behavior_test_source: str
    instrumented_perf_test_source: str
    behavior_file_path: Path
    perf_file_path: Path


class GeneratedTestsList(BaseModel):
    generated_tests: list[GeneratedTests]


class TestFile(BaseModel):
    instrumented_behavior_file_path: Path
    benchmarking_file_path: Path = None
    original_file_path: Optional[Path] = None
    original_source: Optional[str] = None
    test_type: TestType
    tests_in_file: Optional[list[TestsInFile]] = None


class TestFiles(BaseModel):
    test_files: list[TestFile]

    def get_by_type(self, test_type: TestType) -> TestFiles:
        return TestFiles(test_files=[test_file for test_file in self.test_files if test_file.test_type == test_type])

    def add(self, test_file: TestFile) -> None:
        if test_file not in self.test_files:
            self.test_files.append(test_file)
        else:
            msg = "Test file already exists in the list"
            raise ValueError(msg)

    def get_by_original_file_path(self, file_path: Path) -> TestFile | None:
        return next((test_file for test_file in self.test_files if test_file.original_file_path == file_path), None)

    def get_test_type_by_instrumented_file_path(self, file_path: Path) -> TestType | None:
        return next(
            (
                test_file.test_type
                for test_file in self.test_files
                if (file_path in (test_file.instrumented_behavior_file_path, test_file.benchmarking_file_path))
            ),
            None,
        )

    def get_test_type_by_original_file_path(self, file_path: Path) -> TestType | None:
        return next(
            (test_file.test_type for test_file in self.test_files if test_file.original_file_path == file_path), None
        )

    def __iter__(self) -> Iterator[TestFile]:
        return iter(self.test_files)

    def __len__(self) -> int:
        return len(self.test_files)


class OptimizationSet(BaseModel):
    control: list[OptimizedCandidate]
    experiment: Optional[list[OptimizedCandidate]]


@dataclass(frozen=True)
class TestsInFile:
    test_file: Path
    test_class: Optional[str]
    test_function: str
    test_type: TestType


@dataclass(frozen=True)
class OptimizedCandidate:
    source_code: str
    explanation: str
    optimization_id: str


@dataclass(frozen=True)
class FunctionCalledInTest:
    tests_in_file: TestsInFile
    position: CodePosition


@dataclass(frozen=True)
class CodePosition:
    line_no: int
    col_no: int


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str


class OriginalCodeBaseline(BaseModel):
    behavior_test_results: TestResults
    benchmarking_test_results: TestResults
    replay_benchmarking_test_results: Optional[dict[BenchmarkKey, TestResults]] = None
    line_profile_results: dict
    runtime: int
    coverage_results: Optional[CoverageData]


class CoverageStatus(Enum):
    NOT_FOUND = "Coverage Data Not Found"
    PARSED_SUCCESSFULLY = "Parsed Successfully"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CoverageData:
    """Represents the coverage data for a specific function in a source file, using one or more test files."""

    file_path: Path
    coverage: float
    function_name: str
    functions_being_tested: list[str]
    graph: dict[str, dict[str, Collection[object]]]
    code_context: CodeOptimizationContext
    main_func_coverage: FunctionCoverage
    dependent_func_coverage: Optional[FunctionCoverage]
    status: CoverageStatus
    blank_re: Pattern[str] = re.compile(r"\s*(#|$)")
    else_re: Pattern[str] = re.compile(r"\s*else\s*:\s*(#|$)")

    def build_message(self) -> str:
        if self.status == CoverageStatus.NOT_FOUND:
            return f"No coverage data found for {self.function_name}"
        return f"{self.coverage:.1f}%"

    def log_coverage(self) -> None:
        from rich.tree import Tree

        tree = Tree("Test Coverage Results")
        tree.add(f"Main Function: {self.main_func_coverage.name}: {self.coverage:.2f}%")
        if self.dependent_func_coverage:
            tree.add(
                f"Dependent Function: {self.dependent_func_coverage.name}: {self.dependent_func_coverage.coverage:.2f}%"
            )
        tree.add(f"Total Coverage: {self.coverage:.2f}%")
        console.print(tree)
        console.rule()

        if not self.coverage:
            logger.debug(self.graph)
        if is_end_to_end():
            console.print(self)

    @classmethod
    def create_empty(cls, file_path: Path, function_name: str, code_context: CodeOptimizationContext) -> CoverageData:
        return cls(
            file_path=file_path,
            coverage=0.0,
            function_name=function_name,
            functions_being_tested=[function_name],
            graph={
                function_name: {
                    "executed_lines": set(),
                    "unexecuted_lines": set(),
                    "executed_branches": [],
                    "unexecuted_branches": [],
                }
            },
            code_context=code_context,
            main_func_coverage=FunctionCoverage(
                name=function_name,
                coverage=0.0,
                executed_lines=[],
                unexecuted_lines=[],
                executed_branches=[],
                unexecuted_branches=[],
            ),
            dependent_func_coverage=None,
            status=CoverageStatus.NOT_FOUND,
        )


@dataclass
class FunctionCoverage:
    """Represents the coverage data for a specific function in a source file."""

    name: str
    coverage: float
    executed_lines: list[int]
    unexecuted_lines: list[int]
    executed_branches: list[list[int]]
    unexecuted_branches: list[list[int]]


class TestingMode(enum.Enum):
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"
    LINE_PROFILE = "line_profile"


# TODO this class is duplicated in codeflash_capture
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
        if self is TestType.INIT_STATE_TEST:
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
    def from_str_id(string_id: str, iteration_id: str | None = None) -> InvocationId:
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
    stdout: Optional[str] = None

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

    def group_by_benchmarks(
        self, benchmark_keys: list[BenchmarkKey], benchmark_replay_test_dir: Path, project_root: Path
    ) -> dict[BenchmarkKey, TestResults]:
        """Group TestResults by benchmark for calculating improvements for each benchmark."""
        test_results_by_benchmark = defaultdict(TestResults)
        benchmark_module_path = {}
        for benchmark_key in benchmark_keys:
            benchmark_module_path[benchmark_key] = module_name_from_file_path(
                benchmark_replay_test_dir.resolve()
                / f"test_{benchmark_key.module_path.replace('.', '_')}__replay_test_",
                project_root,
            )
        for test_result in self.test_results:
            if test_result.test_type == TestType.REPLAY_TEST:
                for benchmark_key, module_path in benchmark_module_path.items():
                    if test_result.id.test_module_path.startswith(module_path):
                        test_results_by_benchmark[benchmark_key].add(test_result)

        return test_results_by_benchmark

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
            if test_type is TestType.INIT_STATE_TEST:
                continue
            tree.add(
                f"{test_type.to_name()} - Passed: {report[test_type]['passed']}, Failed: {report[test_type]['failed']}"
            )
        return tree

    def usable_runtime_data_by_test_case(self) -> dict[InvocationId, list[int]]:
        # Efficient single traversal, directly accumulating into a dict.
        # can track mins here and only sums can be return in total_passed_runtime
        by_id: dict[InvocationId, list[int]] = {}
        for result in self.test_results:
            if result.did_pass:
                if result.runtime:
                    by_id.setdefault(result.id, []).append(result.runtime)
                else:
                    msg = (
                        f"Ignoring test case that passed but had no runtime -> {result.id}, "
                        f"Loop # {result.loop_index}, Test Type: {result.test_type}, "
                        f"Verification Type: {result.verification_type}"
                    )
                    logger.debug(msg)
        return by_id

    def total_passed_runtime(self) -> int:
        """Calculate the sum of runtimes of all test cases that passed.

        A testcase runtime is the minimum value of all looped execution runtimes.

        :return: The runtime in nanoseconds.
        """
        # TODO this doesn't look at the intersection of tests of baseline and original
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
        cast("TestResults", other)
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
