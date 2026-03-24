from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Collection
from functools import lru_cache
from re import Pattern
from typing import TYPE_CHECKING

from codeflash_core.models import FunctionParent
from codeflash_python.models.test_type import TestType

if TYPE_CHECKING:
    from collections.abc import Iterator

import enum
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError, model_validator
from pydantic.dataclasses import dataclass

from codeflash_python.code_utils.code_utils import module_name_from_file_path, validate_python_code

logger = logging.getLogger("codeflash_python")

DEBUG_MODE = os.environ.get("CODEFLASH_DEBUG", "").lower() in ("1", "true")


# If the method spam is in the class Ham, which is at the top level of the module eggs in the package foo, the fully
# qualified name of the method is foo.eggs.Ham.spam, its qualified name is Ham.spam, and its name is spam. The full name
# of the module is foo.eggs.


class ValidCode(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_code: str
    normalized_code: str


@dataclass(frozen=True)
class FunctionSource:
    file_path: Path
    qualified_name: str
    fully_qualified_name: str
    only_function_name: str
    source_code: str
    definition_type: str | None = None  # e.g. "function", "class"; None for non-Python languages

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
    explanation_v2: str | None = None
    helper_functions: list[FunctionSource]
    code_context: CodeOptimizationContext
    runtime: int
    replay_performance_gain: dict[BenchmarkKey, float] | None = None
    winning_behavior_test_results: TestResults
    winning_benchmarking_test_results: TestResults
    winning_replay_benchmarking_test_results: TestResults | None = None
    line_profiler_test_results: dict[Any, Any]
    async_throughput: int | None = None
    concurrency_metrics: ConcurrencyMetrics | None = None


@dataclass(frozen=True)
class BenchmarkKey:
    module_path: str
    function_name: str

    def __str__(self) -> str:
        return f"{self.module_path}::{self.function_name}"


@dataclass
class ConcurrencyMetrics:
    sequential_time_ns: int
    concurrent_time_ns: int
    concurrency_factor: int
    concurrency_ratio: float  # sequential_time / concurrent_time


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

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"benchmark_details": [detail.to_dict() for detail in self.benchmark_details]}


class CodeString(BaseModel):
    code: str
    file_path: Path | None = None
    language: str = "python"  # Language for validation

    @model_validator(mode="after")
    def validate_code_syntax(self) -> CodeString:
        """Validate code syntax for the specified language."""
        if self.language == "python":
            validate_python_code(self.code)
        else:
            try:
                compile(self.code, "<string>", "exec")
            except SyntaxError:
                msg = f"Invalid {self.language.title()} code"
                raise ValueError(msg) from None
        return self


def get_comment_prefix(file_path: Path) -> str:
    """Get the comment prefix for a given language."""
    return "#"


def get_code_block_splitter(file_path: Path | None) -> str:
    if file_path is None:
        return ""
    comment_prefix = get_comment_prefix(file_path)
    return f"{comment_prefix} file: {file_path.as_posix()}"


# Pattern to match markdown code blocks with optional language tag and file path
# Matches: ```language:filepath\ncode\n``` or ```language\ncode\n```
markdown_pattern = re.compile(r"```(\w+)(?::([^\n]+))?\n(.*?)\n```", re.DOTALL)


class CodeStringsMarkdown(BaseModel):
    code_strings: list[CodeString] = []
    language: str = "python"  # Language for markdown code block tags
    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def flat(self) -> str:
        """Returns the combined source code module from all code blocks.

        Each block is prefixed by a file path comment to indicate its origin.
        The comment prefix is determined by the language attribute.

        Returns:
            str: The concatenated code of all blocks with file path annotations.

        !! Important !!:
        Avoid parsing the flat code with multiple files,
        parsing may result in unexpected behavior.


        """
        if self._cache.get("flat") is not None:
            return self._cache["flat"]
        self._cache["flat"] = "\n".join(
            get_code_block_splitter(block.file_path) + "\n" + block.code for block in self.code_strings
        )
        return self._cache["flat"]

    @property
    def markdown(self) -> str:
        """Returns a Markdown-formatted string containing all code blocks.

        Each block is enclosed in a triple-backtick code block with an optional
        file path suffix (e.g., ```python:filename.py).

        The language tag is determined by the `language` attribute.

        Returns:
            str: Markdown representation of the code blocks.

        """
        return "\n".join(
            [
                f"```{self.language}{':' + code_string.file_path.as_posix() if code_string.file_path else ''}\n{code_string.code.strip()}\n```"
                for code_string in self.code_strings
            ]
        )

    def file_to_path(self) -> dict[str, str]:
        """Return a dictionary mapping file paths to their corresponding code blocks.

        Returns:
            dict[str, str]: Mapping from file path (as string) to code.

        """
        if self._cache.get("file_to_path") is not None:
            return self._cache["file_to_path"]
        self._cache["file_to_path"] = {
            str(code_string.file_path): code_string.code for code_string in self.code_strings
        }
        return self._cache["file_to_path"]

    @staticmethod
    def parse_markdown_code(markdown_code: str, expected_language: str = "python") -> CodeStringsMarkdown:
        """Parse a Markdown string into a CodeStringsMarkdown object.

        Extracts code blocks and their associated file paths and constructs a new CodeStringsMarkdown instance.

        Args:
            markdown_code (str): The Markdown-formatted string to parse.
            expected_language (str): The expected language of code blocks (default: "python").

        Returns:
            CodeStringsMarkdown: Parsed object containing code blocks.

        """
        matches = markdown_pattern.findall(markdown_code)
        code_string_list = []
        detected_language = expected_language
        try:
            for language, file_path, code in matches:
                # Use the first detected language or the expected language
                if language:
                    detected_language = language
                if file_path:
                    path = file_path.strip()
                    code_string_list.append(CodeString(code=code, file_path=Path(path), language=detected_language))
                else:
                    # No file path specified - skip this block or create with None
                    code_string_list.append(CodeString(code=code, file_path=None, language=detected_language))
            return CodeStringsMarkdown(code_strings=code_string_list, language=detected_language)
        except ValidationError:
            # if any file is invalid, return an empty CodeStringsMarkdown for the entire context
            return CodeStringsMarkdown(language=expected_language)


class CodeOptimizationContext(BaseModel):
    testgen_context: CodeStringsMarkdown
    read_writable_code: CodeStringsMarkdown
    read_only_context_code: str = ""
    hashing_code_context: str = ""
    hashing_code_context_hash: str = ""
    helper_functions: list[FunctionSource]
    testgen_helper_fqns: list[str] = []
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]]


class OptimizedCandidateResult(BaseModel):
    max_loop_count: int
    best_test_runtime: int
    behavior_test_results: TestResults
    benchmarking_test_results: TestResults
    replay_benchmarking_test_results: dict[BenchmarkKey, TestResults] | None = None
    optimization_candidate_index: int
    total_candidate_timing: int
    async_throughput: int | None = None
    concurrency_metrics: ConcurrencyMetrics | None = None


class GeneratedTests(BaseModel):
    generated_original_test_source: str
    instrumented_behavior_test_source: str
    instrumented_perf_test_source: str
    raw_generated_test_source: str | None = None
    behavior_file_path: Path
    perf_file_path: Path


class GeneratedTestsList(BaseModel):
    generated_tests: list[GeneratedTests]


class TestFile(BaseModel):
    instrumented_behavior_file_path: Path
    benchmarking_file_path: Path | None = None
    original_file_path: Path | None = None
    original_source: str | None = None
    test_type: TestType
    tests_in_file: list[TestsInFile] | None = None


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
        normalized = self._normalize_path_for_comparison(file_path)
        for test_file in self.test_files:
            if test_file.original_file_path is None:
                continue
            normalized_test_path = self._normalize_path_for_comparison(test_file.original_file_path)
            if normalized == normalized_test_path:
                return test_file
        return None

    def get_test_type_by_instrumented_file_path(self, file_path: Path) -> TestType | None:
        normalized = self._normalize_path_for_comparison(file_path)
        for test_file in self.test_files:
            normalized_behavior_path = self._normalize_path_for_comparison(test_file.instrumented_behavior_file_path)
            if normalized == normalized_behavior_path:
                return test_file.test_type
            if test_file.benchmarking_file_path is not None:
                normalized_benchmark_path = self._normalize_path_for_comparison(test_file.benchmarking_file_path)
                if normalized == normalized_benchmark_path:
                    return test_file.test_type

        # Fallback: try filename-only matching when normalized paths don't match
        file_name = file_path.name
        for test_file in self.test_files:
            if (
                test_file.instrumented_behavior_file_path
                and test_file.instrumented_behavior_file_path.name == file_name
            ):
                return test_file.test_type
            if test_file.benchmarking_file_path and test_file.benchmarking_file_path.name == file_name:
                return test_file.test_type

        return None

    def get_test_type_by_original_file_path(self, file_path: Path) -> TestType | None:
        normalized = self._normalize_path_for_comparison(file_path)
        for test_file in self.test_files:
            if test_file.original_file_path is None:
                continue
            normalized_test_path = self._normalize_path_for_comparison(test_file.original_file_path)
            if normalized == normalized_test_path:
                return test_file.test_type
        return None

    @staticmethod
    @lru_cache(maxsize=4096)
    def _normalize_path_for_comparison(path: Path) -> str:
        """Normalize a path for cross-platform comparison.

        Resolves the path to an absolute path and handles Windows case-insensitivity.
        """
        try:
            resolved = str(path.resolve())
        except (OSError, RuntimeError):
            # If resolve fails (e.g., file doesn't exist), use absolute path
            resolved = str(path.absolute())
        # Only lowercase on Windows where filesystem is case-insensitive
        return resolved.lower() if sys.platform == "win32" else resolved

    def __iter__(self) -> Iterator[TestFile]:  # type: ignore[override]
        return iter(self.test_files)

    def __len__(self) -> int:
        return len(self.test_files)


class OptimizationSet(BaseModel):
    control: list[OptimizedCandidate]
    experiment: list[OptimizedCandidate] | None


@dataclass(frozen=True)
class TestsInFile:
    test_file: Path
    test_class: str | None
    test_function: str
    test_type: TestType


class OptimizedCandidateSource(str, Enum):
    OPTIMIZE = "OPTIMIZE"
    OPTIMIZE_LP = "OPTIMIZE_LP"
    REFINE = "REFINE"
    REPAIR = "REPAIR"
    ADAPTIVE = "ADAPTIVE"
    JIT_REWRITE = "JIT_REWRITE"


@dataclass(frozen=True)
class OptimizedCandidate:
    source_code: CodeStringsMarkdown
    explanation: str
    optimization_id: str
    source: OptimizedCandidateSource
    parent_id: str | None = None
    model: str | None = None  # Which LLM model generated this candidate


@dataclass(frozen=True)
class FunctionCalledInTest:
    tests_in_file: TestsInFile
    position: CodePosition


@dataclass(frozen=True)
class CodePosition:
    line_no: int
    col_no: int


class OriginalCodeBaseline(BaseModel):
    behavior_test_results: TestResults
    benchmarking_test_results: TestResults
    replay_benchmarking_test_results: dict[BenchmarkKey, TestResults] | None = None
    line_profile_results: dict
    runtime: int
    coverage_results: CoverageData | None
    async_throughput: int | None = None
    concurrency_metrics: ConcurrencyMetrics | None = None


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
    dependent_func_coverage: FunctionCoverage | None
    status: CoverageStatus
    blank_re: Pattern[str] = re.compile(r"\s*(#|$)")
    else_re: Pattern[str] = re.compile(r"\s*else\s*:\s*(#|$)")

    def build_message(self) -> str:
        if self.status == CoverageStatus.NOT_FOUND:
            return f"No coverage data found for {self.function_name}"
        return f"{self.coverage:.1f}%"

    def log_coverage(self) -> None:
        lines = ["Test Coverage Results", f"  Main Function: {self.main_func_coverage.name}: {self.coverage:.2f}%"]
        if self.dependent_func_coverage:
            lines.append(
                f"  Dependent Function: {self.dependent_func_coverage.name}: {self.dependent_func_coverage.coverage:.2f}%"
            )
        lines.append(f"  Total Coverage: {self.coverage:.2f}%")
        logger.info("\n".join(lines))

        if not self.coverage:
            logger.debug(self.graph)

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
    CONCURRENCY = "concurrency"


# Intentionally duplicated in codeflash_capture (runs in subprocess, can't import from here)
class VerificationType(str, Enum):
    FUNCTION_CALL = (
        "function_call"  # Correctness verification for a test function, checks input values and output values)
    )
    INIT_STATE_FTO = "init_state_fto"  # Correctness verification for fto class instance attributes after init
    INIT_STATE_HELPER = "init_state_helper"  # Correctness verification for helper class instance attributes after init


@dataclass(frozen=True)
class InvocationId:
    test_module_path: str  # The fully qualified name of the test module
    test_class_name: str | None  # The name of the class where the test is defined
    test_function_name: str | None  # The name of the test_function. Does not include the components of the file_name
    function_getting_tested: str
    iteration_id: str | None

    # test_module_path:TestSuiteClass.test_function_name:function_tested:iteration_id
    def id(self) -> str:
        class_prefix = f"{self.test_class_name}." if self.test_class_name else ""
        return (
            f"{self.test_module_path}:{class_prefix}{self.test_function_name}:"
            f"{self.function_getting_tested}:{self.iteration_id}"
        )

    # TestSuiteClass.test_function_name
    def test_fn_qualified_name(self) -> str:
        # Use f-string with inline conditional to reduce string concatenation operations
        return (
            f"{self.test_class_name}.{self.test_function_name}"
            if self.test_class_name
            else str(self.test_function_name)
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
    runtime: int | None  # Time in nanoseconds
    test_framework: str  # unittest or pytest
    test_type: TestType
    return_value: object | None  # The return value of the function invocation
    timed_out: bool | None
    verification_type: str | None = VerificationType.FUNCTION_CALL
    stdout: str | None = None

    @property
    def unique_invocation_loop_id(self) -> str:
        return f"{self.loop_index}:{self.id.id()}"


class TestResults(BaseModel):  # noqa: PLW1641
    # don't modify these directly, use the add method
    # also we don't support deletion of test results elements - caution is advised
    test_results: list[FunctionTestInvocation] = []
    test_result_idx: dict[str, int] = {}

    perf_stdout: str | None = None
    # mapping between test function name and stdout failure message
    test_failures: dict[str, str] | None = None

    def add(self, function_test_invocation: FunctionTestInvocation) -> None:
        unique_id = function_test_invocation.unique_invocation_loop_id
        test_result_idx = self.test_result_idx
        if unique_id in test_result_idx:
            if DEBUG_MODE:
                logger.warning("Test result with id %s already exists. SKIPPING", unique_id)
            return
        test_results = self.test_results
        test_result_idx[unique_id] = len(test_results)
        test_results.append(function_test_invocation)

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
                traverse_up=True,
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
        report: dict[TestType, dict[str, int]] = {tt: {"passed": 0, "failed": 0} for tt in TestType}
        for test_result in self.test_results:
            if test_result.loop_index != 1:
                continue
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
    def report_to_tree(report: dict[TestType, dict[str, int]], title: str) -> str:
        lines = [title]
        for test_type in TestType:
            if test_type is TestType.INIT_STATE_TEST:
                continue
            lines.append(
                f"  {test_type.to_name()} - Passed: {report[test_type]['passed']}, Failed: {report[test_type]['failed']}"
            )
        return "\n".join(lines)

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

    def effective_loop_count(self) -> int:
        """Calculate the effective number of complete loops.

        Returns the maximum loop_index seen across all test results. This represents
        the number of timing iterations that were performed.

        :return: The effective loop count, or 0 if no test results.
        """
        if not self.test_results:
            return 0
        # Get all loop indices from results that have timing data
        loop_indices = {result.loop_index for result in self.test_results if result.runtime is not None}
        if not loop_indices:
            # Fallback: use all loop indices even without runtime
            loop_indices = {result.loop_index for result in self.test_results}
        return max(loop_indices) if loop_indices else 0

    def file_to_no_of_tests(self, test_functions_to_remove: list[str]) -> Counter[Path]:
        map_gen_test_file_to_no_of_tests = Counter()
        for gen_test_result in self.test_results:
            if (
                gen_test_result.test_type == TestType.GENERATED_REGRESSION
                and gen_test_result.id.test_function_name not in test_functions_to_remove
            ):
                map_gen_test_file_to_no_of_tests[gen_test_result.file_name] += 1
        return map_gen_test_file_to_no_of_tests

    def __iter__(self) -> Iterator[FunctionTestInvocation]:  # type: ignore[override]
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
        if len(self) != len(other):  # type: ignore[arg-type]
            return False
        from codeflash_python.verification.comparator import comparator

        original_recursion_limit = sys.getrecursionlimit()
        cast("TestResults", other)
        for test_result in self:
            other_test_result = other.get_by_unique_invocation_loop_id(test_result.unique_invocation_loop_id)  # type: ignore[attr-defined]
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
