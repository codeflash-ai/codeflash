from __future__ import annotations

import enum
import json
import re
from collections.abc import Collection, Iterator
from enum import Enum, IntEnum
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Optional, Union

import sentry_sdk
from coverage.exceptions import NoDataError
from jedi.api.classes import Name
from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import validate_python_code
from codeflash.code_utils.coverage_utils import (
    build_fully_qualified_name,
    extract_dependent_function,
    generate_candidates,
)
from codeflash.code_utils.env_utils import is_end_to_end
from codeflash.verification.test_results import TestResults, TestType

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
        return (self.file_path == other.file_path and
                self.qualified_name == other.qualified_name and
                self.fully_qualified_name == other.fully_qualified_name and
                self.only_function_name == other.only_function_name and
                self.source_code == other.source_code)

    def __hash__(self) -> int:
        return hash((self.file_path, self.qualified_name, self.fully_qualified_name,
                     self.only_function_name, self.source_code))

class BestOptimization(BaseModel):
    candidate: OptimizedCandidate
    helper_functions: list[FunctionSource]
    runtime: int
    winning_behavioral_test_results: TestResults
    winning_benchmarking_test_results: TestResults


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
    helper_functions: list[FunctionSource]
    preexisting_objects: set[tuple[str, tuple[FunctionParent,...]]]

class CodeContextType(str, Enum):
    READ_WRITABLE = "READ_WRITABLE"
    READ_ONLY = "READ_ONLY"
    TESTGEN = "TESTGEN"


class OptimizedCandidateResult(BaseModel):
    max_loop_count: int
    best_test_runtime: int
    behavior_test_results: TestResults
    benchmarking_test_results: TestResults
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
    behavioral_test_results: TestResults
    benchmarking_test_results: TestResults
    lprofiler_test_results: str
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

    @staticmethod
    def load_from_sqlite_database(
        database_path: Path, config_path: Path, function_name: str, code_context: CodeOptimizationContext, source_code_path: Path
    ) -> CoverageData:
        """Load coverage data from an SQLite database, mimicking the behavior of load_from_coverage_file."""
        from coverage import Coverage
        from coverage.jsonreport import JsonReporter

        cov = Coverage(data_file=database_path,config_file=config_path, data_suffix=True, auto_data=True, branch=True)

        if not database_path.stat().st_size or not database_path.exists():
            logger.debug(f"Coverage database {database_path} is empty or does not exist")
            sentry_sdk.capture_message(f"Coverage database {database_path} is empty or does not exist")
            return CoverageData.create_empty(source_code_path, function_name, code_context)
        cov.load()

        reporter = JsonReporter(cov)
        temp_json_file = database_path.with_suffix(".report.json")
        with temp_json_file.open("w") as f:
            try:
                reporter.report(morfs=[source_code_path.as_posix()], outfile=f)
            except NoDataError:
                sentry_sdk.capture_message(f"No coverage data found for {function_name} in {source_code_path}")
                return CoverageData.create_empty(source_code_path, function_name, code_context)
        with temp_json_file.open() as f:
            original_coverage_data = json.load(f)

        coverage_data, status = CoverageData._parse_coverage_file(temp_json_file, source_code_path)

        main_func_coverage, dependent_func_coverage = CoverageData._fetch_function_coverages(
            function_name, code_context, coverage_data, original_cov_data=original_coverage_data
        )

        total_executed_lines, total_unexecuted_lines = CoverageData._aggregate_coverage(
            main_func_coverage, dependent_func_coverage
        )

        total_lines = total_executed_lines | total_unexecuted_lines
        coverage = len(total_executed_lines) / len(total_lines) * 100 if total_lines else 0.0
        # coverage = (lines covered of the original function + its 1 level deep helpers) / (lines spanned by original function + its 1 level deep helpers), if no helpers then just the original function coverage

        functions_being_tested = [main_func_coverage.name]
        if dependent_func_coverage:
            functions_being_tested.append(dependent_func_coverage.name)

        graph = CoverageData._build_graph(main_func_coverage, dependent_func_coverage)
        temp_json_file.unlink()

        return CoverageData(
            file_path=source_code_path,
            coverage=coverage,
            function_name=function_name,
            functions_being_tested=functions_being_tested,
            graph=graph,
            code_context=code_context,
            main_func_coverage=main_func_coverage,
            dependent_func_coverage=dependent_func_coverage,
            status=status,
        )

    @staticmethod
    def _parse_coverage_file(
        coverage_file_path: Path, source_code_path: Path
    ) -> tuple[dict[str, dict[str, Any]], CoverageStatus]:
        with coverage_file_path.open() as f:
            coverage_data = json.load(f)

        candidates = generate_candidates(source_code_path)

        logger.debug(f"Looking for coverage data in {' -> '.join(candidates)}")
        for candidate in candidates:
            try:
                cov: dict[str, dict[str, Any]] = coverage_data["files"][candidate]["functions"]
                logger.debug(f"Coverage data found for {source_code_path} in {candidate}")
                status = CoverageStatus.PARSED_SUCCESSFULLY
                break
            except KeyError:
                continue
        else:
            logger.debug(f"No coverage data found for {source_code_path} in {candidates}")
            cov = {}
            status = CoverageStatus.NOT_FOUND
        return cov, status

    @staticmethod
    def _fetch_function_coverages(
        function_name: str,
        code_context: CodeOptimizationContext,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
    ) -> tuple[FunctionCoverage, Union[FunctionCoverage, None]]:
        resolved_name = build_fully_qualified_name(function_name, code_context)
        try:
            main_function_coverage = FunctionCoverage(
                name=resolved_name,
                coverage=coverage_data[resolved_name]["summary"]["percent_covered"],
                executed_lines=coverage_data[resolved_name]["executed_lines"],
                unexecuted_lines=coverage_data[resolved_name]["missing_lines"],
                executed_branches=coverage_data[resolved_name]["executed_branches"],
                unexecuted_branches=coverage_data[resolved_name]["missing_branches"],
            )
        except KeyError:
            main_function_coverage = FunctionCoverage(
                name=resolved_name,
                coverage=0,
                executed_lines=[],
                unexecuted_lines=[],
                executed_branches=[],
                unexecuted_branches=[],
            )

        dependent_function = extract_dependent_function(function_name, code_context)
        dependent_func_coverage = (
            CoverageData.grab_dependent_function_from_coverage_data(
                dependent_function, coverage_data, original_cov_data
            )
            if dependent_function
            else None
        )

        return main_function_coverage, dependent_func_coverage

    @staticmethod
    def _aggregate_coverage(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: Union[FunctionCoverage, None]
    ) -> tuple[set[int], set[int]]:
        total_executed_lines = set(main_func_coverage.executed_lines)
        total_unexecuted_lines = set(main_func_coverage.unexecuted_lines)

        if dependent_func_coverage:
            total_executed_lines.update(dependent_func_coverage.executed_lines)
            total_unexecuted_lines.update(dependent_func_coverage.unexecuted_lines)

        return total_executed_lines, total_unexecuted_lines

    @staticmethod
    def _build_graph(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: Union[FunctionCoverage, None]
    ) -> dict[str, dict[str, Collection[object]]]:
        graph = {
            main_func_coverage.name: {
                "executed_lines": set(main_func_coverage.executed_lines),
                "unexecuted_lines": set(main_func_coverage.unexecuted_lines),
                "executed_branches": main_func_coverage.executed_branches,
                "unexecuted_branches": main_func_coverage.unexecuted_branches,
            }
        }

        if dependent_func_coverage:
            graph[dependent_func_coverage.name] = {
                "executed_lines": set(dependent_func_coverage.executed_lines),
                "unexecuted_lines": set(dependent_func_coverage.unexecuted_lines),
                "executed_branches": dependent_func_coverage.executed_branches,
                "unexecuted_branches": dependent_func_coverage.unexecuted_branches,
            }

        return graph

    @staticmethod
    def grab_dependent_function_from_coverage_data(
        dependent_function_name: str,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
    ) -> FunctionCoverage:
        """Grab the dependent function from the coverage data."""
        try:
            return FunctionCoverage(
                name=dependent_function_name,
                coverage=coverage_data[dependent_function_name]["summary"]["percent_covered"],
                executed_lines=coverage_data[dependent_function_name]["executed_lines"],
                unexecuted_lines=coverage_data[dependent_function_name]["missing_lines"],
                executed_branches=coverage_data[dependent_function_name]["executed_branches"],
                unexecuted_branches=coverage_data[dependent_function_name]["missing_branches"],
            )
        except KeyError:
            msg = f"Coverage data not found for dependent function {dependent_function_name} in the coverage data"
            try:
                files = original_cov_data["files"]
                for file in files:
                    functions = files[file]["functions"]
                    for function in functions:
                        if dependent_function_name in function:
                            return FunctionCoverage(
                                name=dependent_function_name,
                                coverage=functions[function]["summary"]["percent_covered"],
                                executed_lines=functions[function]["executed_lines"],
                                unexecuted_lines=functions[function]["missing_lines"],
                                executed_branches=functions[function]["executed_branches"],
                                unexecuted_branches=functions[function]["missing_branches"],
                            )
                msg = f"Coverage data not found for dependent function {dependent_function_name} in the original coverage data"
            except KeyError:
                raise ValueError(msg) from None

        return FunctionCoverage(
            name=dependent_function_name,
            coverage=0,
            executed_lines=[],
            unexecuted_lines=[],
            executed_branches=[],
            unexecuted_branches=[],
        )

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
    LPROF = "lprof"
