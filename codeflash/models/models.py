from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from jedi.api.classes import Name
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.verification.test_results import TestResults, TestType

# If the method spam is in the class Ham, which is at the top level of the module eggs in the package foo, the fully
# qualified name of the method is foo.eggs.Ham.spam, its qualified name is Ham.spam, and its name is spam. The full name
# of the module is foo.eggs.


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FunctionSource:
    file_path: Path
    qualified_name: str
    fully_qualified_name: str
    only_function_name: str
    source_code: str
    jedi_definition: Name


class BestOptimization(BaseModel):
    candidate: OptimizedCandidate
    helper_functions: list[FunctionSource]
    runtime: int
    winning_test_results: TestResults


class CodeOptimizationContext(BaseModel):
    code_to_optimize_with_helpers: str
    contextual_dunder_methods: set[tuple[str, str]]
    helper_functions: list[FunctionSource]
    preexisting_objects: list[tuple[str, list[FunctionParent]]]


class OptimizedCandidateResult(BaseModel):
    max_loop_count: int
    best_test_runtime: int
    test_results: TestResults
    optimization_candidate_index: int
    total_candidate_timing: int


class GeneratedTests(BaseModel):
    generated_original_test_source: str
    instrumented_test_source: str


class GeneratedTestsList(BaseModel):
    generated_tests: list[GeneratedTests]


class TestFile(BaseModel):
    instrumented_file_path: Path
    original_file_path: Optional[Path] = None
    original_source: Optional[str] = None
    test_type: TestType


class TestFiles(BaseModel):
    test_files: list[TestFile]

    def get_by_type(self, test_type: TestType) -> TestFiles:
        return TestFiles(test_files=[test_file for test_file in self.test_files if test_file.test_type == test_type])

    def add(self, test_file: TestFile) -> None:
        if test_file not in self.test_files:
            self.test_files.append(test_file)
        else:
            raise ValueError("Test file already exists in the list")

    def get_by_original_file_path(self, file_path: Path) -> TestFile | None:
        return next((test_file for test_file in self.test_files if test_file.original_file_path == file_path), None)

    def get_test_type_by_instrumented_file_path(self, file_path: Path) -> TestType | None:
        return next(
            (test_file.test_type for test_file in self.test_files if test_file.instrumented_file_path == file_path),
            None,
        )

    def get_test_type_by_original_file_path(self, file_path: Path) -> TestType | None:
        return next(
            (test_file.test_type for test_file in self.test_files if test_file.original_file_path == file_path), None
        )

    def __iter__(self) -> Iterator[TestFile]:
        return iter(self.test_files)

    def __len__(self):
        return len(self.test_files)


class OriginalCodeBaseline(BaseModel):
    generated_test_results: TestResults
    existing_test_results: TestResults
    overall_test_results: Optional[TestResults]
    runtime: int


class OptimizationSet(BaseModel):
    control: list[OptimizedCandidate]
    experiment: Optional[list[OptimizedCandidate]]


@dataclass(frozen=True)
class TestsInFile:
    test_file: Path
    test_class: Optional[str]  # This might be unused...
    test_function: str
    test_suite: Optional[str]
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


@dataclass
class FunctionCoverage:
    """Represents the coverage data for a specific function in a source file."""

    name: str
    coverage: float
    executed_lines: list[int]
    unexecuted_lines: list[int]
