from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class TestOutcomeStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.type}:{self.name}"


@dataclass
class FunctionToOptimize:
    function_name: str
    file_path: Path
    parents: list[FunctionParent] = field(default_factory=list)
    starting_line: int | None = None
    ending_line: int | None = None
    starting_col: int | None = None
    ending_col: int | None = None
    is_async: bool = False
    is_method: bool = False
    language: str = "python"
    doc_start_line: int | None = None
    source_code: str = ""

    @property
    def qualified_name(self) -> str:
        if not self.parents:
            return self.function_name
        parent_path = ".".join(parent.name for parent in self.parents)
        return f"{parent_path}.{self.function_name}"

    @property
    def top_level_parent_name(self) -> str:
        return self.function_name if not self.parents else self.parents[0].name

    @property
    def class_name(self) -> str | None:
        for parent in reversed(self.parents):
            if parent.type == "ClassDef":
                return parent.name
        return None

    def qualified_name_with_modules_from_root(self, project_root_path: Path) -> str:
        from codeflash.code_utils.code_utils import module_name_from_file_path

        return f"{module_name_from_file_path(self.file_path, project_root_path)}.{self.qualified_name}"

    def __str__(self) -> str:
        qualified = f"{'.'.join([p.name for p in self.parents])}{'.' if self.parents else ''}{self.function_name}"
        line_info = f":{self.starting_line}-{self.ending_line}" if self.starting_line and self.ending_line else ""
        return f"{self.file_path}:{qualified}{line_info}"


@dataclass
class HelperFunction:
    name: str
    qualified_name: str
    file_path: Path
    source_code: str
    start_line: int
    end_line: int


@dataclass
class CodeContext:
    target_function: FunctionToOptimize
    target_code: str
    target_file: Path
    helper_functions: list[HelperFunction] = field(default_factory=list)
    read_only_context: str = ""
    imports: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    code: str
    explanation: str
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    parent_id: str = ""
    code_markdown: str = ""


@dataclass
class TestOutcome:
    test_id: str
    status: TestOutcomeStatus
    output: Any = None
    duration: float = 0.0
    error_message: str = ""


@dataclass
class TestResults:
    passed: bool
    outcomes: list[TestOutcome] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkResults:
    timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


@dataclass
class ScoredCandidate:
    candidate: Candidate
    test_results: TestResults
    benchmark_results: BenchmarkResults
    speedup: float
    score: float


@dataclass
class OptimizationResult:
    function: FunctionToOptimize
    original_code: str
    optimized_code: str
    speedup: float
    candidate: Candidate
    test_results: TestResults
    benchmark_results: BenchmarkResults
    diff: str = ""
    explanation: str = ""


@dataclass
class GeneratedTestFile:
    behavior_test_path: Path
    perf_test_path: Path
    behavior_test_source: str
    perf_test_source: str
    original_test_source: str


@dataclass
class GeneratedTestSuite:
    test_files: list[GeneratedTestFile] = field(default_factory=list)

    @property
    def behavior_test_paths(self) -> list[Path]:
        return [f.behavior_test_path for f in self.test_files]

    @property
    def perf_test_paths(self) -> list[Path]:
        return [f.perf_test_path for f in self.test_files]


@dataclass
class FunctionCoverage:
    name: str
    coverage: float
    executed_lines: list[int] = field(default_factory=list)
    unexecuted_lines: list[int] = field(default_factory=list)
    executed_branches: list[list[int]] = field(default_factory=list)
    unexecuted_branches: list[list[int]] = field(default_factory=list)


@dataclass
class CoverageData:
    file_path: Path
    coverage: float
    function_name: str
    main_func_coverage: FunctionCoverage
    dependent_func_coverage: FunctionCoverage | None = None
    threshold_percentage: float = 60.0


@dataclass
class TestDiff:
    test_id: str
    baseline_output: Any = None
    candidate_output: Any = None


@dataclass
class TestRepairInfo:
    function_name: str
    reason: str


@dataclass
class TestReviewResult:
    test_index: int
    functions_to_repair: list[TestRepairInfo] = field(default_factory=list)
