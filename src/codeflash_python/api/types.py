from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from pydantic.dataclasses import dataclass

from codeflash.models.models import OptimizedCandidateSource


@dataclass(frozen=True)
class AIServiceRefinerRequest:
    """Request model for code refinement API.

    Supports multi-language optimization refinement with optional multi-file context.
    """

    optimization_id: str
    original_source_code: str
    read_only_dependency_code: str
    original_code_runtime: int
    optimized_source_code: str
    optimized_explanation: str
    optimized_code_runtime: int
    speedup: str
    trace_id: str
    original_line_profiler_results: str
    optimized_line_profiler_results: str
    function_references: str | None = None
    call_sequence: int | None = None
    language: str = "python"
    language_version: str | None = None
    additional_context_files: dict[str, str] | None = None


@dataclass(frozen=True)
class AdaptiveOptimizedCandidate:
    optimization_id: str
    source_code: str
    explanation: str
    source: OptimizedCandidateSource
    speedup: str


@dataclass(frozen=True)
class AIServiceAdaptiveOptimizeRequest:
    trace_id: str
    original_source_code: str
    candidates: list[AdaptiveOptimizedCandidate]


class TestDiffScope(str, Enum):
    RETURN_VALUE = "return_value"
    STDOUT = "stdout"
    DID_PASS = "did_pass"  # noqa: S105


@dataclass
class TestDiff:
    scope: TestDiffScope
    original_pass: bool
    candidate_pass: bool

    original_value: str | None = None
    candidate_value: str | None = None
    test_src_code: str | None = None
    candidate_pytest_error: str | None = None
    original_pytest_error: str | None = None


@dataclass(frozen=True)
class AIServiceCodeRepairRequest:
    optimization_id: str
    original_source_code: str
    modified_source_code: str
    trace_id: str
    test_diffs: list[TestDiff]
    language: str = "python"


class OptimizationReviewResult(NamedTuple):
    """Result from the optimization review API."""

    review: str  # "high", "medium", "low", or ""
    explanation: str


class FunctionRepairInfo(NamedTuple):
    function_name: str
    reason: str


class TestFileReview(NamedTuple):
    test_index: int
    functions_to_repair: list[FunctionRepairInfo]
