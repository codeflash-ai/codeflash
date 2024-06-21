from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from codeflash.discovery.functions_to_optimize import FunctionParent
from codeflash.api.aiservice import OptimizedCandidate
from codeflash.optimization.function_context import Source
from codeflash.verification.test_results import TestResults


class BestOptimization(BaseModel):
    candidate: OptimizedCandidate
    helper_functions: list[tuple[Source, str, str]]
    runtime: int
    winning_test_results: TestResults


class CodeOptimizationContext(BaseModel):
    code_to_optimize_with_helpers: str
    contextual_dunder_methods: set[tuple[str, str]]
    helper_functions: list[tuple[Source, str, str]]
    preexisting_functions: list[tuple[str, list[FunctionParent]]]


class OptimizedCandidateResult(BaseModel):
    times_run: int
    best_test_runtime: int
    best_test_results: TestResults


class GeneratedTests(BaseModel):
    generated_original_test_source: str
    instrumented_test_source: str


class OriginalCodeBaseline(BaseModel):
    generated_test_results: TestResults
    existing_test_results: TestResults
    overall_test_results: Optional[TestResults]
    runtime: int


class OptimizationSet(BaseModel):
    control: list[OptimizedCandidate]
    experiment: Optional[list[OptimizedCandidate]]
