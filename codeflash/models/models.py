from __future__ import annotations

from typing import Optional

from jedi.api.classes import Name
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from codeflash.api.aiservice import OptimizedCandidate
from codeflash.discovery.functions_to_optimize import FunctionParent
from codeflash.verification.test_results import TestResults

# If the method spam is in the class Ham, which is at the top level of the module eggs in the package foo, the fully
# qualified name of the method is foo.eggs.Ham.spam, its qualified name is Ham.spam, and its name is spam. The full name
# of the module is foo.eggs.


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FunctionSource:
    file_path: str
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
