"""Shared types for cross-repo use between codeflash CLI and codeflash-internal server.

This module defines types that are duplicated or shared between the client (CLI)
and the server. Centralizing them here allows both sides to import from a single
source of truth.
"""

from __future__ import annotations

from enum import Enum

from pydantic.dataclasses import dataclass

# --- Enums ---


class OptimizedCandidateSource(str, Enum):
    OPTIMIZE = "OPTIMIZE"
    OPTIMIZE_LP = "OPTIMIZE_LP"
    REFINE = "REFINE"
    REPAIR = "REPAIR"
    ADAPTIVE = "ADAPTIVE"
    JIT_REWRITE = "JIT_REWRITE"


# --- Models ---


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.type}:{self.name}"


# --- Constants: Language identifiers ---

LANGUAGE_PYTHON = "python"
LANGUAGE_JAVASCRIPT = "javascript"
LANGUAGE_TYPESCRIPT = "typescript"
LANGUAGE_JAVA = "java"

SUPPORTED_LANGUAGES = frozenset({LANGUAGE_PYTHON, LANGUAGE_JAVASCRIPT, LANGUAGE_TYPESCRIPT, LANGUAGE_JAVA})

# --- Constants: Test type names ---

TEST_TYPE_EXISTING_UNIT = "existing_unit_test"
TEST_TYPE_GENERATED_REGRESSION = "generated_regression"
TEST_TYPE_REPLAY = "replay_test"
TEST_TYPE_CONCOLIC_COVERAGE = "concolic_coverage_test"
