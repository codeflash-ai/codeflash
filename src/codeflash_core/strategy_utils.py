"""Shared types and utilities for optimization strategies."""

from __future__ import annotations

import contextlib
import json
import logging
import threading  # noqa: TC003 - used at runtime by OptimizationRuntime.is_cancelled()
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codeflash_core.models import TestDiff

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeflash_core.config import CoreConfig, TestConfig
    from codeflash_core.models import (
        Candidate,
        CodeContext,
        FunctionToOptimize,
        GeneratedTestSuite,
        OptimizationResult,
        ScoredCandidate,
        TestResults,
    )
    from codeflash_core.protocols import LanguagePlugin

logger = logging.getLogger(__name__)

MIN_CORRECT_CANDIDATES = 2


@dataclass
class StageSpec:
    """Describes a single stage in the optimization pipeline."""

    key: str
    description: str


@dataclass
class OptimizationRuntime:
    """Everything a strategy needs to orchestrate an optimization."""

    plugin: LanguagePlugin
    config: CoreConfig
    test_config: TestConfig
    cancel_event: threading.Event
    output_comparator: Callable[..., bool] | None
    trace_id: str

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()


@runtime_checkable
class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies.

    Implement this to define a custom optimization pipeline for a single function.
    The Optimizer handles discovery, indexing, ranking, and the per-function loop;
    the strategy controls everything within a single function's optimization.
    """

    def optimize_function(
        self, function: FunctionToOptimize, runtime: OptimizationRuntime
    ) -> OptimizationResult | None: ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def cleanup_generated_tests(generated_tests: GeneratedTestSuite | None) -> None:
    """Remove generated test files from disk."""
    if not generated_tests:
        return
    for tf in generated_tests.test_files:
        for path in (tf.behavior_test_path, tf.perf_test_path):
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)


def restore_test_snapshots(generated_tests: GeneratedTestSuite, snapshots: dict[int, tuple[str, str, str]]) -> None:
    """Restore test file contents from pre-repair snapshots."""
    for i, tf in enumerate(generated_tests.test_files):
        if i not in snapshots:
            continue
        orig_source, orig_behavior, orig_perf = snapshots[i]
        tf.original_test_source = orig_source
        tf.behavior_test_source = orig_behavior
        tf.perf_test_source = orig_perf
        with contextlib.suppress(OSError):
            tf.behavior_test_path.write_text(orig_behavior, encoding="utf-8")
        with contextlib.suppress(OSError):
            tf.perf_test_path.write_text(orig_perf, encoding="utf-8")


def compute_test_diffs(baseline: TestResults, candidate: TestResults) -> list[TestDiff]:
    """Compute test outcome differences between baseline and candidate runs."""
    diffs: list[TestDiff] = []
    baseline_by_id = {o.test_id: o for o in baseline.outcomes}
    candidate_by_id = {o.test_id: o for o in candidate.outcomes}

    for test_id, base_outcome in baseline_by_id.items():
        cand_outcome = candidate_by_id.get(test_id)
        if (
            cand_outcome is None
            or base_outcome.status != cand_outcome.status
            or base_outcome.output != cand_outcome.output
        ):
            diffs.append(
                TestDiff(
                    test_id=test_id,
                    baseline_output=base_outcome.output,
                    candidate_output=cand_outcome.output if cand_outcome else None,
                )
            )

    return diffs


def log_optimization_run(
    function: FunctionToOptimize,
    runtime: OptimizationRuntime,
    context: CodeContext | None = None,
    candidates: list[Candidate] | None = None,
    generated_tests: GeneratedTestSuite | None = None,
    scored: list[ScoredCandidate] | None = None,
    result: OptimizationResult | None = None,
    stage_reached: str = "",
    exit_reason: str = "",
) -> None:
    """Append a JSON record of the optimization run to codeflash_trace.log."""
    record: dict[str, Any] = {
        "trace_id": runtime.trace_id,
        "function": function.qualified_name,
        "file": str(function.file_path),
        "stage_reached": stage_reached,
        "exit_reason": exit_reason,
    }

    if context:
        record["context"] = {
            "target_code_lines": len(context.target_code.splitlines()),
            "read_only_context_lines": len(context.read_only_context.splitlines()) if context.read_only_context else 0,
            "imports": len(context.imports),
            "helpers": [{"name": h.qualified_name, "file": str(h.file_path)} for h in context.helper_functions],
        }

    record["candidates_count"] = len(candidates) if candidates else 0
    if candidates:
        record["candidates"] = [{"id": c.candidate_id, "source": c.source} for c in candidates]

    record["tests_count"] = len(generated_tests.test_files) if generated_tests and generated_tests.test_files else 0

    if scored:
        record["evaluation"] = [
            {
                "id": s.candidate.candidate_id,
                "source": s.candidate.source,
                "speedup": s.speedup,
                "score": s.score,
                "passed": s.test_results.passed,
            }
            for s in scored
        ]

    if result:
        record["result"] = {
            "speedup": result.speedup,
            "candidate_id": result.candidate.candidate_id,
            "explanation": result.explanation,
            "diff": result.diff,
        }

    log_path = runtime.config.project_root / "codeflash_trace.log"
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        logger.debug("Failed to write optimization trace log", exc_info=True)
