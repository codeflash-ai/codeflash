from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import codeflash.languages.function_optimizer as function_optimizer_module
from codeflash.languages.function_optimizer import FunctionOptimizer

if TYPE_CHECKING:
    import pytest


class RecordingExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def submit(self, fn: object, *args: object, **kwargs: object) -> object:
        self.submissions.append((fn, args, kwargs))
        return SimpleNamespace(kind="future")


class FakeCandidateProcessor:
    last_init: dict[str, object] | None = None

    def __init__(
        self,
        candidates: list[object],
        future_line_profile_results: object,
        eval_ctx: object,
        effort: str,
        original_markdown_code: str,
        future_all_refinements: list[object],
        future_all_code_repair: list[object],
        future_adaptive_optimizations: list[object],
        *,
        normalize_fn: object,
        normalized_original: str,
        original_flat_code: str,
    ) -> None:
        type(self).last_init = {
            "future_line_profile_results": future_line_profile_results,
            "eval_ctx": eval_ctx,
            "effort": effort,
            "original_markdown_code": original_markdown_code,
            "future_all_refinements": future_all_refinements,
            "future_all_code_repair": future_all_code_repair,
            "future_adaptive_optimizations": future_adaptive_optimizations,
            "normalize_fn": normalize_fn,
            "normalized_original": normalized_original,
            "original_flat_code": original_flat_code,
        }
        self.candidate_len = len(candidates)
        self.normalized_cache = {
            candidate.optimization_id: f"cached-{candidate.optimization_id}" for candidate in candidates
        }
        self._nodes = [SimpleNamespace(candidate=candidate) for candidate in candidates]

    def is_done(self) -> bool:
        return not self._nodes

    def get_next_candidate(self) -> object | None:
        if not self._nodes:
            return None
        return self._nodes.pop(0)


def build_optimizer() -> tuple[FunctionOptimizer, RecordingExecutor, object, object]:
    optimizer = object.__new__(FunctionOptimizer)
    executor = RecordingExecutor()
    control_client = SimpleNamespace(optimize_python_code_line_profiler=MagicMock(name="control_line_profiler"))
    local_client = SimpleNamespace(optimize_python_code_line_profiler=MagicMock(name="local_line_profiler"))

    optimizer.future_all_refinements = [object()]
    optimizer.future_all_code_repair = [object()]
    optimizer.future_adaptive_optimizations = [object()]
    optimizer.repair_counter = 5
    optimizer.adaptive_optimization_counter = 7
    optimizer.aiservice_client = control_client
    optimizer.local_aiservice_client = local_client
    optimizer.executor = executor
    optimizer.experiment_id = "experiment-id"
    optimizer.effort = "medium"
    optimizer.is_numerical_code = True
    optimizer.args = SimpleNamespace(no_jit_opts=False, rerun=None)
    optimizer.language_support = SimpleNamespace(
        language_version="3.12",
        normalize_code=lambda code: f"normalized::{code}",
    )
    optimizer.function_to_optimize = SimpleNamespace(
        qualified_name="pkg.fn",
        language="python",
        file_path=Path("target.py"),
    )
    optimizer.function_to_optimize_source_code = "original source"
    optimizer.get_trace_id = MagicMock(side_effect=lambda exp_type: f"trace-{exp_type}")
    optimizer.process_single_candidate = MagicMock()
    optimizer.write_code_and_helpers = MagicMock()
    optimizer.select_best_optimization = MagicMock(return_value=None)
    optimizer.log_evaluation_results = MagicMock()
    return optimizer, executor, control_client, local_client


def test_determine_best_candidate_processes_candidates_and_logs_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, executor, control_client, _ = build_optimizer()
    best_optimization = SimpleNamespace(name="best")
    optimizer.select_best_optimization.return_value = best_optimization
    monkeypatch.setattr(function_optimizer_module, "CandidateProcessor", FakeCandidateProcessor)

    code_context = SimpleNamespace(
        read_writable_code=SimpleNamespace(markdown="markdown code", flat="flat code"),
        read_only_context_code="dependency code",
    )
    original_code_baseline = SimpleNamespace(line_profile_results={"str_out": "line profile"})
    candidates = [
        SimpleNamespace(optimization_id="candidate-1"),
        SimpleNamespace(optimization_id="candidate-2"),
    ]
    original_helper_code = {Path("helper.py"): "helper code"}

    result = optimizer.determine_best_candidate(
        candidates=candidates,
        code_context=code_context,
        original_code_baseline=original_code_baseline,
        original_helper_code=original_helper_code,
        file_path_to_helper_classes={},
        exp_type="EXP0",
        function_references="function references",
    )

    assert result is best_optimization
    assert optimizer.future_all_refinements == []
    assert optimizer.future_all_code_repair == []
    assert optimizer.future_adaptive_optimizations == []
    assert optimizer.repair_counter == 0
    assert optimizer.adaptive_optimization_counter == 0
    assert executor.submissions[0][0] is control_client.optimize_python_code_line_profiler
    assert executor.submissions[0][2]["trace_id"] == "trace-EXP0"
    assert executor.submissions[0][2]["source_code"] == "markdown code"
    assert FakeCandidateProcessor.last_init is not None
    assert FakeCandidateProcessor.last_init["normalized_original"] == "normalized::flat code"
    assert [call.kwargs["candidate_index"] for call in optimizer.process_single_candidate.call_args_list] == [1, 2]
    assert [
        call.kwargs["cached_normalized_code"] for call in optimizer.process_single_candidate.call_args_list
    ] == ["cached-candidate-1", "cached-candidate-2"]
    assert optimizer.write_code_and_helpers.call_count == 2
    assert optimizer.select_best_optimization.call_args.kwargs["ai_service_client"] is control_client
    optimizer.log_evaluation_results.assert_called_once()


def test_determine_best_candidate_uses_local_client_for_experiment_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer, executor, _, local_client = build_optimizer()
    monkeypatch.setattr(function_optimizer_module, "CandidateProcessor", FakeCandidateProcessor)

    code_context = SimpleNamespace(
        read_writable_code=SimpleNamespace(markdown="markdown code", flat="flat code"),
        read_only_context_code="dependency code",
    )
    original_code_baseline = SimpleNamespace(line_profile_results={"str_out": "line profile"})

    result = optimizer.determine_best_candidate(
        candidates=[],
        code_context=code_context,
        original_code_baseline=original_code_baseline,
        original_helper_code={},
        file_path_to_helper_classes={},
        exp_type="EXP1",
        function_references="function references",
    )

    assert result is None
    assert executor.submissions[0][0] is local_client.optimize_python_code_line_profiler
    assert optimizer.select_best_optimization.call_args.kwargs["ai_service_client"] is local_client
    optimizer.log_evaluation_results.assert_not_called()
