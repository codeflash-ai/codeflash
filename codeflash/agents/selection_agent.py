from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.models.models import ConcurrencyMetrics, OptimizedCandidateResult


class SelectionAgent(BaseAgent):
    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        super().__init__(agent_id="selection", coordinator=coordinator)

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "evaluate_candidate":
            return self._evaluate_candidate(task)
        if task_type == "select_best":
            return self._select_best(task)
        if task_type == "rank_candidates":
            return self._rank_candidates(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _evaluate_candidate(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.result.critic import coverage_critic, quantity_of_tests_critic, speedup_critic

            candidate_result: OptimizedCandidateResult = payload["candidate_result"]
            original_runtime: int = payload["original_runtime"]
            best_runtime_until_now: int | None = payload.get("best_runtime_until_now")
            original_async_throughput: int | None = payload.get("original_async_throughput")
            best_throughput_until_now: int | None = payload.get("best_throughput_until_now")
            original_concurrency_metrics: ConcurrencyMetrics | None = payload.get("original_concurrency_metrics")
            best_concurrency_ratio_until_now: float | None = payload.get("best_concurrency_ratio_until_now")
            original_coverage = payload.get("original_coverage")

            passes_quantity_check = quantity_of_tests_critic(candidate_result)
            passes_speedup_check = speedup_critic(
                candidate_result,
                original_runtime,
                best_runtime_until_now,
                original_async_throughput=original_async_throughput,
                best_throughput_until_now=best_throughput_until_now,
                original_concurrency_metrics=original_concurrency_metrics,
                best_concurrency_ratio_until_now=best_concurrency_ratio_until_now,
            )
            passes_coverage_check = coverage_critic(original_coverage) if original_coverage else True

            is_acceptable = passes_quantity_check and passes_speedup_check and passes_coverage_check

            result_data = {
                "candidate_result": candidate_result,
                "is_acceptable": is_acceptable,
                "passes_quantity_check": passes_quantity_check,
                "passes_speedup_check": passes_speedup_check,
                "passes_coverage_check": passes_coverage_check,
            }

            logger.debug(
                f"SelectionAgent: Evaluated candidate, acceptable={is_acceptable} "
                f"(qty={passes_quantity_check}, speed={passes_speedup_check}, cov={passes_coverage_check})"
            )
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"SelectionAgent evaluation failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _select_best(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.result.critic import get_acceptance_reason, performance_gain

            candidates: list[dict[str, Any]] = payload["candidates"]
            original_runtime: int = payload["original_runtime"]
            original_async_throughput: int | None = payload.get("original_async_throughput")
            original_concurrency_metrics: ConcurrencyMetrics | None = payload.get("original_concurrency_metrics")

            if not candidates:
                return wrap_result({"best_candidate": None, "reason": "No candidates provided"}, task.task_id, self.agent_id)

            acceptable_candidates = [c for c in candidates if c.get("is_acceptable", False)]

            if not acceptable_candidates:
                return wrap_result({"best_candidate": None, "reason": "No acceptable candidates"}, task.task_id, self.agent_id)

            best = min(acceptable_candidates, key=lambda c: c.get("runtime", float("inf")))
            best_runtime = best.get("runtime", original_runtime)

            perf_gain = performance_gain(original_runtime_ns=original_runtime, optimized_runtime_ns=best_runtime)
            speedup = (original_runtime / best_runtime) if best_runtime > 0 else 0.0

            acceptance_reason = get_acceptance_reason(
                original_runtime_ns=original_runtime,
                optimized_runtime_ns=best_runtime,
                original_async_throughput=original_async_throughput,
                optimized_async_throughput=best.get("async_throughput"),
                original_concurrency_metrics=original_concurrency_metrics,
                optimized_concurrency_metrics=best.get("concurrency_metrics"),
            )

            result_data = {
                "best_candidate": best,
                "performance_gain": perf_gain,
                "speedup": speedup,
                "acceptance_reason": acceptance_reason.value,
            }

            logger.info(f"SelectionAgent: Selected best candidate with {speedup:.2f}x speedup")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"SelectionAgent selection failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _rank_candidates(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.result.critic import performance_gain

            candidates: list[dict[str, Any]] = payload["candidates"]
            original_runtime: int = payload["original_runtime"]

            ranked = []
            for candidate in candidates:
                runtime = candidate.get("runtime", float("inf"))
                perf = performance_gain(original_runtime_ns=original_runtime, optimized_runtime_ns=runtime)
                ranked.append({
                    **candidate,
                    "performance_gain": perf,
                    "speedup": (original_runtime / runtime) if runtime > 0 else 0.0,
                })

            ranked.sort(key=lambda c: c.get("performance_gain", 0), reverse=True)

            result_data = {
                "ranked_candidates": ranked,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"SelectionAgent ranking failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)


def create_evaluation_task(
    candidate_result: Any,
    original_runtime: int,
    best_runtime_until_now: int | None = None,
    original_async_throughput: int | None = None,
    best_throughput_until_now: int | None = None,
    original_concurrency_metrics: Any | None = None,
    best_concurrency_ratio_until_now: float | None = None,
    original_coverage: Any | None = None,
) -> AgentTask:
    return AgentTask.create(
        task_type="evaluate_candidate",
        payload={
            "candidate_result": candidate_result,
            "original_runtime": original_runtime,
            "best_runtime_until_now": best_runtime_until_now,
            "original_async_throughput": original_async_throughput,
            "best_throughput_until_now": best_throughput_until_now,
            "original_concurrency_metrics": original_concurrency_metrics,
            "best_concurrency_ratio_until_now": best_concurrency_ratio_until_now,
            "original_coverage": original_coverage,
        },
        priority=5,
    )


def create_selection_task(
    candidates: list[dict[str, Any]],
    original_runtime: int,
    original_async_throughput: int | None = None,
    original_concurrency_metrics: Any | None = None,
) -> AgentTask:
    return AgentTask.create(
        task_type="select_best",
        payload={
            "candidates": candidates,
            "original_runtime": original_runtime,
            "original_async_throughput": original_async_throughput,
            "original_concurrency_metrics": original_concurrency_metrics,
        },
        priority=4,
    )


def create_ranking_task(
    candidates: list[dict[str, Any]],
    original_runtime: int,
) -> AgentTask:
    return AgentTask.create(
        task_type="rank_candidates",
        payload={
            "candidates": candidates,
            "original_runtime": original_runtime,
        },
        priority=4,
    )
