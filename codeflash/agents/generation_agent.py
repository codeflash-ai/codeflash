from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.api.aiservice import AiServiceClient
    from codeflash.models.models import CodeOptimizationContext


class GenerationAgent(BaseAgent):
    def __init__(
        self,
        coordinator: AgentCoordinator | None = None,
        aiservice_client: AiServiceClient | None = None,
    ) -> None:
        super().__init__(agent_id="generation", coordinator=coordinator)
        self._aiservice_client = aiservice_client

    @property
    def aiservice_client(self) -> AiServiceClient:
        if self._aiservice_client is None:
            from codeflash.api.aiservice import AiServiceClient

            self._aiservice_client = AiServiceClient()
        return self._aiservice_client

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "generate_candidates":
            return self._generate_candidates(task)
        if task_type == "refine_candidate":
            return self._refine_candidate(task)
        if task_type == "repair_candidate":
            return self._repair_candidate(task)
        if task_type == "adaptive_optimize":
            return self._adaptive_optimize(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _generate_candidates(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            code_context: CodeOptimizationContext = payload["code_context"]
            trace_id: str = payload["trace_id"]
            is_async: bool = payload.get("is_async", False)
            n_candidates: int = payload.get("n_candidates", 5)
            is_numerical: bool | None = payload.get("is_numerical")
            language: str = payload.get("language", "python")

            source_code = code_context.read_writable_code.markdown
            dependency_code = code_context.read_only_context_code or ""

            candidates = self.aiservice_client.optimize_code(
                source_code=source_code,
                dependency_code=dependency_code,
                trace_id=trace_id,
                language=language,
                is_async=is_async,
                n_candidates=n_candidates,
                is_numerical_code=is_numerical,
            )

            result_data = {
                "candidates": candidates,
                "candidate_count": len(candidates),
            }

            logger.info(f"GenerationAgent: Generated {len(candidates)} optimization candidates")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"GenerationAgent candidate generation failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _refine_candidate(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.models.models import AIServiceRefinerRequest

            refiner_request: AIServiceRefinerRequest = payload["refiner_request"]
            trace_id: str = payload["trace_id"]

            refined_candidates = self.aiservice_client.refine_code(refiner_request, trace_id)

            result_data = {
                "refined_candidates": refined_candidates,
                "original_optimization_id": refiner_request.optimization_id,
            }

            logger.debug(f"GenerationAgent: Refined candidate {refiner_request.optimization_id}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"GenerationAgent refinement failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _repair_candidate(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.models.models import AIServiceCodeRepairRequest

            repair_request: AIServiceCodeRepairRequest = payload["repair_request"]
            trace_id: str = payload["trace_id"]

            repaired_candidate = self.aiservice_client.repair_code(repair_request, trace_id)

            result_data = {
                "repaired_candidate": repaired_candidate,
                "original_optimization_id": repair_request.optimization_id,
            }

            logger.debug(f"GenerationAgent: Repaired candidate {repair_request.optimization_id}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"GenerationAgent repair failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _adaptive_optimize(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.models.models import AIServiceAdaptiveOptimizeRequest

            adaptive_request: AIServiceAdaptiveOptimizeRequest = payload["adaptive_request"]
            trace_id: str = payload["trace_id"]

            adaptive_candidates = self.aiservice_client.adaptive_optimize(adaptive_request, trace_id)

            result_data = {
                "adaptive_candidates": adaptive_candidates,
            }

            logger.debug("GenerationAgent: Generated adaptive optimization candidates")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"GenerationAgent adaptive optimization failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)


def create_generation_task(
    code_context: Any,
    trace_id: str,
    is_async: bool = False,
    n_candidates: int = 5,
    is_numerical: bool | None = None,
    language: str = "python",
) -> AgentTask:
    return AgentTask.create(
        task_type="generate_candidates",
        payload={
            "code_context": code_context,
            "trace_id": trace_id,
            "is_async": is_async,
            "n_candidates": n_candidates,
            "is_numerical": is_numerical,
            "language": language,
        },
        priority=7,
    )


def create_refinement_task(refiner_request: Any, trace_id: str) -> AgentTask:
    return AgentTask.create(
        task_type="refine_candidate",
        payload={
            "refiner_request": refiner_request,
            "trace_id": trace_id,
        },
        priority=5,
    )


def create_repair_task(repair_request: Any, trace_id: str) -> AgentTask:
    return AgentTask.create(
        task_type="repair_candidate",
        payload={
            "repair_request": repair_request,
            "trace_id": trace_id,
        },
        priority=5,
    )


def create_adaptive_optimization_task(adaptive_request: Any, trace_id: str) -> AgentTask:
    return AgentTask.create(
        task_type="adaptive_optimize",
        payload={
            "adaptive_request": adaptive_request,
            "trace_id": trace_id,
        },
        priority=4,
    )
