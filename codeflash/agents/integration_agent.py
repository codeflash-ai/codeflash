from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import BestOptimization


class IntegrationAgent(BaseAgent):
    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        super().__init__(agent_id="integration", coordinator=coordinator)

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "create_pr":
            return self._create_pr(task)
        if task_type == "apply_optimization":
            return self._apply_optimization(task)
        if task_type == "generate_explanation":
            return self._generate_explanation(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _create_pr(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.result.create_pr import check_create_pr

            best_optimization: BestOptimization = payload["best_optimization"]
            function_to_optimize: FunctionToOptimize = payload["function_to_optimize"]
            args: Namespace = payload["args"]

            pr_result = check_create_pr(
                best_optimization=best_optimization,
                function_to_optimize=function_to_optimize,
                args=args,
            )

            result_data = {
                "pr_result": pr_result,
                "function_name": function_to_optimize.qualified_name,
            }

            if pr_result:
                logger.info(f"IntegrationAgent: Created PR for {function_to_optimize.qualified_name}")
            else:
                logger.debug(f"IntegrationAgent: PR creation skipped for {function_to_optimize.qualified_name}")

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"IntegrationAgent PR creation failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _apply_optimization(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            best_optimization: BestOptimization = payload["best_optimization"]
            function_to_optimize: FunctionToOptimize = payload["function_to_optimize"]

            read_writable_code = best_optimization.code_context.read_writable_code
            files_modified = []

            for code_string in read_writable_code.code_strings:
                file_path = code_string.file_path
                file_path.write_text(code_string.code, encoding="utf-8")
                files_modified.append(file_path)

            result_data = {
                "files_modified": files_modified,
                "function_name": function_to_optimize.qualified_name,
            }

            logger.info(f"IntegrationAgent: Applied optimization to {len(files_modified)} files")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"IntegrationAgent optimization application failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _generate_explanation(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.result.explanation import Explanation

            best_optimization: BestOptimization = payload["best_optimization"]
            function_to_optimize: FunctionToOptimize = payload["function_to_optimize"]
            original_runtime: int = payload["original_runtime"]

            explanation = Explanation(
                best_optimization=best_optimization,
                function_to_optimize=function_to_optimize,
                original_runtime=original_runtime,
            )

            result_data = {
                "explanation": explanation,
                "summary": explanation.summary if hasattr(explanation, "summary") else None,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"IntegrationAgent explanation generation failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)


def create_pr_task(
    best_optimization: Any,
    function_to_optimize: Any,
    args: Any,
) -> AgentTask:
    return AgentTask.create(
        task_type="create_pr",
        payload={
            "best_optimization": best_optimization,
            "function_to_optimize": function_to_optimize,
            "args": args,
        },
        priority=3,
    )


def create_apply_optimization_task(
    best_optimization: Any,
    function_to_optimize: Any,
) -> AgentTask:
    return AgentTask.create(
        task_type="apply_optimization",
        payload={
            "best_optimization": best_optimization,
            "function_to_optimize": function_to_optimize,
        },
        priority=3,
    )


def create_explanation_task(
    best_optimization: Any,
    function_to_optimize: Any,
    original_runtime: int,
) -> AgentTask:
    return AgentTask.create(
        task_type="generate_explanation",
        payload={
            "best_optimization": best_optimization,
            "function_to_optimize": function_to_optimize,
            "original_runtime": original_runtime,
        },
        priority=2,
    )
