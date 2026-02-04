from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CodeOptimizationContext


class AnalysisAgent(BaseAgent):
    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        super().__init__(agent_id="analysis", coordinator=coordinator)

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "extract_context":
            return self._extract_context(task)
        if task_type == "assess_complexity":
            return self._assess_complexity(task)
        if task_type == "check_numerical":
            return self._check_numerical(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _extract_context(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.context.code_context_extractor import get_code_optimization_context

            function_to_optimize: FunctionToOptimize = payload["function_to_optimize"]
            project_root: Path = payload["project_root"]

            code_context = get_code_optimization_context(function_to_optimize, project_root)

            result_data = {
                "code_context": code_context,
                "function_to_optimize": function_to_optimize,
            }

            logger.debug(f"AnalysisAgent: Extracted context for {function_to_optimize.qualified_name}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"AnalysisAgent context extraction failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _assess_complexity(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            code_context: CodeOptimizationContext = payload["code_context"]

            complexity_info = self._analyze_code_complexity(code_context)

            result_data = {
                "complexity": complexity_info,
                "code_context": code_context,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _check_numerical(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.code_utils.code_extractor import is_numerical_code

            source_code: str = payload["source_code"]

            is_numerical = is_numerical_code(source_code)

            result_data = {
                "is_numerical": is_numerical,
            }
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _analyze_code_complexity(self, code_context: CodeOptimizationContext) -> dict[str, Any]:
        from codeflash.code_utils.code_utils import encoded_tokens_len

        read_writable_tokens = encoded_tokens_len(code_context.read_writable_code.markdown)
        read_only_tokens = (
            encoded_tokens_len(code_context.read_only_context_code) if code_context.read_only_context_code else 0
        )

        helper_count = len(code_context.helper_functions)
        file_count = len(code_context.read_writable_code.code_strings)

        return {
            "total_tokens": read_writable_tokens + read_only_tokens,
            "read_writable_tokens": read_writable_tokens,
            "read_only_tokens": read_only_tokens,
            "helper_count": helper_count,
            "file_count": file_count,
            "complexity_level": self._determine_complexity_level(
                read_writable_tokens + read_only_tokens, helper_count, file_count
            ),
        }

    def _determine_complexity_level(self, total_tokens: int, helper_count: int, file_count: int) -> str:
        if total_tokens > 10000 or helper_count > 10 or file_count > 5:
            return "high"
        if total_tokens > 5000 or helper_count > 5 or file_count > 2:
            return "medium"
        return "low"


def create_context_extraction_task(
    function_to_optimize: Any,
    project_root: Path,
) -> AgentTask:
    return AgentTask.create(
        task_type="extract_context",
        payload={
            "function_to_optimize": function_to_optimize,
            "project_root": project_root,
        },
        priority=8,
    )


def create_complexity_assessment_task(code_context: Any) -> AgentTask:
    return AgentTask.create(
        task_type="assess_complexity",
        payload={
            "code_context": code_context,
        },
        priority=7,
    )


def create_numerical_check_task(source_code: str) -> AgentTask:
    return AgentTask.create(
        task_type="check_numerical",
        payload={
            "source_code": source_code,
        },
        priority=7,
    )
