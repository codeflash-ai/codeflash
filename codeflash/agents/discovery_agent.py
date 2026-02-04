from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.state.store import StateStore
    from codeflash.verification.verification_utils import TestConfig


class DiscoveryAgent(BaseAgent):
    def __init__(
        self,
        coordinator: AgentCoordinator | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        super().__init__(agent_id="discovery", coordinator=coordinator)
        self.state_store = state_store

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "discover_functions":
            return self._discover_functions(task)
        if task_type == "discover_tests":
            return self._discover_tests(task)
        if task_type == "filter_functions":
            return self._filter_functions(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _discover_functions(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.discovery.functions_to_optimize import get_functions_to_optimize

            test_cfg: TestConfig = payload["test_cfg"]
            optimize_all = payload.get("optimize_all")
            replay_test = payload.get("replay_test")
            file_path = payload.get("file")
            function_name = payload.get("function")
            ignore_paths = payload.get("ignore_paths", [])
            project_root = payload["project_root"]
            module_root = payload["module_root"]
            previous_checkpoint_functions = payload.get("previous_checkpoint_functions")

            file_to_funcs, num_functions, trace_file = get_functions_to_optimize(
                optimize_all=optimize_all,
                replay_test=replay_test,
                file=file_path,
                only_get_this_function=function_name,
                test_cfg=test_cfg,
                ignore_paths=ignore_paths,
                project_root=project_root,
                module_root=module_root,
                previous_checkpoint_functions=previous_checkpoint_functions,
            )

            if self.state_store:
                file_to_funcs = self._filter_by_history(file_to_funcs)
                num_functions = sum(len(funcs) for funcs in file_to_funcs.values())

            result_data = {
                "file_to_funcs": file_to_funcs,
                "num_functions": num_functions,
                "trace_file": trace_file,
            }

            logger.info(f"DiscoveryAgent: Found {num_functions} functions to optimize")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"DiscoveryAgent failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _discover_tests(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.discovery.discover_unit_tests import discover_unit_tests

            test_cfg: TestConfig = payload["test_cfg"]
            file_to_funcs: dict[Path, list[FunctionToOptimize]] = payload["file_to_funcs"]

            function_to_tests, num_tests, num_replay_tests = discover_unit_tests(
                test_cfg, file_to_funcs_to_optimize=file_to_funcs
            )

            result_data = {
                "function_to_tests": function_to_tests,
                "num_tests": num_tests,
                "num_replay_tests": num_replay_tests,
            }

            logger.info(f"DiscoveryAgent: Discovered {num_tests} tests and {num_replay_tests} replay tests")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"DiscoveryAgent test discovery failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _filter_functions(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            file_to_funcs: dict[Path, list[FunctionToOptimize]] = payload["file_to_funcs"]
            filtered = self._filter_by_history(file_to_funcs)
            num_functions = sum(len(funcs) for funcs in filtered.values())

            result_data = {
                "file_to_funcs": filtered,
                "num_functions": num_functions,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _filter_by_history(
        self, file_to_funcs: dict[Path, list[FunctionToOptimize]]
    ) -> dict[Path, list[FunctionToOptimize]]:
        if not self.state_store:
            return file_to_funcs

        from codeflash.state.history import OptimizationHistory

        history = OptimizationHistory(self.state_store)
        filtered: dict[Path, list[FunctionToOptimize]] = {}

        for file_path, functions in file_to_funcs.items():
            kept_functions = []
            for func in functions:
                should_skip, reason = history.should_skip_function(func.qualified_name)
                if should_skip:
                    logger.debug(f"Skipping {func.qualified_name}: {reason}")
                    continue
                kept_functions.append(func)

            if kept_functions:
                filtered[file_path] = kept_functions

        return filtered


def create_discovery_task(
    test_cfg: Any,
    project_root: Path,
    module_root: Path,
    optimize_all: Path | str | None = None,
    replay_test: list[Path] | None = None,
    file: Path | None = None,
    function: str | None = None,
    ignore_paths: list[Path] | None = None,
    previous_checkpoint_functions: dict[str, dict[str, str]] | None = None,
) -> AgentTask:
    return AgentTask.create(
        task_type="discover_functions",
        payload={
            "test_cfg": test_cfg,
            "project_root": project_root,
            "module_root": module_root,
            "optimize_all": optimize_all,
            "replay_test": replay_test,
            "file": file,
            "function": function,
            "ignore_paths": ignore_paths or [],
            "previous_checkpoint_functions": previous_checkpoint_functions,
        },
        priority=10,
    )


def create_test_discovery_task(
    test_cfg: Any,
    file_to_funcs: dict[Path, list[Any]],
) -> AgentTask:
    return AgentTask.create(
        task_type="discover_tests",
        payload={
            "test_cfg": test_cfg,
            "file_to_funcs": file_to_funcs,
        },
        priority=9,
    )
