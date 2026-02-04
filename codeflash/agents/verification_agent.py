from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent, wrap_error, wrap_result
from codeflash.cli_cmds.console import logger
from codeflash.either import Result

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator
    from codeflash.models.models import OptimizedCandidate, TestFiles
    from codeflash.verification.verification_utils import TestConfig


class VerificationAgent(BaseAgent):
    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        super().__init__(agent_id="verification", coordinator=coordinator)

    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        task_type = task.task_type
        if task_type == "generate_tests":
            return self._generate_tests(task)
        if task_type == "run_behavioral_tests":
            return self._run_behavioral_tests(task)
        if task_type == "run_benchmark_tests":
            return self._run_benchmark_tests(task)
        if task_type == "establish_baseline":
            return self._establish_baseline(task)
        if task_type == "verify_candidate":
            return self._verify_candidate(task)
        return wrap_error(f"Unknown task type: {task_type}", task.task_id, self.agent_id)

    def _generate_tests(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.verification.verifier import generate_tests

            code_context = payload["code_context"]
            function_to_optimize = payload["function_to_optimize"]
            test_cfg: TestConfig = payload["test_cfg"]
            project_root: Path = payload["project_root"]
            trace_id: str = payload["trace_id"]

            generated_tests = generate_tests(
                code_context=code_context,
                function_to_optimize=function_to_optimize,
                test_cfg=test_cfg,
                project_root_path=project_root,
                trace_id=trace_id,
            )

            result_data = {
                "generated_tests": generated_tests,
            }

            logger.debug(f"VerificationAgent: Generated tests for {function_to_optimize.qualified_name}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"VerificationAgent test generation failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _run_behavioral_tests(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.verification.test_runner import run_behavioral_tests

            test_files: TestFiles = payload["test_files"]
            test_cfg: TestConfig = payload["test_cfg"]
            project_root: Path = payload["project_root"]
            test_timeout: int = payload.get("test_timeout", 60)

            test_output = run_behavioral_tests(
                test_files=test_files,
                test_cfg=test_cfg,
                project_root_path=project_root,
                test_timeout=test_timeout,
            )

            result_data = {
                "test_output": test_output,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"VerificationAgent behavioral tests failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _run_benchmark_tests(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.verification.test_runner import run_benchmarking_tests

            test_files: TestFiles = payload["test_files"]
            test_cfg: TestConfig = payload["test_cfg"]
            project_root: Path = payload["project_root"]
            test_timeout: int = payload.get("test_timeout", 300)

            benchmark_output = run_benchmarking_tests(
                test_files=test_files,
                test_cfg=test_cfg,
                project_root_path=project_root,
                test_timeout=test_timeout,
            )

            result_data = {
                "benchmark_output": benchmark_output,
            }

            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"VerificationAgent benchmark tests failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _establish_baseline(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.verification.parse_test_output import parse_test_results
            from codeflash.verification.test_runner import run_behavioral_tests, run_benchmarking_tests

            test_files: TestFiles = payload["test_files"]
            test_cfg: TestConfig = payload["test_cfg"]
            project_root: Path = payload["project_root"]
            function_qualified_name: str = payload["function_qualified_name"]

            behavior_output = run_behavioral_tests(
                test_files=test_files,
                test_cfg=test_cfg,
                project_root_path=project_root,
            )

            behavior_results = parse_test_results(
                test_output=behavior_output,
                function_qualified_name=function_qualified_name,
            )

            benchmark_output = run_benchmarking_tests(
                test_files=test_files,
                test_cfg=test_cfg,
                project_root_path=project_root,
            )

            benchmark_results = parse_test_results(
                test_output=benchmark_output,
                function_qualified_name=function_qualified_name,
            )

            result_data = {
                "behavior_results": behavior_results,
                "benchmark_results": benchmark_results,
                "behavior_output": behavior_output,
                "benchmark_output": benchmark_output,
            }

            logger.debug(f"VerificationAgent: Established baseline for {function_qualified_name}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"VerificationAgent baseline establishment failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)

    def _verify_candidate(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        payload = task.payload
        try:
            from codeflash.verification.equivalence import compare_test_results
            from codeflash.verification.parse_test_output import parse_test_results
            from codeflash.verification.test_runner import run_behavioral_tests, run_benchmarking_tests

            candidate: OptimizedCandidate = payload["candidate"]
            test_files: TestFiles = payload["test_files"]
            test_cfg: TestConfig = payload["test_cfg"]
            project_root: Path = payload["project_root"]
            function_qualified_name: str = payload["function_qualified_name"]
            original_behavior_results = payload["original_behavior_results"]

            behavior_output = run_behavioral_tests(
                test_files=test_files,
                test_cfg=test_cfg,
                project_root_path=project_root,
            )

            behavior_results = parse_test_results(
                test_output=behavior_output,
                function_qualified_name=function_qualified_name,
            )

            is_equivalent, diffs = compare_test_results(
                original_results=original_behavior_results,
                optimized_results=behavior_results,
            )

            benchmark_results = None
            benchmark_output = None
            if is_equivalent:
                benchmark_output = run_benchmarking_tests(
                    test_files=test_files,
                    test_cfg=test_cfg,
                    project_root_path=project_root,
                )

                benchmark_results = parse_test_results(
                    test_output=benchmark_output,
                    function_qualified_name=function_qualified_name,
                )

            result_data = {
                "candidate": candidate,
                "is_equivalent": is_equivalent,
                "diffs": diffs,
                "behavior_results": behavior_results,
                "benchmark_results": benchmark_results,
            }

            logger.debug(f"VerificationAgent: Verified candidate {candidate.optimization_id}, equivalent={is_equivalent}")
            return wrap_result(result_data, task.task_id, self.agent_id)

        except Exception as e:
            logger.exception(f"VerificationAgent candidate verification failed: {e}")
            return wrap_error(str(e), task.task_id, self.agent_id)


def create_test_generation_task(
    code_context: Any,
    function_to_optimize: Any,
    test_cfg: Any,
    project_root: Path,
    trace_id: str,
) -> AgentTask:
    return AgentTask.create(
        task_type="generate_tests",
        payload={
            "code_context": code_context,
            "function_to_optimize": function_to_optimize,
            "test_cfg": test_cfg,
            "project_root": project_root,
            "trace_id": trace_id,
        },
        priority=8,
    )


def create_baseline_task(
    test_files: Any,
    test_cfg: Any,
    project_root: Path,
    function_qualified_name: str,
) -> AgentTask:
    return AgentTask.create(
        task_type="establish_baseline",
        payload={
            "test_files": test_files,
            "test_cfg": test_cfg,
            "project_root": project_root,
            "function_qualified_name": function_qualified_name,
        },
        priority=7,
    )


def create_verification_task(
    candidate: Any,
    test_files: Any,
    test_cfg: Any,
    project_root: Path,
    function_qualified_name: str,
    original_behavior_results: Any,
) -> AgentTask:
    return AgentTask.create(
        task_type="verify_candidate",
        payload={
            "candidate": candidate,
            "test_files": test_files,
            "test_cfg": test_cfg,
            "project_root": project_root,
            "function_qualified_name": function_qualified_name,
            "original_behavior_results": original_behavior_results,
        },
        priority=6,
    )
