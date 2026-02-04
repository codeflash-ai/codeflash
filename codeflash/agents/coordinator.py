from __future__ import annotations

import uuid
from pathlib import Path
from queue import PriorityQueue
from typing import TYPE_CHECKING, Any

from codeflash.agents.base import AgentResult, AgentTask, BaseAgent
from codeflash.cli_cmds.console import console, logger
from codeflash.either import Failure, Result, Success
from codeflash.state.models import AgentStateSnapshot, OptimizationAttempt, PipelineState
from codeflash.state.store import StateStore

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import BestOptimization, CodeOptimizationContext
    from codeflash.verification.verification_utils import TestConfig


class AgentCoordinator:
    def __init__(self, state_store: StateStore | None = None) -> None:
        self.agents: dict[str, BaseAgent] = {}
        self.state_store = state_store or StateStore()
        self.task_queue: PriorityQueue[AgentTask] = PriorityQueue()
        self._initialized = False

    def register_agents(self) -> None:
        from codeflash.agents.analysis_agent import AnalysisAgent
        from codeflash.agents.discovery_agent import DiscoveryAgent
        from codeflash.agents.generation_agent import GenerationAgent
        from codeflash.agents.integration_agent import IntegrationAgent
        from codeflash.agents.selection_agent import SelectionAgent
        from codeflash.agents.verification_agent import VerificationAgent

        self.agents = {
            "discovery": DiscoveryAgent(coordinator=self, state_store=self.state_store),
            "analysis": AnalysisAgent(coordinator=self),
            "generation": GenerationAgent(coordinator=self),
            "verification": VerificationAgent(coordinator=self),
            "selection": SelectionAgent(coordinator=self),
            "integration": IntegrationAgent(coordinator=self),
        }
        self._initialized = True

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        return self.agents.get(agent_id)

    def submit_task(self, task: AgentTask) -> None:
        self.task_queue.put(task)

    def execute_task(self, agent_id: str, task: AgentTask) -> Result[AgentResult[Any], str]:
        agent = self.agents.get(agent_id)
        if agent is None:
            return Failure(f"Agent {agent_id} not found")

        logger.debug(f"Coordinator: Executing task {task.task_type} on agent {agent_id}")
        return agent.execute(task)

    def persist_agent_state(self, agent_id: str, state_snapshot: dict[str, Any]) -> None:
        agent = self.agents.get(agent_id)
        if agent is None:
            return

        snapshot = AgentStateSnapshot.create(
            agent_id=agent_id,
            agent_type=agent.name,
            state=state_snapshot.get("state", "unknown"),
            current_task_id=state_snapshot.get("current_task"),
            context=state_snapshot,
        )
        self.state_store.persist_agent_state(snapshot)

    def notify_failure(self, agent_id: str, error: str) -> None:
        logger.warning(f"Agent {agent_id} failed: {error}")

    def run_optimization_pipeline(
        self,
        function_to_optimize: FunctionToOptimize,
        test_cfg: TestConfig,
        args: Namespace,
        function_to_tests: dict[str, set[Any]] | None = None,
    ) -> Result[BestOptimization, str]:
        if not self._initialized:
            self.register_agents()

        pipeline_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())

        pipeline_state = PipelineState.create(
            pipeline_id=pipeline_id,
            function_qualified_name=function_to_optimize.qualified_name,
            file_path=function_to_optimize.file_path,
        )
        self.state_store.persist_pipeline_state(pipeline_state)

        attempt = OptimizationAttempt.create(
            attempt_id=pipeline_id,
            function_qualified_name=function_to_optimize.qualified_name,
            file_path=function_to_optimize.file_path,
        )
        self.state_store.persist_optimization_attempt(attempt.mark_in_progress())

        logger.info(f"Starting agentic optimization pipeline for {function_to_optimize.qualified_name}")
        console.rule()

        try:
            context_result = self._run_analysis_stage(function_to_optimize, args.project_root)
            if context_result.is_failure():
                return self._handle_pipeline_failure(attempt, context_result.failure())
            code_context = context_result.unwrap()

            candidates_result = self._run_generation_stage(code_context, trace_id, function_to_optimize)
            if candidates_result.is_failure():
                return self._handle_pipeline_failure(attempt, candidates_result.failure())
            candidates = candidates_result.unwrap()

            if not candidates:
                return self._handle_pipeline_failure(attempt, "No optimization candidates generated")

            best_result = self._run_verification_and_selection_stage(
                candidates=candidates,
                code_context=code_context,
                function_to_optimize=function_to_optimize,
                test_cfg=test_cfg,
                args=args,
                function_to_tests=function_to_tests,
                trace_id=trace_id,
            )
            if best_result.is_failure():
                return self._handle_pipeline_failure(attempt, best_result.failure())

            best_optimization = best_result.unwrap()

            integration_result = self._run_integration_stage(
                best_optimization=best_optimization,
                function_to_optimize=function_to_optimize,
                args=args,
            )
            if integration_result.is_failure():
                logger.warning(f"Integration stage failed: {integration_result.failure()}")

            speedup = 1.0
            if best_optimization.runtime > 0:
                original_runtime = best_optimization.winning_benchmarking_test_results.get_best_runtime() or 0
                if original_runtime > 0:
                    speedup = original_runtime / best_optimization.runtime

            completed_attempt = attempt.mark_completed(
                speedup=speedup,
                original_runtime_ns=original_runtime,
                optimized_runtime_ns=best_optimization.runtime,
                pr_url=integration_result.unwrap().get("pr_url") if integration_result.is_successful() else None,
            )
            self.state_store.persist_optimization_attempt(completed_attempt)

            logger.info(f"Agentic optimization complete for {function_to_optimize.qualified_name}")
            return Success(best_optimization)

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return self._handle_pipeline_failure(attempt, str(e))

    def _run_analysis_stage(
        self,
        function_to_optimize: FunctionToOptimize,
        project_root: Path,
    ) -> Result[CodeOptimizationContext, str]:
        from codeflash.agents.analysis_agent import create_context_extraction_task

        task = create_context_extraction_task(function_to_optimize, project_root)
        result = self.execute_task("analysis", task)

        if result.is_failure():
            return Failure(result.failure())

        agent_result = result.unwrap()
        if not agent_result.success:
            return Failure(agent_result.error or "Analysis failed")

        return Success(agent_result.data["code_context"])

    def _run_generation_stage(
        self,
        code_context: CodeOptimizationContext,
        trace_id: str,
        function_to_optimize: FunctionToOptimize,
    ) -> Result[list[Any], str]:
        from codeflash.agents.generation_agent import create_generation_task

        language = function_to_optimize.language or "python"
        is_async = function_to_optimize.is_async

        task = create_generation_task(
            code_context=code_context,
            trace_id=trace_id,
            is_async=is_async,
            language=language,
        )
        result = self.execute_task("generation", task)

        if result.is_failure():
            return Failure(result.failure())

        agent_result = result.unwrap()
        if not agent_result.success:
            return Failure(agent_result.error or "Generation failed")

        return Success(agent_result.data["candidates"])

    def _run_verification_and_selection_stage(
        self,
        candidates: list[Any],
        code_context: CodeOptimizationContext,
        function_to_optimize: FunctionToOptimize,
        test_cfg: TestConfig,
        args: Namespace,
        function_to_tests: dict[str, set[Any]] | None,
        trace_id: str,
    ) -> Result[BestOptimization, str]:
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        function_optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=code_context.read_writable_code.code_strings[0].code
            if code_context.read_writable_code.code_strings
            else "",
            function_to_tests=function_to_tests,
            function_to_optimize_ast=None,
            aiservice_client=self.agents["generation"].aiservice_client
            if "generation" in self.agents
            else None,
            args=args,
        )

        try:
            return function_optimizer.optimize_function()
        finally:
            function_optimizer.executor.shutdown(wait=True)
            function_optimizer.cleanup_generated_files()

    def _run_integration_stage(
        self,
        best_optimization: BestOptimization,
        function_to_optimize: FunctionToOptimize,
        args: Namespace,
    ) -> Result[dict[str, Any], str]:
        from codeflash.agents.integration_agent import create_pr_task

        if getattr(args, "no_pr", False):
            return Success({"pr_created": False, "reason": "PR creation disabled"})

        task = create_pr_task(
            best_optimization=best_optimization,
            function_to_optimize=function_to_optimize,
            args=args,
        )
        result = self.execute_task("integration", task)

        if result.is_failure():
            return Failure(result.failure())

        agent_result = result.unwrap()
        if not agent_result.success:
            return Failure(agent_result.error or "Integration failed")

        return Success(agent_result.data)

    def _handle_pipeline_failure(
        self,
        attempt: OptimizationAttempt,
        error: str,
    ) -> Result[BestOptimization, str]:
        failed_attempt = attempt.mark_failed(error)
        self.state_store.persist_optimization_attempt(failed_attempt)
        return Failure(error)

    def reset_all_agents(self) -> None:
        for agent in self.agents.values():
            agent.reset()
        while not self.task_queue.empty():
            self.task_queue.get_nowait()
