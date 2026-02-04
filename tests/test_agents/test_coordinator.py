from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.agents.base import AgentResult, AgentTask
from codeflash.agents.coordinator import AgentCoordinator
from codeflash.either import Failure, Success
from codeflash.state.store import StateStore


class TestAgentCoordinator:
    @pytest.fixture
    def coordinator(self) -> AgentCoordinator:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir))
            coord = AgentCoordinator(state_store=store)
            yield coord

    def test_coordinator_initialization(self, coordinator: AgentCoordinator) -> None:
        assert coordinator.agents == {}
        assert coordinator.state_store is not None
        assert not coordinator._initialized

    def test_register_agents(self, coordinator: AgentCoordinator) -> None:
        coordinator.register_agents()

        assert "discovery" in coordinator.agents
        assert "analysis" in coordinator.agents
        assert "generation" in coordinator.agents
        assert "verification" in coordinator.agents
        assert "selection" in coordinator.agents
        assert "integration" in coordinator.agents
        assert coordinator._initialized

    def test_get_agent(self, coordinator: AgentCoordinator) -> None:
        coordinator.register_agents()

        agent = coordinator.get_agent("discovery")
        assert agent is not None
        assert agent.agent_id == "discovery"

        missing = coordinator.get_agent("nonexistent")
        assert missing is None

    def test_submit_task(self, coordinator: AgentCoordinator) -> None:
        task = AgentTask.create("test", {"data": "value"})
        coordinator.submit_task(task)

        assert not coordinator.task_queue.empty()

    def test_execute_task_agent_not_found(self, coordinator: AgentCoordinator) -> None:
        task = AgentTask.create("test", {})
        result = coordinator.execute_task("nonexistent", task)

        assert result.is_failure()
        assert "not found" in result.failure()

    def test_persist_agent_state(self, coordinator: AgentCoordinator) -> None:
        coordinator.register_agents()

        state_snapshot = {
            "state": "running",
            "current_task": "task-123",
        }
        coordinator.persist_agent_state("discovery", state_snapshot)

        retrieved = coordinator.state_store.get_agent_state("discovery")
        assert retrieved is not None
        assert retrieved.state == "running"

    def test_notify_failure_logs_warning(self, coordinator: AgentCoordinator) -> None:
        with patch("codeflash.agents.coordinator.logger") as mock_logger:
            coordinator.notify_failure("test-agent", "Test error message")
            mock_logger.warning.assert_called_once()

    def test_reset_all_agents(self, coordinator: AgentCoordinator) -> None:
        coordinator.register_agents()
        task = AgentTask.create("test", {})
        coordinator.submit_task(task)

        coordinator.reset_all_agents()

        assert coordinator.task_queue.empty()
        for agent in coordinator.agents.values():
            from codeflash.agents.base import AgentState

            assert agent.state == AgentState.IDLE


class TestCoordinatorExecution:
    @pytest.fixture
    def coordinator_with_agents(self) -> AgentCoordinator:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir))
            coord = AgentCoordinator(state_store=store)
            coord.register_agents()
            yield coord

    def test_execute_task_on_registered_agent(self, coordinator_with_agents: AgentCoordinator) -> None:
        from codeflash.agents.analysis_agent import create_numerical_check_task

        task = create_numerical_check_task("def foo(): return 1")
        result = coordinator_with_agents.execute_task("analysis", task)

        assert result.is_successful()
        agent_result = result.unwrap()
        assert agent_result.success
        assert "is_numerical" in agent_result.data
