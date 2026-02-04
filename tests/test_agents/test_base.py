from __future__ import annotations

import pytest

from codeflash.agents.base import (
    AgentMessage,
    AgentResult,
    AgentState,
    AgentTask,
    BaseAgent,
    wrap_error,
    wrap_result,
)
from codeflash.either import Result


class MockAgent(BaseAgent):
    def __init__(self, agent_id: str = "mock", should_fail: bool = False) -> None:
        super().__init__(agent_id=agent_id, coordinator=None)
        self.should_fail = should_fail
        self.process_called = False

    def process(self, task: AgentTask) -> Result:
        self.process_called = True
        if self.should_fail:
            return wrap_error("Mock failure", task.task_id, self.agent_id)
        return wrap_result({"mock_data": "test"}, task.task_id, self.agent_id)


class TestAgentState:
    def test_agent_states(self) -> None:
        assert AgentState.IDLE.value == "idle"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.WAITING.value == "waiting"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"


class TestAgentTask:
    def test_create_task(self) -> None:
        task = AgentTask.create(
            task_type="test_task",
            payload={"key": "value"},
            priority=5,
        )
        assert task.task_type == "test_task"
        assert task.payload == {"key": "value"}
        assert task.priority == 5
        assert task.task_id is not None
        assert task.parent_task_id is None

    def test_task_with_parent(self) -> None:
        task = AgentTask.create(
            task_type="child_task",
            payload={},
            parent_task_id="parent-123",
        )
        assert task.parent_task_id == "parent-123"

    def test_task_comparison(self) -> None:
        high_priority = AgentTask.create("high", {}, priority=10)
        low_priority = AgentTask.create("low", {}, priority=1)
        assert high_priority < low_priority


class TestAgentMessage:
    def test_create_message(self) -> None:
        message = AgentMessage.create(
            sender_id="agent-1",
            message_type="notification",
            payload={"status": "ready"},
        )
        assert message.sender_id == "agent-1"
        assert message.message_type == "notification"
        assert message.payload == {"status": "ready"}
        assert message.message_id is not None
        assert message.timestamp > 0


class TestAgentResult:
    def test_success_result(self) -> None:
        result = AgentResult.success_result(
            task_id="task-1",
            agent_id="agent-1",
            data={"result": "success"},
        )
        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.error is None

    def test_failure_result(self) -> None:
        result = AgentResult.failure_result(
            task_id="task-1",
            agent_id="agent-1",
            error="Something went wrong",
        )
        assert result.success is False
        assert result.data is None
        assert result.error == "Something went wrong"

    def test_result_with_metadata(self) -> None:
        result = AgentResult.success_result(
            task_id="task-1",
            agent_id="agent-1",
            data={},
            metadata={"duration_ms": 100},
        )
        assert result.metadata == {"duration_ms": 100}


class TestBaseAgent:
    def test_agent_initialization(self) -> None:
        agent = MockAgent("test-agent")
        assert agent.agent_id == "test-agent"
        assert agent.state == AgentState.IDLE
        assert agent.coordinator is None
        assert agent.current_task is None

    def test_agent_execute_success(self) -> None:
        agent = MockAgent("test-agent")
        task = AgentTask.create("test", {"data": "value"})

        result = agent.execute(task)

        assert result.is_successful()
        assert agent.state == AgentState.COMPLETED
        assert agent.process_called

    def test_agent_execute_failure(self) -> None:
        agent = MockAgent("test-agent", should_fail=True)
        task = AgentTask.create("test", {})

        result = agent.execute(task)

        assert result.is_failure()
        assert agent.state == AgentState.FAILED

    def test_agent_messaging(self) -> None:
        agent = MockAgent()

        assert not agent.has_pending_messages()

        message = AgentMessage.create("sender", "test", {})
        agent.receive_message(message)

        assert agent.has_pending_messages()

        received = agent.get_next_message()
        assert received == message
        assert not agent.has_pending_messages()

    def test_agent_reset(self) -> None:
        agent = MockAgent()
        agent.state = AgentState.RUNNING
        agent.receive_message(AgentMessage.create("sender", "test", {}))

        agent.reset()

        assert agent.state == AgentState.IDLE
        assert not agent.has_pending_messages()

    def test_get_state_snapshot(self) -> None:
        agent = MockAgent("snapshot-agent")
        snapshot = agent.get_state_snapshot()

        assert snapshot["agent_id"] == "snapshot-agent"
        assert snapshot["state"] == "idle"
        assert snapshot["current_task"] is None


class TestWrapFunctions:
    def test_wrap_result(self) -> None:
        result = wrap_result({"data": "test"}, "task-1", "agent-1")
        assert result.is_successful()
        agent_result = result.unwrap()
        assert agent_result.data == {"data": "test"}

    def test_wrap_error(self) -> None:
        result = wrap_error("error message", "task-1", "agent-1")
        assert result.is_failure()
        assert result.failure() == "error message"
