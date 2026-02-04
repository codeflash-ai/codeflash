from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from codeflash.either import Failure, Result, Success

if TYPE_CHECKING:
    from codeflash.agents.coordinator import AgentCoordinator


class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    task_id: str
    task_type: str
    payload: dict[str, Any]
    priority: int = 0
    parent_task_id: str | None = None

    @classmethod
    def create(
        cls,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        parent_task_id: str | None = None,
    ) -> AgentTask:
        return cls(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            parent_task_id=parent_task_id,
        )

    def __lt__(self, other: AgentTask) -> bool:
        return self.priority > other.priority


@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    message_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    @classmethod
    def create(
        cls,
        sender_id: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> AgentMessage:
        return cls(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=message_type,
            payload=payload,
        )


T = TypeVar("T")


@dataclass
class AgentResult(Generic[T]):
    task_id: str
    agent_id: str
    success: bool
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(
        cls,
        task_id: str,
        agent_id: str,
        data: T,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult[T]:
        return cls(
            task_id=task_id,
            agent_id=agent_id,
            success=True,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        task_id: str,
        agent_id: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult[T]:
        return cls(
            task_id=task_id,
            agent_id=agent_id,
            success=False,
            error=error,
            metadata=metadata or {},
        )


class BaseAgent(ABC):
    def __init__(self, agent_id: str, coordinator: AgentCoordinator | None = None) -> None:
        self.agent_id = agent_id
        self.state = AgentState.IDLE
        self.coordinator = coordinator
        self.inbox: Queue[AgentMessage] = Queue()
        self.current_task: AgentTask | None = None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def set_coordinator(self, coordinator: AgentCoordinator) -> None:
        self.coordinator = coordinator

    @abstractmethod
    def process(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        pass

    def execute(self, task: AgentTask) -> Result[AgentResult[Any], str]:
        self.state = AgentState.RUNNING
        self.current_task = task

        try:
            result = self.process(task)
            if result.is_successful():
                self.state = AgentState.COMPLETED
            else:
                self.state = AgentState.FAILED
            return result
        except Exception as e:
            self.state = AgentState.FAILED
            self.handle_failure(str(e))
            return Failure(f"Agent {self.agent_id} failed: {e}")
        finally:
            self.current_task = None

    def handle_failure(self, error: str) -> None:
        if self.coordinator:
            self.coordinator.persist_agent_state(self.agent_id, self.get_state_snapshot())
            self.coordinator.notify_failure(self.agent_id, error)

    def get_state_snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "inbox_size": self.inbox.qsize(),
        }

    def receive_message(self, message: AgentMessage) -> None:
        self.inbox.put(message)

    def has_pending_messages(self) -> bool:
        return not self.inbox.empty()

    def get_next_message(self) -> AgentMessage | None:
        if self.inbox.empty():
            return None
        return self.inbox.get_nowait()

    def reset(self) -> None:
        self.state = AgentState.IDLE
        self.current_task = None
        while not self.inbox.empty():
            self.inbox.get_nowait()


def wrap_result(data: Any, task_id: str, agent_id: str) -> Result[AgentResult[Any], str]:
    return Success(AgentResult.success_result(task_id=task_id, agent_id=agent_id, data=data))


def wrap_error(error: str, task_id: str, agent_id: str) -> Result[AgentResult[Any], str]:
    return Failure(error)
