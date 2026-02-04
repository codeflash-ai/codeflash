from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OptimizationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OptimizationAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str
    function_qualified_name: str
    file_path: str
    status: OptimizationStatus
    started_at: float
    completed_at: float | None = None
    speedup: float | None = None
    original_runtime_ns: int | None = None
    optimized_runtime_ns: int | None = None
    error_message: str | None = None
    pr_url: str | None = None
    code_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        attempt_id: str,
        function_qualified_name: str,
        file_path: str | Path,
        code_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationAttempt:
        return cls(
            attempt_id=attempt_id,
            function_qualified_name=function_qualified_name,
            file_path=str(file_path),
            status=OptimizationStatus.PENDING,
            started_at=time.time(),
            code_hash=code_hash,
            metadata=metadata or {},
        )

    def mark_in_progress(self) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt_id=self.attempt_id,
            function_qualified_name=self.function_qualified_name,
            file_path=self.file_path,
            status=OptimizationStatus.IN_PROGRESS,
            started_at=self.started_at,
            code_hash=self.code_hash,
            metadata=self.metadata,
        )

    def mark_completed(
        self,
        speedup: float,
        original_runtime_ns: int,
        optimized_runtime_ns: int,
        pr_url: str | None = None,
    ) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt_id=self.attempt_id,
            function_qualified_name=self.function_qualified_name,
            file_path=self.file_path,
            status=OptimizationStatus.COMPLETED,
            started_at=self.started_at,
            completed_at=time.time(),
            speedup=speedup,
            original_runtime_ns=original_runtime_ns,
            optimized_runtime_ns=optimized_runtime_ns,
            pr_url=pr_url,
            code_hash=self.code_hash,
            metadata=self.metadata,
        )

    def mark_failed(self, error_message: str) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt_id=self.attempt_id,
            function_qualified_name=self.function_qualified_name,
            file_path=self.file_path,
            status=OptimizationStatus.FAILED,
            started_at=self.started_at,
            completed_at=time.time(),
            error_message=error_message,
            code_hash=self.code_hash,
            metadata=self.metadata,
        )

    def mark_skipped(self, reason: str) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt_id=self.attempt_id,
            function_qualified_name=self.function_qualified_name,
            file_path=self.file_path,
            status=OptimizationStatus.SKIPPED,
            started_at=self.started_at,
            completed_at=time.time(),
            error_message=reason,
            code_hash=self.code_hash,
            metadata=self.metadata,
        )


class AgentStateSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_type: str
    state: str
    current_task_id: str | None = None
    last_updated: float = Field(default_factory=time.time)
    context: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        agent_id: str,
        agent_type: str,
        state: str,
        current_task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentStateSnapshot:
        return cls(
            agent_id=agent_id,
            agent_type=agent_type,
            state=state,
            current_task_id=current_task_id,
            context=context or {},
        )


class PipelineState(BaseModel):
    model_config = ConfigDict(frozen=True)

    pipeline_id: str
    status: str
    current_stage: str | None = None
    started_at: float
    completed_at: float | None = None
    function_qualified_name: str
    file_path: str
    stages_completed: list[str] = Field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        pipeline_id: str,
        function_qualified_name: str,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineState:
        return cls(
            pipeline_id=pipeline_id,
            status="pending",
            started_at=time.time(),
            function_qualified_name=function_qualified_name,
            file_path=str(file_path),
            metadata=metadata or {},
        )
