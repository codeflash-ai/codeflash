from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from codeflash.state.history import OptimizationHistory
from codeflash.state.models import AgentStateSnapshot, OptimizationAttempt, OptimizationStatus, PipelineState
from codeflash.state.store import StateStore


class TestOptimizationStatus:
    def test_status_values(self) -> None:
        assert OptimizationStatus.PENDING.value == "pending"
        assert OptimizationStatus.IN_PROGRESS.value == "in_progress"
        assert OptimizationStatus.COMPLETED.value == "completed"
        assert OptimizationStatus.FAILED.value == "failed"
        assert OptimizationStatus.SKIPPED.value == "skipped"


class TestOptimizationAttempt:
    def test_create_attempt(self) -> None:
        attempt = OptimizationAttempt.create(
            attempt_id="attempt-1",
            function_qualified_name="module.function",
            file_path="/path/to/file.py",
            code_hash="abc123",
        )
        assert attempt.attempt_id == "attempt-1"
        assert attempt.function_qualified_name == "module.function"
        assert attempt.file_path == "/path/to/file.py"
        assert attempt.status == OptimizationStatus.PENDING
        assert attempt.code_hash == "abc123"

    def test_mark_in_progress(self) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py")
        in_progress = attempt.mark_in_progress()

        assert in_progress.status == OptimizationStatus.IN_PROGRESS
        assert in_progress.attempt_id == attempt.attempt_id

    def test_mark_completed(self) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py")
        completed = attempt.mark_completed(
            speedup=2.5,
            original_runtime_ns=1000000,
            optimized_runtime_ns=400000,
            pr_url="https://github.com/test/pr/1",
        )

        assert completed.status == OptimizationStatus.COMPLETED
        assert completed.speedup == 2.5
        assert completed.original_runtime_ns == 1000000
        assert completed.optimized_runtime_ns == 400000
        assert completed.pr_url == "https://github.com/test/pr/1"
        assert completed.completed_at is not None

    def test_mark_failed(self) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py")
        failed = attempt.mark_failed("Test failure message")

        assert failed.status == OptimizationStatus.FAILED
        assert failed.error_message == "Test failure message"
        assert failed.completed_at is not None

    def test_mark_skipped(self) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py")
        skipped = attempt.mark_skipped("Already optimized")

        assert skipped.status == OptimizationStatus.SKIPPED
        assert skipped.error_message == "Already optimized"


class TestAgentStateSnapshot:
    def test_create_snapshot(self) -> None:
        snapshot = AgentStateSnapshot.create(
            agent_id="discovery",
            agent_type="DiscoveryAgent",
            state="running",
            current_task_id="task-123",
            context={"extra": "data"},
        )

        assert snapshot.agent_id == "discovery"
        assert snapshot.agent_type == "DiscoveryAgent"
        assert snapshot.state == "running"
        assert snapshot.current_task_id == "task-123"
        assert snapshot.context == {"extra": "data"}


class TestPipelineState:
    def test_create_pipeline_state(self) -> None:
        state = PipelineState.create(
            pipeline_id="pipeline-1",
            function_qualified_name="module.func",
            file_path="/path/file.py",
        )

        assert state.pipeline_id == "pipeline-1"
        assert state.status == "pending"
        assert state.function_qualified_name == "module.func"


class TestStateStore:
    @pytest.fixture
    def temp_store(self) -> StateStore:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StateStore(Path(tmpdir))

    def test_store_initialization(self, temp_store: StateStore) -> None:
        assert temp_store.db_path.exists()

    def test_persist_and_get_optimization_attempt(self, temp_store: StateStore) -> None:
        attempt = OptimizationAttempt.create("test-id", "func.name", "/file.py")
        temp_store.persist_optimization_attempt(attempt)

        retrieved = temp_store.get_optimization_attempt("test-id")
        assert retrieved is not None
        assert retrieved.attempt_id == "test-id"
        assert retrieved.function_qualified_name == "func.name"

    def test_get_function_history(self, temp_store: StateStore) -> None:
        for i in range(3):
            attempt = OptimizationAttempt.create(f"id-{i}", "module.func", "/file.py")
            temp_store.persist_optimization_attempt(attempt)

        history = temp_store.get_function_history("module.func")
        assert len(history) == 3

    def test_get_recent_attempts(self, temp_store: StateStore) -> None:
        attempt1 = OptimizationAttempt.create("id-1", "func1", "/file1.py")
        attempt2 = OptimizationAttempt.create("id-2", "func2", "/file2.py")

        temp_store.persist_optimization_attempt(attempt1)
        temp_store.persist_optimization_attempt(attempt2)

        recent = temp_store.get_recent_attempts(limit=10)
        assert len(recent) == 2

    def test_was_function_recently_optimized(self, temp_store: StateStore) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py", code_hash="hash123")
        completed = attempt.mark_completed(2.0, 1000, 500)
        temp_store.persist_optimization_attempt(completed)

        assert temp_store.was_function_recently_optimized("func", code_hash="hash123")
        assert not temp_store.was_function_recently_optimized("other_func")

    def test_persist_and_get_agent_state(self, temp_store: StateStore) -> None:
        snapshot = AgentStateSnapshot.create("agent-1", "TestAgent", "running")
        temp_store.persist_agent_state(snapshot)

        retrieved = temp_store.get_agent_state("agent-1")
        assert retrieved is not None
        assert retrieved.agent_type == "TestAgent"

    def test_persist_and_get_pipeline_state(self, temp_store: StateStore) -> None:
        state = PipelineState.create("pipe-1", "func", "/file.py")
        temp_store.persist_pipeline_state(state)

        retrieved = temp_store.get_pipeline_state("pipe-1")
        assert retrieved is not None
        assert retrieved.function_qualified_name == "func"


class TestOptimizationHistory:
    @pytest.fixture
    def history(self) -> OptimizationHistory:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = StateStore(Path(tmpdir))
            yield OptimizationHistory(store)

    def test_record_and_get_attempts(self, history: OptimizationHistory) -> None:
        attempt = OptimizationAttempt.create("id", "func", "/file.py")
        history.record_attempt(attempt)

        attempts = history.get_function_attempts("func")
        assert len(attempts) == 1

    def test_should_skip_recently_optimized(self, history: OptimizationHistory) -> None:
        for i in range(3):
            attempt = OptimizationAttempt.create(f"id-{i}", "func", "/file.py")
            completed = attempt.mark_completed(2.0, 1000, 500)
            history.record_attempt(completed)

        should_skip, reason = history.should_skip_function("func")
        assert should_skip
        assert "successfully optimized" in reason

    def test_should_skip_repeatedly_failed(self, history: OptimizationHistory) -> None:
        for i in range(3):
            attempt = OptimizationAttempt.create(f"id-{i}", "func", "/file.py")
            failed = attempt.mark_failed("error")
            history.record_attempt(failed)

        should_skip, reason = history.should_skip_function("func")
        assert should_skip
        assert "failed" in reason

    def test_get_best_speedup(self, history: OptimizationHistory) -> None:
        for speedup in [1.5, 2.0, 1.8]:
            attempt = OptimizationAttempt.create(f"id-{speedup}", "func", "/file.py")
            completed = attempt.mark_completed(speedup, 1000, int(1000 / speedup))
            history.record_attempt(completed)

        best = history.get_best_speedup("func")
        assert best == 2.0

    def test_get_statistics(self, history: OptimizationHistory) -> None:
        attempt1 = OptimizationAttempt.create("id-1", "func1", "/file.py")
        attempt2 = OptimizationAttempt.create("id-2", "func2", "/file.py")

        history.record_attempt(attempt1.mark_completed(2.0, 1000, 500))
        history.record_attempt(attempt2.mark_failed("error"))

        stats = history.get_statistics()
        assert stats["total_attempts"] == 2
        assert stats["completed"] == 1
        assert stats["failed"] == 1
