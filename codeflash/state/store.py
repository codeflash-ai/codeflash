from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from codeflash.state.models import AgentStateSnapshot, OptimizationAttempt, OptimizationStatus, PipelineState


class StateStore:
    def __init__(self, storage_path: Path | None = None) -> None:
        if storage_path is None:
            from codeflash.code_utils.compat import codeflash_temp_dir

            storage_path = codeflash_temp_dir
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "codeflash_agent.db"
        self._init_database()

    def _init_database(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    function_qualified_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    speedup REAL,
                    original_runtime_ns INTEGER,
                    optimized_runtime_ns INTEGER,
                    error_message TEXT,
                    pr_url TEXT,
                    code_hash TEXT,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_function_name
                ON optimization_attempts(function_qualified_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON optimization_attempts(status)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    current_task_id TEXT,
                    last_updated REAL NOT NULL,
                    context TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_states (
                    pipeline_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    function_qualified_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    stages_completed TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def persist_optimization_attempt(self, attempt: OptimizationAttempt) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO optimization_attempts
                (attempt_id, function_qualified_name, file_path, status, started_at,
                 completed_at, speedup, original_runtime_ns, optimized_runtime_ns,
                 error_message, pr_url, code_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    attempt.function_qualified_name,
                    attempt.file_path,
                    attempt.status.value,
                    attempt.started_at,
                    attempt.completed_at,
                    attempt.speedup,
                    attempt.original_runtime_ns,
                    attempt.optimized_runtime_ns,
                    attempt.error_message,
                    attempt.pr_url,
                    attempt.code_hash,
                    json.dumps(attempt.metadata),
                ),
            )
            conn.commit()

    def get_optimization_attempt(self, attempt_id: str) -> OptimizationAttempt | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM optimization_attempts WHERE attempt_id = ?", (attempt_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_optimization_attempt(row)

    def get_function_history(self, qualified_name: str, limit: int = 100) -> list[OptimizationAttempt]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM optimization_attempts
                WHERE function_qualified_name = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (qualified_name, limit),
            )
            return [self._row_to_optimization_attempt(row) for row in cursor.fetchall()]

    def get_recent_attempts(
        self, since_timestamp: float | None = None, status: OptimizationStatus | None = None, limit: int = 100
    ) -> list[OptimizationAttempt]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM optimization_attempts WHERE 1=1"
            params: list[Any] = []

            if since_timestamp is not None:
                query += " AND started_at >= ?"
                params.append(since_timestamp)

            if status is not None:
                query += " AND status = ?"
                params.append(status.value)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [self._row_to_optimization_attempt(row) for row in cursor.fetchall()]

    def was_function_recently_optimized(
        self, qualified_name: str, code_hash: str | None = None, within_days: int = 7
    ) -> bool:
        cutoff_time = time.time() - (within_days * 24 * 60 * 60)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if code_hash:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM optimization_attempts
                    WHERE function_qualified_name = ?
                    AND code_hash = ?
                    AND status = ?
                    AND started_at >= ?
                    """,
                    (qualified_name, code_hash, OptimizationStatus.COMPLETED.value, cutoff_time),
                )
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM optimization_attempts
                    WHERE function_qualified_name = ?
                    AND status = ?
                    AND started_at >= ?
                    """,
                    (qualified_name, OptimizationStatus.COMPLETED.value, cutoff_time),
                )

            count = cursor.fetchone()[0]
            return count > 0

    def persist_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_states
                (agent_id, agent_type, state, current_task_id, last_updated, context)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.agent_id,
                    snapshot.agent_type,
                    snapshot.state,
                    snapshot.current_task_id,
                    snapshot.last_updated,
                    json.dumps(snapshot.context),
                ),
            )
            conn.commit()

    def get_agent_state(self, agent_id: str) -> AgentStateSnapshot | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM agent_states WHERE agent_id = ?", (agent_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return AgentStateSnapshot(
                agent_id=row["agent_id"],
                agent_type=row["agent_type"],
                state=row["state"],
                current_task_id=row["current_task_id"],
                last_updated=row["last_updated"],
                context=json.loads(row["context"]) if row["context"] else {},
            )

    def persist_pipeline_state(self, state: PipelineState) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO pipeline_states
                (pipeline_id, status, current_stage, started_at, completed_at,
                 function_qualified_name, file_path, stages_completed, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.pipeline_id,
                    state.status,
                    state.current_stage,
                    state.started_at,
                    state.completed_at,
                    state.function_qualified_name,
                    state.file_path,
                    json.dumps(state.stages_completed),
                    state.error_message,
                    json.dumps(state.metadata),
                ),
            )
            conn.commit()

    def get_pipeline_state(self, pipeline_id: str) -> PipelineState | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM pipeline_states WHERE pipeline_id = ?", (pipeline_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return PipelineState(
                pipeline_id=row["pipeline_id"],
                status=row["status"],
                current_stage=row["current_stage"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                function_qualified_name=row["function_qualified_name"],
                file_path=row["file_path"],
                stages_completed=json.loads(row["stages_completed"]) if row["stages_completed"] else [],
                error_message=row["error_message"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )

    def cleanup_old_records(self, days: int = 30) -> int:
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM optimization_attempts WHERE started_at < ?", (cutoff_time,))
            deleted_attempts = cursor.rowcount

            cursor.execute("DELETE FROM pipeline_states WHERE started_at < ?", (cutoff_time,))
            deleted_pipelines = cursor.rowcount

            conn.commit()
            return deleted_attempts + deleted_pipelines

    def _row_to_optimization_attempt(self, row: sqlite3.Row) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt_id=row["attempt_id"],
            function_qualified_name=row["function_qualified_name"],
            file_path=row["file_path"],
            status=OptimizationStatus(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            speedup=row["speedup"],
            original_runtime_ns=row["original_runtime_ns"],
            optimized_runtime_ns=row["optimized_runtime_ns"],
            error_message=row["error_message"],
            pr_url=row["pr_url"],
            code_hash=row["code_hash"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
