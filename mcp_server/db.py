from __future__ import annotations

import contextlib
import json
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults
from codeflash.models.test_type import TestType

_DEFAULT_DB_PATH = Path.home() / ".codeflash" / "mcp_results.db"


def get_db_path() -> Path:
    path = Path(os.environ.get("CODEFLASH_MCP_DB_PATH", str(_DEFAULT_DB_PATH)))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            project_root TEXT NOT NULL,
            test_files TEXT NOT NULL,
            total_runtime_ns INTEGER,
            total_tests INTEGER,
            passed INTEGER,
            failed INTEGER,
            loops_executed INTEGER,
            raw_stdout TEXT,
            raw_stderr TEXT
        );

        CREATE TABLE IF NOT EXISTS test_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            test_module_path TEXT,
            test_class_name TEXT,
            test_function_name TEXT,
            function_getting_tested TEXT,
            loop_index INTEGER,
            iteration_id TEXT,
            runtime_ns INTEGER,
            return_value BLOB,
            verification_type TEXT,
            did_pass INTEGER NOT NULL,
            timed_out INTEGER DEFAULT 0,
            error_message TEXT,
            stdout TEXT,
            test_type TEXT
        );
    """)
    conn.commit()


def store_run(
    conn: sqlite3.Connection,
    run_id: str,
    run_type: str,
    project_root: str,
    test_files: list[str],
    test_results: TestResults,
    raw_stdout: str = "",
    raw_stderr: str = "",
) -> None:
    total_runtime_ns = test_results.total_passed_runtime() if test_results else 0
    total_tests = len(test_results)
    passed = sum(1 for r in test_results if r.did_pass)
    failed = total_tests - passed
    loops_executed = test_results.effective_loop_count() if test_results else 0

    conn.execute(
        "INSERT INTO runs (run_id, run_type, created_at, project_root, test_files, "
        "total_runtime_ns, total_tests, passed, failed, loops_executed, raw_stdout, raw_stderr) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            run_type,
            datetime.now(timezone.utc).isoformat(),
            project_root,
            json.dumps(test_files),
            total_runtime_ns,
            total_tests,
            passed,
            failed,
            loops_executed,
            raw_stdout,
            raw_stderr,
        ),
    )

    for invocation in test_results:
        return_value_blob = None
        if invocation.return_value is not None:
            with contextlib.suppress(Exception):
                return_value_blob = pickle.dumps(invocation.return_value)

        conn.execute(
            "INSERT INTO test_invocations (run_id, test_module_path, test_class_name, "
            "test_function_name, function_getting_tested, loop_index, iteration_id, "
            "runtime_ns, return_value, verification_type, did_pass, timed_out, error_message, stdout, test_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                invocation.id.test_module_path,
                invocation.id.test_class_name,
                invocation.id.test_function_name,
                invocation.id.function_getting_tested,
                invocation.loop_index,
                invocation.id.iteration_id,
                invocation.runtime,
                return_value_blob,
                invocation.verification_type,
                int(invocation.did_pass),
                int(invocation.timed_out or False),
                None,
                invocation.stdout,
                invocation.test_type.value if invocation.test_type else None,
            ),
        )
    conn.commit()


def load_test_results(conn: sqlite3.Connection, run_id: str) -> TestResults:
    rows = conn.execute(
        "SELECT test_module_path, test_class_name, test_function_name, function_getting_tested, "
        "loop_index, iteration_id, runtime_ns, return_value, verification_type, did_pass, "
        "timed_out, error_message, stdout, test_type FROM test_invocations WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    test_results = TestResults()
    for row in rows:
        (
            test_module_path,
            test_class_name,
            test_function_name,
            function_getting_tested,
            loop_index,
            iteration_id,
            runtime_ns,
            return_value_blob,
            verification_type,
            did_pass,
            timed_out,
            _,
            stdout,
            test_type_str,
        ) = row

        return_value = None
        if return_value_blob is not None:
            with contextlib.suppress(Exception):
                return_value = pickle.loads(return_value_blob)

        try:
            test_type = TestType(int(test_type_str)) if test_type_str else TestType.EXISTING_UNIT_TEST
        except (ValueError, TypeError):
            test_type = TestType.EXISTING_UNIT_TEST

        test_results.add(
            FunctionTestInvocation(
                loop_index=loop_index or 1,
                id=InvocationId(
                    test_module_path=test_module_path or "",
                    test_class_name=test_class_name,
                    test_function_name=test_function_name or "",
                    function_getting_tested=function_getting_tested or "",
                    iteration_id=iteration_id,
                ),
                file_name=Path("unknown"),
                did_pass=bool(did_pass),
                runtime=runtime_ns,
                test_framework="pytest",
                test_type=test_type,
                return_value=return_value,
                timed_out=bool(timed_out),
                verification_type=verification_type,
                stdout=stdout,
            )
        )
    return test_results


def load_run_metadata(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT run_type, created_at, project_root, test_files, total_runtime_ns, "
        "total_tests, passed, failed, loops_executed FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_type": row[0],
        "created_at": row[1],
        "project_root": row[2],
        "test_files": json.loads(row[3]),
        "total_runtime_ns": row[4],
        "total_tests": row[5],
        "passed": row[6],
        "failed": row[7],
        "loops_executed": row[8],
    }
