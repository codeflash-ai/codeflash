from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp_server.db import get_connection, load_run_metadata, store_run
from mcp_server.models import BenchmarkRunResult, SpeedupInfo, TestInvocationResult
from mcp_server.runner import TestingMode, run_and_parse

if TYPE_CHECKING:
    import sqlite3

    from codeflash.models.models import TestResults


def run_benchmarking_tests(
    test_files: list[str],
    project_root: str,
    language: str = "python",
    timeout: int = 300,
    min_loops: int = 5,
    max_loops: int = 100_000,
    target_duration_seconds: float = 10.0,
    run_id: str | None = None,
    baseline_run_id: str | None = None,
    function_name: str | None = None,
    module_path: str | None = None,
    test_framework: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())
    project_root_path = Path(project_root).resolve()

    test_results, run_result = run_and_parse(
        mode=TestingMode.BENCHMARKING,
        test_files=test_files,
        project_root=project_root_path,
        language=language,
        timeout=timeout,
        min_loops=min_loops,
        max_loops=max_loops,
        target_duration_seconds=target_duration_seconds,
        function_name=function_name,
        module_path=module_path,
        test_framework=test_framework,
    )

    conn = get_connection()
    store_run(
        conn=conn,
        run_id=run_id,
        run_type="benchmarking",
        project_root=project_root,
        test_files=test_files,
        test_results=test_results,
        raw_stdout=run_result.stdout or "",
        raw_stderr=run_result.stderr or "",
    )

    speedup_info = None
    if baseline_run_id:
        speedup_info = _compute_speedup(conn, baseline_run_id, test_results)

    best_summed_runtime_ns = test_results.total_passed_runtime() if test_results else 0
    loops_executed = test_results.effective_loop_count() if test_results else 0

    invocation_results = [
        TestInvocationResult(
            test_module_path=inv.id.test_module_path,
            test_class_name=inv.id.test_class_name,
            test_function_name=inv.id.test_function_name,
            function_getting_tested=inv.id.function_getting_tested,
            loop_index=inv.loop_index,
            iteration_id=inv.id.iteration_id,
            runtime_ns=inv.runtime,
            did_pass=inv.did_pass,
            timed_out=inv.timed_out or False,
        )
        for inv in test_results
    ]

    result = BenchmarkRunResult(
        run_id=run_id,
        best_summed_runtime_ns=best_summed_runtime_ns,
        loops_executed=loops_executed,
        test_results=invocation_results,
        speedup=speedup_info,
    )

    output: dict[str, Any] = {
        "run_id": result.run_id,
        "best_summed_runtime_ns": result.best_summed_runtime_ns,
        "loops_executed": result.loops_executed,
        "test_results": [
            {
                "name": r.test_function_name,
                "module": r.test_module_path,
                "class": r.test_class_name,
                "passed": r.did_pass,
                "runtime_ns": r.runtime_ns,
                "loop_index": r.loop_index,
            }
            for r in result.test_results
        ],
    }

    if result.speedup:
        output["speedup"] = {
            "baseline_run_id": result.speedup.baseline_run_id,
            "baseline_runtime_ns": result.speedup.baseline_runtime_ns,
            "candidate_runtime_ns": result.speedup.candidate_runtime_ns,
            "performance_gain": result.speedup.performance_gain,
            "speedup_x": result.speedup.speedup_x,
            "speedup_pct": result.speedup.speedup_pct,
        }

    return output


def _compute_speedup(
    conn: sqlite3.Connection, baseline_run_id: str, candidate_results: TestResults
) -> SpeedupInfo | None:
    from codeflash.result.critic import performance_gain

    baseline_meta = load_run_metadata(conn, baseline_run_id)
    if baseline_meta is None:
        return None

    baseline_runtime_ns = baseline_meta["best_summed_runtime_ns"]
    candidate_runtime_ns = candidate_results.total_passed_runtime() if candidate_results else 0

    if candidate_runtime_ns == 0 or baseline_runtime_ns == 0:
        return None

    gain = performance_gain(original_runtime_ns=baseline_runtime_ns, optimized_runtime_ns=candidate_runtime_ns)

    return SpeedupInfo(
        baseline_run_id=baseline_run_id,
        baseline_runtime_ns=baseline_runtime_ns,
        candidate_runtime_ns=candidate_runtime_ns,
        performance_gain=round(gain, 4),
        speedup_x=f"{gain + 1:.3f}x",
        speedup_pct=f"{gain * 100:.1f}%",
    )
