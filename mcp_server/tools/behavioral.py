from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from mcp_server.db import get_connection, store_run
from mcp_server.models import BehavioralRunResult, TestInvocationResult
from mcp_server.runner import TestingMode, run_and_parse


def run_behavioral_tests(
    test_files: list[str],
    project_root: str,
    language: str = "python",
    timeout: int = 300,
    run_id: str | None = None,
    function_name: str | None = None,
    module_path: str | None = None,
    test_framework: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())
    project_root_path = Path(project_root).resolve()

    test_results, run_result = run_and_parse(
        mode=TestingMode.BEHAVIORAL,
        test_files=test_files,
        project_root=project_root_path,
        language=language,
        timeout=timeout,
        min_loops=1,
        max_loops=1,
        target_duration_seconds=0.5,  # not used in behavioral mode
        function_name=function_name,
        module_path=module_path,
        test_framework=test_framework,
    )

    conn = get_connection()
    try:
        store_run(
            conn=conn,
            run_id=run_id,
            run_type="behavioral",
            project_root=project_root,
            test_files=test_files,
            test_results=test_results,
            raw_stdout=run_result.stdout or "",
            raw_stderr=run_result.stderr or "",
        )
    finally:
        conn.close()

    invocation_results = []
    errors = []
    for inv in test_results:
        invocation_results.append(
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
        )
        if not inv.did_pass:
            test_name = f"{inv.id.test_module_path}::{inv.id.test_function_name}"
            errors.append(test_name)

    best_summed_runtime_ns = test_results.total_passed_runtime() if test_results else 0
    result = BehavioralRunResult(
        run_id=run_id,
        total_tests=len(test_results),
        passed=sum(1 for r in test_results if r.did_pass),
        failed=sum(1 for r in test_results if not r.did_pass),
        best_summed_runtime_ns=best_summed_runtime_ns,
        test_results=invocation_results,
        errors=errors,
    )

    return {
        "run_id": result.run_id,
        "total_tests": result.total_tests,
        "passed": result.passed,
        "failed": result.failed,
        "best_summed_runtime_ns": result.best_summed_runtime_ns,
        "test_results": [
            {
                "name": r.test_function_name,
                "module": r.test_module_path,
                "class": r.test_class_name,
                "passed": r.did_pass,
                "runtime_ns": r.runtime_ns,
            }
            for r in result.test_results
        ],
        "errors": result.errors,
    }
