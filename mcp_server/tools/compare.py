from __future__ import annotations

from typing import Any

from mcp_server.db import get_connection, load_test_results


def compare_test_results(original_run_id: str, candidate_run_id: str, pass_fail_only: bool = False) -> dict[str, Any]:
    conn = get_connection()
    try:
        original_results = load_test_results(conn, original_run_id)
        candidate_results = load_test_results(conn, candidate_run_id)
    finally:
        conn.close()

    if not original_results:
        return {"error": f"No results found for run_id: {original_run_id}"}
    if not candidate_results:
        return {"error": f"No results found for run_id: {candidate_run_id}"}

    from codeflash.verification.equivalence import compare_test_results as cf_compare

    equivalent, test_diffs = cf_compare(original_results, candidate_results, pass_fail_only=pass_fail_only)

    diffs = []
    for diff in test_diffs:
        diffs.append(
            {
                "scope": diff.scope.value if hasattr(diff.scope, "value") else str(diff.scope),
                "test_name": diff.test_src_code or "",
                "original_value": diff.original_value or "",
                "candidate_value": diff.candidate_value or "",
                "original_passed": diff.original_pass,
                "candidate_passed": diff.candidate_pass,
            }
        )

    total_compared = min(len(original_results), len(candidate_results))

    return {"equivalent": equivalent, "total_compared": total_compared, "diffs": diffs}
