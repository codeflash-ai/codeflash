"""End-to-end test of the codeflash-mcp workflow.

Tests the full cycle: behavioral baseline → optimize → behavioral candidate → compare → benchmark.
Uses code_to_optimize/bubble_sort.py as the target function.
"""

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CODE_DIR = PROJECT_ROOT / "code_to_optimize"
TEST_FILE = str(CODE_DIR / "tests/pytest/test_bubble_sort.py")
MODULE_PATH = str(CODE_DIR / "bubble_sort.py")
FUNCTION_NAME = "sorter"
run_id = "e2e-test-" + str(uuid.uuid4())

def main() -> int:
    from mcp_server.tools.behavioral import run_behavioral_tests
    from mcp_server.tools.benchmarking import run_benchmarking_tests
    from mcp_server.tools.compare import compare_test_results

    print("=" * 60)
    print("MCP WORKFLOW TEST")
    print("=" * 60)

    original_code = Path(MODULE_PATH).read_text(encoding="utf-8")

    # ── Step 1: Run behavioral baseline (original code) ──────────
    print("\n[1] Running behavioral tests (baseline)...")
    baseline = run_behavioral_tests(
        test_files=[TEST_FILE],
        project_root=str(CODE_DIR),
        language="python",
        timeout=60,
        run_id=run_id,
        function_name=FUNCTION_NAME,
        module_path=MODULE_PATH,
    )
    print(f"    run_id:      {baseline['run_id']}")
    print(f"    total_tests: {baseline['total_tests']}")
    print(f"    passed:      {baseline['passed']}")
    print(f"    failed:      {baseline['failed']}")
    print(f"    runtime_ns:  {baseline['total_runtime_ns']}")
    if baseline["errors"]:
        print(f"    errors:      {baseline['errors'][:3]}")

    if baseline["passed"] == 0:
        print("\n    FAIL: No tests passed in baseline. Check instrumentation.")
        return 1

    # ── Step 2: Benchmark baseline (original code) ───────────────
    print("\n[2] Running benchmark (baseline, original code)...")
    bench_baseline = run_benchmarking_tests(
        test_files=[TEST_FILE],
        project_root=str(CODE_DIR),
        language="python",
        timeout=60,
        min_loops=3,
        max_loops=10,
        target_duration_seconds=2.0,
        run_id=run_id + "-bench-baseline",
        function_name=FUNCTION_NAME,
        module_path=MODULE_PATH,
    )
    print(f"    run_id:         {bench_baseline['run_id']}")
    print(f"    total_runtime:  {bench_baseline['total_runtime_ns']}ns")
    print(f"    loops_executed: {bench_baseline['loops_executed']}")

    if bench_baseline["total_runtime_ns"] == 0:
        print("\n    FAIL: Benchmark baseline produced 0 runtime.")
        return 1

    # ── Step 3: Apply optimization ───────────────────────────────
    print("\n[3] Applying optimization to bubble_sort.py...")
    optimized_code = """\
def sorter(arr):
    return sorted(arr)
"""
    Path(MODULE_PATH).write_text(optimized_code, encoding="utf-8")
    print(f"    module: {MODULE_PATH} (optimized in-place)")

    # ── Step 4: Run behavioral candidate (optimized code) ────────
    print("\n[4] Running behavioral tests (candidate)...")
    candidate = run_behavioral_tests(
        test_files=[TEST_FILE],
        project_root=str(CODE_DIR),
        language="python",
        timeout=60,
        run_id=run_id + "-candidate",
        function_name=FUNCTION_NAME,
        module_path=MODULE_PATH,
    )
    print(f"    run_id:      {candidate['run_id']}")
    print(f"    total_tests: {candidate['total_tests']}")
    print(f"    passed:      {candidate['passed']}")
    print(f"    failed:      {candidate['failed']}")
    print(f"    runtime_ns:  {candidate['total_runtime_ns']}")

    if candidate["passed"] == 0:
        print("\n    FAIL: No tests passed in candidate.")
        return 1

    # ── Step 5: Compare results ──────────────────────────────────
    print("\n[5] Comparing baseline vs candidate...")
    comparison = compare_test_results(
        original_run_id=run_id,
        candidate_run_id=run_id + "-candidate",
    )
    print(f"    equivalent:     {comparison['equivalent']}")
    print(f"    total_compared: {comparison['total_compared']}")
    if comparison["diffs"]:
        print(f"    diffs:          {comparison['diffs'][:3]}")

    # ── Step 6: Benchmark candidate with speedup ─────────────────
    print("\n[6] Running benchmark (candidate, optimized code)...")
    bench_candidate = run_benchmarking_tests(
        test_files=[TEST_FILE],
        project_root=str(CODE_DIR),
        language="python",
        timeout=60,
        min_loops=3,
        max_loops=10,
        target_duration_seconds=2.0,
        run_id=run_id + "-bench-candidate",
        baseline_run_id=run_id + "-bench-baseline",
        function_name=FUNCTION_NAME,
        module_path=MODULE_PATH,
    )
    print(f"    run_id:         {bench_candidate['run_id']}")
    print(f"    total_runtime:  {bench_candidate['total_runtime_ns']}ns")
    print(f"    loops_executed: {bench_candidate['loops_executed']}")

    if bench_candidate.get("speedup"):
        sp = bench_candidate["speedup"]
        print(f"    speedup_x:      {sp['speedup_x']}")
        print(f"    speedup_pct:    {sp['speedup_pct']}")
        print(f"    baseline_ns:    {sp['baseline_runtime_ns']}")
        print(f"    candidate_ns:   {sp['candidate_runtime_ns']}")
    else:
        print("    speedup:        None (could not compute)")

    # ── Cleanup: restore original source ───────────────────────────
    Path(MODULE_PATH).write_text(original_code, encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline:   {baseline['passed']}/{baseline['total_tests']} passed")
    print(f"  Candidate:  {candidate['passed']}/{candidate['total_tests']} passed")
    print(f"  Equivalent: {comparison['equivalent']}")
    print(f"  Benchmark baseline:  {bench_baseline['total_runtime_ns'] / 1e6:.2f}ms")
    print(f"  Benchmark candidate: {bench_candidate['total_runtime_ns'] / 1e6:.2f}ms")
    if bench_candidate.get("speedup"):
        print(f"  Speedup:    {bench_candidate['speedup']['speedup_x']}")

    all_passed = (
        baseline["passed"] > 0
        and candidate["passed"] > 0
        and comparison["equivalent"]
        and bench_baseline["total_runtime_ns"] > 0
        and bench_candidate["total_runtime_ns"] > 0
    )
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
