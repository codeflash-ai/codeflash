"""End-to-end test of the codeflash-mcp workflow.

Tests the full cycle: behavioral baseline → optimize → behavioral candidate → compare → benchmark.
Uses code_to_optimize/bubble_sort.py as the target function.
"""

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CODE_DIR = PROJECT_ROOT / "code_to_optimize"



def get_e2e_test_config_by_language(language: str) -> dict[str, str]:
    if language == "python":
        return {
            "language": language,
            "test_file": str(CODE_DIR / "tests/pytest/test_bubble_sort.py"),
            "module_path": str(CODE_DIR / "bubble_sort.py"),
            "function_name": "sorter",
            "root": str(CODE_DIR),
            "optimized_code": "def sorter(arr):\n    return sorted(arr)",
        }
    if language == "javascript":
        return {
            "language": language,
            "test_file": str(CODE_DIR / "js/code_to_optimize_js/tests/bubble_sort.test.js"),
            "module_path": str(CODE_DIR / "js/code_to_optimize_js/bubble_sort.js"),
            "function_name": "bubbleSort",
            "root": str(CODE_DIR / "js/code_to_optimize_js"),
            "optimized_code": """/**
 * Bubble sort implementation - heavily optimized.
 */

/**
 * Sort an array.
 * @param {number[]} arr - The array to sort
 * @returns {number[]} - The sorted array
 */
function bubbleSort(arr) {
    // V8 native sort is highly optimized in C++
    return arr.slice().sort((a, b) => a - b);
}

module.exports = { bubbleSort };
""",
        }
    return None



def doTest(config: dict[str, str]) -> int:
    print("===========================================================")
    print("*"* 10 + config["language"] + "*"* 10)
    print("===========================================================")
    run_id = "e2e-test-" + str(uuid.uuid4())
    from mcp_server.tools.behavioral import run_behavioral_tests
    from mcp_server.tools.benchmarking import run_benchmarking_tests
    from mcp_server.tools.compare import compare_test_results

    original_code = Path(config["module_path"]).read_text(encoding="utf-8")

    # ── Step 1: Run behavioral baseline (original code) ──────────
    print("\n[1] Running behavioral tests (baseline)...")
    baseline = run_behavioral_tests(
        test_files=[config["test_file"]],
        project_root=config["root"],
        language=config["language"],
        timeout=60,
        run_id=run_id,
        function_name=config["function_name"],
        module_path=config["module_path"],
    )
    print(f"    run_id:      {baseline['run_id']}")
    print(f"    total_tests: {baseline['total_tests']}")
    print(f"    passed:      {baseline['passed']}")
    print(f"    failed:      {baseline['failed']}")
    print(f"    runtime_ns:  {baseline['best_summed_runtime_ns']}")
    if baseline["errors"]:
        print(f"    errors:      {baseline['errors'][:3]}")

    if baseline["passed"] == 0:
        print("\n    FAIL: No tests passed in baseline. Check instrumentation.")
        return 1

    # ── Step 2: Benchmark baseline (original code) ───────────────
    print("\n[2] Running benchmark (baseline, original code)...")
    bench_baseline = run_benchmarking_tests(
        test_files=[config["test_file"]],
        project_root=config["root"],
        language=config["language"],
        timeout=60,
        min_loops=3,
        max_loops=10,
        target_duration_seconds=2.0,
        run_id=run_id + "-bench-baseline",
        function_name=config["function_name"],
        module_path=config["module_path"],
    )
    print(f"    run_id:         {bench_baseline['run_id']}")
    print(f"    total_runtime:  {bench_baseline['best_summed_runtime_ns']}ns")
    print(f"    loops_executed: {bench_baseline['loops_executed']}")

    if bench_baseline["best_summed_runtime_ns"] == 0:
        print("\n    FAIL: Benchmark baseline produced 0 runtime.")
        return 1

    # ── Step 3: Apply optimization ───────────────────────────────
    print("\n[3] Applying optimization to bubble_sort.py...")
    Path(config["module_path"]).write_text(config["optimized_code"], encoding="utf-8")
    print(f"    module: {config['module_path']} (optimized in-place)")

    # ── Step 4: Run behavioral candidate (optimized code) ────────
    print("\n[4] Running behavioral tests (candidate)...")
    candidate = run_behavioral_tests(
        test_files=[config["test_file"]],
        project_root=config["root"],
        language=config["language"],
        timeout=60,
        run_id=run_id + "-candidate",
        function_name=config["function_name"],
        module_path=config["module_path"],
    )
    print(f"    run_id:      {candidate['run_id']}")
    print(f"    total_tests: {candidate['total_tests']}")
    print(f"    passed:      {candidate['passed']}")
    print(f"    failed:      {candidate['failed']}")
    print(f"    runtime_ns:  {candidate['best_summed_runtime_ns']}")

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
        test_files=[config["test_file"]],
        project_root=config["root"],
        language=config["language"],
        timeout=60,
        min_loops=3,
        max_loops=10,
        target_duration_seconds=2.0,
        run_id=run_id + "-bench-candidate",
        baseline_run_id=run_id + "-bench-baseline",
        function_name=config["function_name"],
        module_path=config["module_path"],
    )
    print(f"    run_id:         {bench_candidate['run_id']}")
    print(f"    total_runtime:  {bench_candidate['best_summed_runtime_ns']}ns")
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
    Path(config["module_path"]).write_text(original_code, encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline:   {baseline['passed']}/{baseline['total_tests']} passed")
    print(f"  Candidate:  {candidate['passed']}/{candidate['total_tests']} passed")
    print(f"  Equivalent: {comparison['equivalent']}")
    print(f"  Benchmark baseline:  {bench_baseline['best_summed_runtime_ns'] / 1e6:.2f}ms")
    print(f"  Benchmark candidate: {bench_candidate['best_summed_runtime_ns'] / 1e6:.2f}ms")
    if bench_candidate.get("speedup"):
        print(f"  Speedup:    {bench_candidate['speedup']['speedup_x']}")

    all_passed = (
        baseline["passed"] > 0
        and candidate["passed"] > 0
        and comparison["equivalent"]
        and bench_baseline["best_summed_runtime_ns"] > 0
        and bench_candidate["best_summed_runtime_ns"] > 0
    )
    print(f"\n  Overall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1

def main() -> int:
    print("=" * 60)
    print("MCP WORKFLOW TEST")
    print("=" * 60)
    print("\n")
    langs = ["python", "javascript"]
    for lang in langs:
        exit_code = doTest(get_e2e_test_config_by_language(lang))
        if exit_code != 0:
            return exit_code
    return 0



if __name__ == "__main__":
    sys.exit(main())
