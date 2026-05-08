from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

mcp = FastMCP(
    "codeflash-mcp",
    instructions="Run behavioral tests, compare results, and benchmark performance for code optimization.",
)


@mcp.tool()
def run_behavioral_tests(
    test_files: list[str],
    project_root: str,
    language: str = "python",
    timeout: int = 300,
    run_id: str | None = None,
    function_name: str | None = None,
    module_path: str | None = None,
) -> dict[str, Any]:
    """Run tests and capture function return values + timing for each test invocation.

    Dispatches to the language-specific test runner (pytest for Python, jest/vitest for JS/TS,
    Maven/Gradle for Java). When function_name and module_path are provided, the tool
    instruments test files before execution — injecting timing + return-value capture code
    around calls to the target function. Without these, tests run as-is (pass/fail only,
    no deep return value capture unless the test files are already instrumented).

    Use this to capture a behavioral baseline before optimizing, then again after optimizing
    to verify output equivalence via compare_test_results.

    Args:
        test_files: Absolute paths to test files. Must be absolute, not relative.
        project_root: Absolute path to project root (added to PYTHONPATH/equivalent).
        language: Project language — determines which test runner is invoked.
        timeout: Max seconds before the subprocess is killed.
        run_id: Identifier for this run. Use descriptive IDs like "baseline-exp-1". Auto-generated UUID if omitted.
        function_name: Name of the function being optimized. When provided with module_path, enables automatic instrumentation of test files to capture return values and precise timing.
        module_path: Absolute path to the source file containing the function being optimized. Required together with function_name for instrumentation.

    Returns:
        run_id, total_tests, passed, failed, total_runtime_ns, test_results (per-test detail), errors.

    """
    from mcp_server.tools.behavioral import run_behavioral_tests as impl

    return impl(
        test_files=test_files,
        project_root=project_root,
        language=language,
        timeout=timeout,
        run_id=run_id,
        function_name=function_name,
        module_path=module_path,
    )


@mcp.tool()
def compare_test_results(original_run_id: str, candidate_run_id: str, pass_fail_only: bool = False) -> dict[str, Any]:
    """Deep-compare return values and pass/fail status between two behavioral test runs.

    Loads both runs from the persistent DB, matches test invocations by identity
    (module + class + function + iteration), then compares using codeflash's deep comparator
    which handles numpy arrays, pandas DataFrames, torch tensors, floating-point tolerance,
    and nested structures. This is NOT simple equality — it catches semantic differences that
    repr() would hide (e.g., int vs float, array dtype mismatches).

    Use this after running behavioral tests on both original and optimized code to verify
    the optimization preserves correctness. If equivalent is false, the diffs tell you exactly
    which test produced different output.

    Args:
        original_run_id: Run ID of the baseline run (before optimization).
        candidate_run_id: Run ID of the candidate run (after optimization).
        pass_fail_only: If true, only compare pass/fail status — skip return value comparison. Use when the function intentionally changes output format but not correctness.

    Returns:
        equivalent (bool), total_compared (int), diffs (list of {scope, test_name, original_value, candidate_value, original_passed, candidate_passed}).

    """
    from mcp_server.tools.compare import compare_test_results as impl

    return impl(original_run_id=original_run_id, candidate_run_id=candidate_run_id, pass_fail_only=pass_fail_only)


@mcp.tool()
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
) -> dict[str, Any]:
    """Run tests in multi-loop mode for stable timing, then compute speedup against a baseline.

    Dispatches to the language-specific test runner with benchmarking parameters. Each test
    function runs repeatedly (starting at min_loops, doubling until max_loops or
    target_duration_seconds is reached). Stability checks detect timing variance and add
    iterations to converge. The final runtime is the sum of the minimum time per test case
    across all loops — this filters out GC jitter and OS scheduling noise.

    When function_name and module_path are provided, the tool instruments test files with
    performance-mode capture code (precise nanosecond timing around target function calls,
    no return value serialization overhead). Without these, tests must already contain
    timing markers (e.g., codeflash_wrap calls) for the timing data to be captured.

    If baseline_run_id is provided, computes:
      performance_gain = (baseline_ns - candidate_ns) / candidate_ns
      speedup_x = gain + 1 (e.g., 2.000x means twice as fast)
      speedup_pct = gain * 100 (e.g., 100.0% means twice as fast)

    Use this after correctness is verified via compare_test_results. The speedup field gives
    the authoritative performance measurement for the optimization.

    Args:
        test_files: Absolute paths to test files. Must exercise the optimized function with representative inputs.
        project_root: Absolute path to project root.
        language: Project language — determines which test runner is invoked.
        timeout: Max seconds for the entire benchmark run. Increase for slow functions or high max_loops.
        min_loops: Minimum iterations per test. Higher = more stable but slower.
        max_loops: Maximum iterations. The runner stops early if target_duration_seconds is reached first.
        target_duration_seconds: Target wall-clock time. The runner doubles loops until this is reached.
        run_id: Identifier for this benchmark run. Auto-generated if omitted.
        baseline_run_id: Run ID of a previous benchmark to compare against. Omit for baseline capture.
        function_name: Name of the function being benchmarked. When provided with module_path, enables automatic instrumentation with performance-mode timing capture.
        module_path: Absolute path to the source file containing the function. Required together with function_name.

    Returns:
        run_id, total_runtime_ns, loops_executed, test_results, speedup (null if no baseline_run_id).

    """
    from mcp_server.tools.benchmarking import run_benchmarking_tests as impl

    return impl(
        test_files=test_files,
        project_root=project_root,
        language=language,
        timeout=timeout,
        min_loops=min_loops,
        max_loops=max_loops,
        target_duration_seconds=target_duration_seconds,
        run_id=run_id,
        baseline_run_id=baseline_run_id,
        function_name=function_name,
        module_path=module_path,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
