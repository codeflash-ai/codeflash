#!/usr/bin/env python3
"""Runner script for all JavaScript/TypeScript end-to-end tests.

This script runs all JS/TS e2e tests and reports results.
Usage:
    python run_js_e2e_tests.py [--test TEST_NAME] [--parallel]

Examples:
    python run_js_e2e_tests.py                     # Run all tests sequentially
    python run_js_e2e_tests.py --test fibonacci    # Run only fibonacci tests
    python run_js_e2e_tests.py --parallel          # Run tests in parallel

"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple


class TestResult(NamedTuple):
    name: str
    success: bool
    duration: float
    output: str


# List of all JS/TS e2e tests - one per module type, each testing different code patterns
JS_E2E_TESTS = [
    # CommonJS - Simple function optimization (recursive fibonacci)
    "end_to_end_test_js_cjs_function.py",
    # TypeScript - Class method optimization (DataProcessor.findDuplicates)
    "end_to_end_test_js_ts_class.py",
    # ES Modules - Async function optimization (processItemsSequential)
    "end_to_end_test_js_esm_async.py",
]


def run_single_test(test_file: str) -> TestResult:
    """Run a single test and return the result."""
    script_dir = Path(__file__).parent
    test_path = script_dir / test_file

    start_time = time.time()
    try:
        result = subprocess.run(
            ["python", str(test_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(script_dir),
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = "Test timed out after 600 seconds"
    except Exception as e:
        success = False
        output = f"Error running test: {e}"

    duration = time.time() - start_time
    return TestResult(name=test_file.replace(".py", ""), success=success, duration=duration, output=output)


def run_tests_sequential(tests: list[str]) -> list[TestResult]:
    """Run tests sequentially."""
    results = []
    for test in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test}")
        print("=" * 60)
        result = run_single_test(test)
        results.append(result)
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"{status} in {result.duration:.1f}s")
        if not result.success:
            print(f"Output:\n{result.output}")
    return results


def run_tests_parallel(tests: list[str], max_workers: int = 4) -> list[TestResult]:
    """Run tests in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_test, test): test for test in tests}
        for future in as_completed(futures):
            test = futures[future]
            result = future.result()
            results.append(result)
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"{status}: {result.name} ({result.duration:.1f}s)")
    return results


def print_summary(results: list[TestResult]) -> None:
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nTotal: {len(results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  ❌ {r.name}")

    total_duration = sum(r.duration for r in results)
    print(f"\nTotal duration: {total_duration:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run JS/TS e2e tests")
    parser.add_argument("--test", type=str, help="Run only tests matching this pattern")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    # Filter tests if pattern specified
    tests = JS_E2E_TESTS
    if args.test:
        tests = [t for t in tests if args.test.lower() in t.lower()]

    if not tests:
        print(f"No tests matching pattern: {args.test}")
        return 1

    print(f"Running {len(tests)} test(s)...")

    # Run tests
    if args.parallel:
        results = run_tests_parallel(tests, args.workers)
    else:
        results = run_tests_sequential(tests)

    # Print summary
    print_summary(results)

    # Return exit code
    failed = [r for r in results if not r.success]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
