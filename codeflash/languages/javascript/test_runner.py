"""JavaScript test runner using Jest.

This module provides functions for running Jest tests for behavioral
verification and performance benchmarking.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


def _find_node_project_root(file_path: Path) -> Path | None:
    """Find the Node.js project root by looking for package.json.

    Traverses up from the given file path to find the nearest directory
    containing package.json or jest.config.js.

    Args:
        file_path: A file path within the Node.js project.

    Returns:
        The project root directory, or None if not found.

    """
    current = file_path.parent if file_path.is_file() else file_path
    while current != current.parent:  # Stop at filesystem root
        if (
            (current / "package.json").exists()
            or (current / "jest.config.js").exists()
            or (current / "jest.config.ts").exists()
            or (current / "tsconfig.json").exists()
        ):
            return current
        current = current.parent
    return None


def _ensure_runtime_files(project_root: Path) -> None:
    """Ensure JavaScript runtime files are present in the project root.

    Copies codeflash-jest-helper.js and related files to the project root
    if they don't already exist or are outdated.

    Args:
        project_root: The project root directory.

    """
    from codeflash.languages.javascript.runtime import get_all_runtime_files

    for runtime_file in get_all_runtime_files():
        dest_path = project_root / runtime_file.name
        # Always copy to ensure we have the latest version
        if not dest_path.exists() or dest_path.stat().st_mtime < runtime_file.stat().st_mtime:
            shutil.copy2(runtime_file, dest_path)
            logger.debug(f"Copied {runtime_file.name} to {project_root}")


def run_jest_behavioral_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """Run Jest tests and return results in a format compatible with pytest output.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: JavaScript project root (directory containing package.json).
        enable_coverage: Whether to collect coverage information.
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_file_path, subprocess_result, coverage_json_path, None).

    """
    result_file_path = get_run_tmp_file(Path("jest_results.xml"))

    # Get test files to run
    test_files = [str(file.instrumented_behavior_file_path) for file in test_paths.test_files]

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    # Use the project root, or fall back to provided cwd
    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest working directory: {effective_cwd}")

    # Ensure runtime files (codeflash-jest-helper.js, etc.) are present
    _ensure_runtime_files(effective_cwd)

    # Coverage output directory
    coverage_dir = get_run_tmp_file(Path("jest_coverage"))
    coverage_json_path = coverage_dir / "coverage-final.json" if enable_coverage else None

    # Build Jest command
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Run tests serially for consistent timing
        "--forceExit",
    ]

    # Add coverage flags if enabled
    if enable_coverage:
        jest_cmd.extend(["--coverage", "--coverageReporters=json", f"--coverageDirectory={coverage_dir}"])

    if test_files:
        jest_cmd.append("--runTestsByPath")
        jest_cmd.extend(str(Path(f).resolve()) for f in test_files)

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")  # Jest uses milliseconds

    # Set up environment
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    # Configure jest-junit to use filepath-based classnames for proper parsing
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    # Include console.log output in JUnit XML for timing marker parsing
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    # Set codeflash output file for the jest helper to write timing/behavior data (SQLite format)
    # Use candidate_index to differentiate between baseline (0) and optimization candidates
    codeflash_sqlite_file = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "behavior"
    # Seed random number generator for reproducible test runs across original and optimized code
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"

    logger.debug(f"Running Jest tests with command: {' '.join(jest_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=timeout or 600, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        # Jest sends console.log output to stderr by default - move it to stdout
        # so our timing markers (printed via console.log) are in the expected place
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            # Combine stderr into stdout if both have content
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Jest result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest tests timed out after {timeout}s")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Test execution timed out")
    except FileNotFoundError:
        logger.error("Jest not found. Make sure Jest is installed (npm install jest)")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found. Run: npm install jest jest-junit"
        )
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Jest behavioral tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result, coverage_json_path, None


def run_jest_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100_000,
    target_duration_ms: int = 10_000,
    stability_check: bool = True,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Jest benchmarking tests with internal looping for stable measurements.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the subprocess.
        project_root: JavaScript project root (directory containing package.json).
        min_loops: Minimum number of loops to run for each test case.
        max_loops: Maximum number of loops to run for each test case.
        target_duration_ms: Target TOTAL duration in milliseconds for looping.
            This is divided among test cases since JavaScript uses capturePerfLooped
            which loops internally per test case, unlike Python's external looping.
        stability_check: Whether to enable stability-based early stopping.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("jest_perf_results.xml"))

    # Get performance test files
    test_files = [str(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]

    # Count approximate number of test cases to divide time budget
    # JavaScript's capturePerfLooped loops internally per test case, so we need to divide
    # the total time budget among test cases to avoid timeout
    num_test_cases = len(test_files) * 10  # Estimate ~10 test cases per file (conservative)
    # Use at least 500ms per test case for fast functions, cap at 2 seconds
    per_test_duration_ms = max(500, min(2000, target_duration_ms // max(1, num_test_cases)))

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest benchmarking working directory: {effective_cwd}")

    # Ensure runtime files (codeflash-jest-helper.js, etc.) are present
    _ensure_runtime_files(effective_cwd)

    # Build Jest command for performance tests
    jest_cmd = ["npx", "jest", "--reporters=default", "--reporters=jest-junit", "--runInBand", "--forceExit"]

    if test_files:
        jest_cmd.append("--runTestsByPath")
        jest_cmd.extend(str(Path(f).resolve()) for f in test_files)

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")

    # Set up environment
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    # Include console.log output in JUnit XML for timing marker parsing
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    # Set codeflash output file for the jest helper to write timing data (SQLite format)
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "performance"
    # Looping configuration for stable performance measurements
    jest_env["CODEFLASH_MIN_LOOPS"] = str(min_loops)
    jest_env["CODEFLASH_MAX_LOOPS"] = str(max_loops)
    # Use per-test duration instead of total duration
    jest_env["CODEFLASH_TARGET_DURATION_MS"] = str(per_test_duration_ms)
    jest_env["CODEFLASH_STABILITY_CHECK"] = "true" if stability_check else "false"
    # Seed random number generator for reproducible test runs across original and optimized code
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Use 10 minutes subprocess timeout - sufficient for benchmarking suite
    subprocess_timeout = 600

    logger.debug(f"Running Jest benchmarking tests: {' '.join(jest_cmd)}")
    logger.debug(
        f"Jest benchmarking config: {num_test_cases} estimated test cases, "
        f"{per_test_duration_ms}ms per test, min_loops={min_loops}, {subprocess_timeout}s subprocess timeout"
    )

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        # Jest sends console.log output to stderr by default - move it to stdout
        # so our timing markers (printed via console.log) are in the expected place
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            # Combine stderr into stdout if both have content
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Jest benchmarking result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest benchmarking tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Benchmarking tests timed out"
        )
    except FileNotFoundError:
        logger.error("Jest not found for benchmarking")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found")
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Jest benchmarking tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result
