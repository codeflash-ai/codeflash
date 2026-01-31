"""Vitest test runner for JavaScript/TypeScript.

This module provides functions for running Vitest tests for behavioral
verification and performance benchmarking.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


def _find_vitest_project_root(file_path: Path) -> Path | None:
    """Find the Vitest project root by looking for vitest/vite config or package.json.

    Traverses up from the given file path to find the nearest directory
    containing vitest.config.js/ts, vite.config.js/ts, or package.json.

    Args:
        file_path: A file path within the Vitest project.

    Returns:
        The project root directory, or None if not found.

    """
    current = file_path.parent if file_path.is_file() else file_path
    while current != current.parent:  # Stop at filesystem root
        # Check for Vitest-specific config files first
        if (
            (current / "vitest.config.js").exists()
            or (current / "vitest.config.ts").exists()
            or (current / "vitest.config.mjs").exists()
            or (current / "vitest.config.mts").exists()
            or (current / "vite.config.js").exists()
            or (current / "vite.config.ts").exists()
            or (current / "vite.config.mjs").exists()
            or (current / "vite.config.mts").exists()
            or (current / "package.json").exists()
        ):
            return current
        current = current.parent
    return None


def _is_vitest_coverage_available(project_root: Path) -> bool:
    """Check if Vitest coverage package is available.

    Args:
        project_root: The project root directory.

    Returns:
        True if @vitest/coverage-v8 or @vitest/coverage-istanbul is installed.

    """
    node_modules = project_root / "node_modules"
    return (node_modules / "@vitest" / "coverage-v8").exists() or (
        node_modules / "@vitest" / "coverage-istanbul"
    ).exists()


def _ensure_runtime_files(project_root: Path) -> None:
    """Ensure JavaScript runtime package is installed in the project.

    Installs codeflash package if not already present.
    The package provides all runtime files needed for test instrumentation.

    Args:
        project_root: The project root directory.

    """
    # Check if package is already installed
    node_modules_pkg = project_root / "node_modules" / "codeflash"
    if node_modules_pkg.exists():
        logger.debug("codeflash already installed")
        return

    # Try to install from local package first (for development)
    local_package_path = Path(__file__).parent.parent.parent.parent / "packages" / "codeflash"
    if local_package_path.exists():
        try:
            result = subprocess.run(
                ["npm", "install", "--save-dev", str(local_package_path)],
                check=False,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.debug("Installed codeflash from local package")
                return
            logger.warning(f"Failed to install local package: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error installing local package: {e}")

    # Try to install from npm registry
    try:
        result = subprocess.run(
            ["npm", "install", "--save-dev", "codeflash"],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.debug("Installed codeflash from npm registry")
            return
        logger.warning(f"Failed to install from npm: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error installing from npm: {e}")

    logger.error("Could not install codeflash. Please install it manually: npm install --save-dev codeflash")


def _build_vitest_behavioral_command(
    test_files: list[Path], timeout: int | None = None, output_file: Path | None = None
) -> list[str]:
    """Build Vitest command for behavioral tests.

    Args:
        test_files: List of test files to run.
        timeout: Optional timeout in seconds.
        output_file: Optional path for JUnit XML output.

    Returns:
        Command list for subprocess execution.

    """
    cmd = [
        "npx",
        "vitest",
        "run",  # Single execution (not watch mode)
        "--reporter=default",
        "--reporter=junit",
        "--no-file-parallelism",  # Serial execution for deterministic timing
    ]

    if output_file:
        cmd.append(f"--outputFile={output_file}")

    if timeout:
        cmd.append(f"--test-timeout={timeout * 1000}")  # Vitest uses milliseconds

    # Add test files as positional arguments (Vitest style)
    cmd.extend(str(f.resolve()) for f in test_files)

    return cmd


def _build_vitest_benchmarking_command(
    test_files: list[Path], timeout: int | None = None, output_file: Path | None = None
) -> list[str]:
    """Build Vitest command for benchmarking tests.

    Args:
        test_files: List of test files to run.
        timeout: Optional timeout in seconds.
        output_file: Optional path for JUnit XML output.

    Returns:
        Command list for subprocess execution.

    """
    cmd = [
        "npx",
        "vitest",
        "run",  # Single execution (not watch mode)
        "--reporter=default",
        "--reporter=junit",
        "--no-file-parallelism",  # Serial execution for consistent benchmarking
    ]

    if output_file:
        cmd.append(f"--outputFile={output_file}")

    if timeout:
        cmd.append(f"--test-timeout={timeout * 1000}")

    # Add test files as positional arguments
    cmd.extend(str(f.resolve()) for f in test_files)

    return cmd


def run_vitest_behavioral_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """Run Vitest tests and return results in a format compatible with pytest output.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Vitest project root (directory containing vitest.config or package.json).
        enable_coverage: Whether to collect coverage information.
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_file_path, subprocess_result, coverage_json_path, None).

    """
    result_file_path = get_run_tmp_file(Path("vitest_results.xml"))

    # Get test files to run
    test_files = [Path(file.instrumented_behavior_file_path) for file in test_paths.test_files]

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        project_root = _find_vitest_project_root(test_files[0])

    # Use the project root, or fall back to provided cwd
    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Vitest working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Coverage output directory - only enable if coverage package is available
    coverage_dir = get_run_tmp_file(Path("vitest_coverage"))
    coverage_available = _is_vitest_coverage_available(effective_cwd) if enable_coverage else False
    coverage_json_path = coverage_dir / "coverage-final.json" if coverage_available else None

    if enable_coverage and not coverage_available:
        logger.debug("Vitest coverage package not installed, running without coverage")

    # Build Vitest command
    vitest_cmd = _build_vitest_behavioral_command(test_files=test_files, timeout=timeout, output_file=result_file_path)

    # Add coverage flags only if coverage is available
    if coverage_available:
        vitest_cmd.extend(["--coverage", "--coverage.reporter=json", f"--coverage.reportsDirectory={coverage_dir}"])

    # Set up environment
    vitest_env = test_env.copy()
    # Set codeflash output file for the vitest helper to write timing/behavior data (SQLite format)
    codeflash_sqlite_file = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))
    vitest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    vitest_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    vitest_env["CODEFLASH_LOOP_INDEX"] = "1"
    vitest_env["CODEFLASH_MODE"] = "behavior"
    # Seed random number generator for reproducible test runs across original and optimized code
    vitest_env["CODEFLASH_RANDOM_SEED"] = "42"

    logger.debug(f"Running Vitest tests with command: {' '.join(vitest_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=vitest_env, timeout=timeout or 600, check=False, text=True, capture_output=True
        )
        result = subprocess.run(vitest_cmd, **run_args)  # noqa: PLW1510
        # Combine stderr into stdout for timing markers
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Vitest result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Vitest tests timed out after {timeout}s")
        result = subprocess.CompletedProcess(
            args=vitest_cmd, returncode=-1, stdout="", stderr="Test execution timed out"
        )
    except FileNotFoundError:
        logger.error("Vitest not found. Make sure Vitest is installed (npm install vitest)")
        result = subprocess.CompletedProcess(
            args=vitest_cmd, returncode=-1, stdout="", stderr="Vitest not found. Run: npm install vitest"
        )
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Vitest behavioral tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result, coverage_json_path, None


def run_vitest_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100,
    target_duration_ms: int = 10_000,
    stability_check: bool = True,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Vitest benchmarking tests with external looping from Python.

    Uses external process-level looping to run tests multiple times and
    collect timing data. This matches the Python pytest approach where
    looping is controlled externally for simplicity.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the entire benchmark run.
        project_root: Vitest project root (directory containing vitest.config or package.json).
        min_loops: Minimum number of loop iterations.
        max_loops: Maximum number of loop iterations.
        target_duration_ms: Target TOTAL duration in milliseconds for all loops.
        stability_check: Whether to enable stability-based early stopping.

    Returns:
        Tuple of (result_file_path, subprocess_result with stdout from all iterations).

    """
    result_file_path = get_run_tmp_file(Path("vitest_perf_results.xml"))

    # Get performance test files
    test_files = [Path(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        project_root = _find_vitest_project_root(test_files[0])

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Vitest benchmarking working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Build Vitest command for performance tests
    vitest_cmd = _build_vitest_benchmarking_command(
        test_files=test_files, timeout=timeout, output_file=result_file_path
    )

    # Base environment setup
    vitest_env = test_env.copy()
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    vitest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    vitest_env["CODEFLASH_TEST_ITERATION"] = "0"
    vitest_env["CODEFLASH_MODE"] = "performance"
    vitest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Internal loop configuration for capturePerf
    vitest_env["CODEFLASH_PERF_LOOP_COUNT"] = str(max_loops)
    vitest_env["CODEFLASH_PERF_MIN_LOOPS"] = str(min_loops)
    vitest_env["CODEFLASH_PERF_TARGET_DURATION_MS"] = str(target_duration_ms)
    vitest_env["CODEFLASH_PERF_STABILITY_CHECK"] = "true" if stability_check else "false"
    vitest_env["CODEFLASH_LOOP_INDEX"] = "1"

    # Total timeout for the entire benchmark run
    total_timeout = max(120, (target_duration_ms // 1000) + 60, timeout or 120)

    logger.debug(f"Running Vitest benchmarking tests: {' '.join(vitest_cmd)}")
    logger.debug(
        f"Vitest benchmarking config: min_loops={min_loops}, max_loops={max_loops}, "
        f"target_duration={target_duration_ms}ms, stability_check={stability_check}"
    )

    total_start_time = time.time()

    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=vitest_env, timeout=total_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(vitest_cmd, **run_args)  # noqa: PLW1510

        # Combine stderr into stdout for timing markers
        stdout = result.stdout or ""
        if result.stderr:
            stdout = stdout + "\n" + result.stderr if stdout else result.stderr

        result = subprocess.CompletedProcess(args=result.args, returncode=result.returncode, stdout=stdout, stderr="")

    except subprocess.TimeoutExpired:
        logger.warning(f"Vitest benchmarking timed out after {total_timeout}s")
        result = subprocess.CompletedProcess(args=vitest_cmd, returncode=-1, stdout="", stderr="Benchmarking timed out")
    except FileNotFoundError:
        logger.error("Vitest not found for benchmarking")
        result = subprocess.CompletedProcess(args=vitest_cmd, returncode=-1, stdout="", stderr="Vitest not found")

    wall_clock_seconds = time.time() - total_start_time
    logger.debug(f"Vitest benchmarking completed in {wall_clock_seconds:.2f}s")

    return result_file_path, result


def run_vitest_line_profile_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    line_profile_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Vitest tests for line profiling.

    This runs tests against source code that has been instrumented with line profiler.
    The instrumentation collects execution counts and timing per line.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the subprocess.
        project_root: Vitest project root (directory containing vitest.config or package.json).
        line_profile_output_file: Path where line profile results will be written.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("vitest_line_profile_results.xml"))

    # Get test files to run - use instrumented behavior files if available, otherwise benchmarking files
    test_files = []
    for file in test_paths.test_files:
        if file.instrumented_behavior_file_path:
            test_files.append(Path(file.instrumented_behavior_file_path))
        elif file.benchmarking_file_path:
            test_files.append(Path(file.benchmarking_file_path))

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        project_root = _find_vitest_project_root(test_files[0])

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Vitest line profiling working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Build Vitest command for line profiling - simple run without benchmarking loops
    vitest_cmd = [
        "npx",
        "vitest",
        "run",
        "--reporter=default",
        "--reporter=junit",
        "--no-file-parallelism",  # Serial execution for consistent line profiling
    ]

    vitest_cmd.append(f"--outputFile={result_file_path}")

    if timeout:
        vitest_cmd.append(f"--test-timeout={timeout * 1000}")

    vitest_cmd.extend(str(f.resolve()) for f in test_files)

    # Set up environment
    vitest_env = test_env.copy()
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_line_profile.sqlite"))
    vitest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    vitest_env["CODEFLASH_TEST_ITERATION"] = "0"
    vitest_env["CODEFLASH_LOOP_INDEX"] = "1"
    vitest_env["CODEFLASH_MODE"] = "line_profile"
    vitest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Pass the line profile output file path to the instrumented code
    if line_profile_output_file:
        vitest_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    subprocess_timeout = timeout or 600

    logger.debug(f"Running Vitest line profile tests: {' '.join(vitest_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=vitest_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(vitest_cmd, **run_args)  # noqa: PLW1510
        # Combine stderr into stdout
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Vitest line profile result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Vitest line profile tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(
            args=vitest_cmd, returncode=-1, stdout="", stderr="Line profile tests timed out"
        )
    except FileNotFoundError:
        logger.error("Vitest not found for line profiling")
        result = subprocess.CompletedProcess(args=vitest_cmd, returncode=-1, stdout="", stderr="Vitest not found")
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Vitest line profile tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result
