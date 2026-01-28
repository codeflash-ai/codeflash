"""JavaScript test runner using Jest.

This module provides functions for running Jest tests for behavioral
verification and performance benchmarking.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import (
    STABILITY_CENTER_TOLERANCE,
    STABILITY_SPREAD_TOLERANCE,
    STABILITY_WINDOW_SIZE,
)
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


def _is_esm_project(project_root: Path) -> bool:
    """Check if the project uses ES Modules.

    Detects ESM by checking package.json for "type": "module".

    Args:
        project_root: The project root directory.

    Returns:
        True if the project uses ES Modules, False otherwise.

    """
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                return pkg.get("type") == "module"
        except Exception as e:
            logger.debug(f"Failed to read package.json: {e}")
    return False


def _uses_ts_jest(project_root: Path) -> bool:
    """Check if the project uses ts-jest for TypeScript transformation.

    ts-jest handles ESM transformation internally, so we don't need the
    --experimental-vm-modules flag when it's being used. Adding that flag
    can actually break Jest's module resolution for jest.mock() with relative paths.

    Args:
        project_root: The project root directory.

    Returns:
        True if ts-jest is being used, False otherwise.

    """
    # Check for ts-jest in devDependencies
    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open("r") as f:
                pkg = json.load(f)
                dev_deps = pkg.get("devDependencies", {})
                deps = pkg.get("dependencies", {})
                if "ts-jest" in dev_deps or "ts-jest" in deps:
                    return True
        except Exception as e:
            logger.debug(f"Failed to read package.json for ts-jest detection: {e}")

    # Also check for jest.config with ts-jest preset
    for config_file in ["jest.config.js", "jest.config.cjs", "jest.config.ts", "jest.config.mjs"]:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                content = config_path.read_text()
                if "ts-jest" in content:
                    return True
            except Exception as e:
                logger.debug(f"Failed to read {config_file}: {e}")

    return False


def _configure_esm_environment(jest_env: dict[str, str], project_root: Path) -> None:
    """Configure environment variables for ES Module support in Jest.

    Jest requires --experimental-vm-modules flag for ESM support.
    This is passed via NODE_OPTIONS environment variable.

    IMPORTANT: When ts-jest is being used, we skip adding --experimental-vm-modules
    because ts-jest handles ESM transformation internally. Adding this flag can
    break Jest's module resolution for jest.mock() calls with relative paths.

    Args:
        jest_env: Environment variables dictionary to modify.
        project_root: The project root directory.

    """
    if _is_esm_project(project_root):
        # Skip if ts-jest is being used - it handles ESM internally and
        # --experimental-vm-modules breaks module resolution for relative mocks
        if _uses_ts_jest(project_root):
            logger.debug("Skipping --experimental-vm-modules: ts-jest handles ESM transformation")
            return

        logger.debug("Configuring Jest for ES Module support")
        existing_node_options = jest_env.get("NODE_OPTIONS", "")
        esm_flag = "--experimental-vm-modules"
        if esm_flag not in existing_node_options:
            jest_env["NODE_OPTIONS"] = f"{existing_node_options} {esm_flag}".strip()


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
            ["npm", "install", "--save-dev", "codeflash"], cwd=project_root, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.debug("Installed codeflash from npm registry")
            return
        logger.warning(f"Failed to install from npm: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error installing from npm: {e}")

    logger.error("Could not install codeflash. Please install it manually: npm install --save-dev codeflash")


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

    # Ensure the codeflash npm package is installed
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

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

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


def _parse_timing_from_jest_output(stdout: str) -> dict[str, int]:
    """Parse timing data from Jest stdout markers.

    Extracts timing information from markers like:
    !######testModule:testFunc:funcName:loopIndex:invocationId:durationNs######!

    Args:
        stdout: Jest stdout containing timing markers.

    Returns:
        Dictionary mapping test case IDs to duration in nanoseconds.

    """
    import re

    # Pattern: !######module:testFunc:funcName:loopIndex:invocationId:durationNs######!
    pattern = re.compile(r"!######([^:]+):([^:]*):([^:]+):([^:]+):([^:]+):(\d+)######!")

    timings: dict[str, int] = {}
    for match in pattern.finditer(stdout):
        module, test_class, func_name, _loop_index, invocation_id, duration_ns = match.groups()
        # Create test case ID (same format as Python)
        test_id = f"{module}:{test_class}:{func_name}:{invocation_id}"
        timings[test_id] = int(duration_ns)

    return timings


def _should_stop_stability(
    runtimes: list[int],
    window: int,
    min_window_size: int,
    center_rel_tol: float = STABILITY_CENTER_TOLERANCE,
    spread_rel_tol: float = STABILITY_SPREAD_TOLERANCE,
) -> bool:
    """Check if performance has stabilized (matches Python's pytest_plugin.should_stop exactly).

    This function implements the same stability criteria as the Python pytest_plugin.py
    to ensure consistent behavior between Python and JavaScript performance testing.

    Args:
        runtimes: List of aggregate runtimes (sum of min per test case).
        window: Size of the window to check for stability.
        min_window_size: Minimum number of data points required.
        center_rel_tol: Center tolerance - all recent points must be within this fraction of median.
        spread_rel_tol: Spread tolerance - (max-min)/min must be within this fraction.

    Returns:
        True if performance has stabilized, False otherwise.

    """
    if len(runtimes) < window:
        return False

    if len(runtimes) < min_window_size:
        return False

    recent = runtimes[-window:]

    # Use sorted array for faster median and min/max operations
    recent_sorted = sorted(recent)
    mid = window // 2
    m = recent_sorted[mid] if window % 2 else (recent_sorted[mid - 1] + recent_sorted[mid]) / 2

    # 1) All recent points close to the median
    centered = True
    for r in recent:
        if abs(r - m) / m > center_rel_tol:
            centered = False
            break

    # 2) Window spread is small
    r_min, r_max = recent_sorted[0], recent_sorted[-1]
    if r_min == 0:
        return False
    spread_ok = (r_max - r_min) / r_min <= spread_rel_tol

    return centered and spread_ok


def run_jest_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100,
    target_duration_ms: int = 10_000,  # 10 seconds for benchmarking tests
    stability_check: bool = True,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Jest benchmarking tests with in-process session-level looping.

    Uses a custom Jest runner (codeflash/loop-runner) to loop all tests
    within a single Jest process, eliminating process startup overhead.

    This matches Python's pytest_plugin behavior:
    - All tests are run multiple times within a single Jest process
    - Timing data is collected per iteration
    - Stability is checked within the runner

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the entire benchmark run.
        project_root: JavaScript project root (directory containing package.json).
        min_loops: Minimum number of loop iterations.
        max_loops: Maximum number of loop iterations.
        target_duration_ms: Target TOTAL duration in milliseconds for all loops.
        stability_check: Whether to enable stability-based early stopping.

    Returns:
        Tuple of (result_file_path, subprocess_result with stdout from all iterations).

    """
    result_file_path = get_run_tmp_file(Path("jest_perf_results.xml"))

    # Get performance test files
    test_files = [str(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest benchmarking working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Build Jest command for performance tests with custom loop runner
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Ensure serial execution even though runner enforces it
        "--forceExit",
        "--runner=codeflash/loop-runner",  # Use custom loop runner for in-process looping
    ]

    if test_files:
        jest_cmd.append("--runTestsByPath")
        jest_cmd.extend(str(Path(f).resolve()) for f in test_files)

    if timeout:
        jest_cmd.append(f"--testTimeout={timeout * 1000}")

    # Base environment setup
    jest_env = test_env.copy()
    jest_env["JEST_JUNIT_OUTPUT_FILE"] = str(result_file_path)
    jest_env["JEST_JUNIT_OUTPUT_DIR"] = str(result_file_path.parent)
    jest_env["JEST_JUNIT_OUTPUT_NAME"] = result_file_path.name
    jest_env["JEST_JUNIT_CLASSNAME"] = "{filepath}"
    jest_env["JEST_JUNIT_SUITE_NAME"] = "{filepath}"
    jest_env["JEST_JUNIT_ADD_FILE_ATTRIBUTE"] = "true"
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_MODE"] = "performance"
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"

    # Internal loop configuration for capturePerf (eliminates Jest environment overhead)
    # Looping happens inside capturePerf() for maximum efficiency
    jest_env["CODEFLASH_PERF_LOOP_COUNT"] = str(max_loops)
    jest_env["CODEFLASH_PERF_MIN_LOOPS"] = str(min_loops)
    jest_env["CODEFLASH_PERF_TARGET_DURATION_MS"] = str(target_duration_ms)
    jest_env["CODEFLASH_PERF_STABILITY_CHECK"] = "true" if stability_check else "false"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"  # Initial value for compatibility

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

    # Total timeout for the entire benchmark run (longer than single-loop timeout)
    # Account for startup overhead + target duration + buffer
    total_timeout = max(120, (target_duration_ms // 1000) + 60, timeout or 120)

    logger.debug(f"Running Jest benchmarking tests with in-process loop runner: {' '.join(jest_cmd)}")
    logger.debug(
        f"Jest benchmarking config: min_loops={min_loops}, max_loops={max_loops}, "
        f"target_duration={target_duration_ms}ms, stability_check={stability_check}"
    )

    total_start_time = time.time()

    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=total_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510

        # Combine stderr into stdout for timing markers
        stdout = result.stdout or ""
        if result.stderr:
            stdout = stdout + "\n" + result.stderr if stdout else result.stderr

        # Create result with combined stdout
        result = subprocess.CompletedProcess(
            args=result.args, returncode=result.returncode, stdout=stdout, stderr=""
        )

    except subprocess.TimeoutExpired:
        logger.warning(f"Jest benchmarking timed out after {total_timeout}s")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Benchmarking timed out"
        )
    except FileNotFoundError:
        logger.error("Jest not found for benchmarking")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found")

    wall_clock_seconds = time.time() - total_start_time
    logger.debug(f"Jest benchmarking completed in {wall_clock_seconds:.2f}s")

    return result_file_path, result


def run_jest_line_profile_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
    project_root: Path | None = None,
    line_profile_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Jest tests for line profiling.

    This runs tests against source code that has been instrumented with line profiler.
    The instrumentation collects execution counts and timing per line.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds for the subprocess.
        project_root: JavaScript project root (directory containing package.json).
        line_profile_output_file: Path where line profile results will be written.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("jest_line_profile_results.xml"))

    # Get test files to run - use instrumented behavior files if available, otherwise benchmarking files
    test_files = []
    for file in test_paths.test_files:
        if file.instrumented_behavior_file_path:
            test_files.append(str(file.instrumented_behavior_file_path))
        elif file.benchmarking_file_path:
            test_files.append(str(file.benchmarking_file_path))

    # Use provided project_root, or detect it as fallback
    if project_root is None and test_files:
        first_test_file = Path(test_files[0])
        project_root = _find_node_project_root(first_test_file)

    effective_cwd = project_root if project_root else cwd
    logger.debug(f"Jest line profiling working directory: {effective_cwd}")

    # Ensure the codeflash npm package is installed
    _ensure_runtime_files(effective_cwd)

    # Build Jest command for line profiling - simple run without benchmarking loops
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Run tests serially for consistent line profiling
        "--forceExit",
    ]

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
    jest_env["JEST_JUNIT_INCLUDE_CONSOLE_OUTPUT"] = "true"
    # Set codeflash output file for the jest helper
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_line_profile.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "line_profile"
    # Seed random number generator for reproducibility
    jest_env["CODEFLASH_RANDOM_SEED"] = "42"
    # Pass the line profile output file path to the instrumented code
    if line_profile_output_file:
        jest_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    # Configure ESM support if project uses ES Modules
    _configure_esm_environment(jest_env, effective_cwd)

    subprocess_timeout = timeout or 600

    logger.debug(f"Running Jest line profile tests: {' '.join(jest_cmd)}")

    start_time_ns = time.perf_counter_ns()
    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=subprocess_timeout, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        # Jest sends console.log output to stderr by default - move it to stdout
        if result.stderr and not result.stdout:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stderr, stderr=""
            )
        elif result.stderr:
            result = subprocess.CompletedProcess(
                args=result.args, returncode=result.returncode, stdout=result.stdout + "\n" + result.stderr, stderr=""
            )
        logger.debug(f"Jest line profile result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest line profile tests timed out after {subprocess_timeout}s")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Line profile tests timed out"
        )
    except FileNotFoundError:
        logger.error("Jest not found for line profiling")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found")
    finally:
        wall_clock_ns = time.perf_counter_ns() - start_time_ns
        logger.debug(f"Jest line profile tests completed in {wall_clock_ns / 1e9:.2f}s")

    return result_file_path, result
