from __future__ import annotations

import contextlib
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import custom_addopts, get_run_tmp_file
from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME_EFFECTIVE
from codeflash.code_utils.coverage_utils import prepare_coverage_files
from codeflash.code_utils.shell_utils import get_cross_platform_subprocess_run_args
from codeflash.models.models import TestFiles, TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles

BEHAVIORAL_BLOCKLISTED_PLUGINS = ["benchmark", "codspeed", "xdist", "sugar"]
BENCHMARKING_BLOCKLISTED_PLUGINS = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]


def _find_js_project_root(file_path: Path) -> Path | None:
    """Find the JavaScript/TypeScript project root by looking for package.json.

    Traverses up from the given file path to find the nearest directory
    containing package.json or jest.config.js.

    Args:
        file_path: A file path within the JavaScript project.

    Returns:
        The project root directory, or None if not found.

    """
    current = file_path.parent if file_path.is_file() else file_path
    while current != current.parent:  # Stop at filesystem root
        if (current / "package.json").exists() or (current / "jest.config.js").exists():
            return current
        current = current.parent
    return None


def run_jest_behavioral_tests(
    test_paths: TestFiles, test_env: dict[str, str], cwd: Path, *, timeout: int | None = None
) -> tuple[Path, subprocess.CompletedProcess, None, None]:
    """Run Jest tests and return results in a format compatible with pytest output.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.

    Returns:
        Tuple of (result_file_path, subprocess_result, None, None).

    """
    result_file_path = get_run_tmp_file(Path("jest_results.xml"))

    # Get test files to run
    test_files = [str(file.instrumented_behavior_file_path) for file in test_paths.test_files]

    # Find the JavaScript project root from the test file paths
    # Jest needs to run from the directory containing package.json or jest.config.js
    js_project_root = None
    if test_files:
        first_test_file = Path(test_files[0])
        js_project_root = _find_js_project_root(first_test_file)

    # Use the detected JS project root, or fall back to provided cwd
    effective_cwd = js_project_root if js_project_root else cwd
    logger.debug(f"Jest working directory: {effective_cwd}")

    # Build Jest command
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",  # Run tests serially for consistent timing
        "--forceExit",
    ]

    # Add test pattern if we have specific files
    if test_files:
        # Jest uses regex for test path matching - use full paths for temp directory tests
        test_pattern = "|".join(str(Path(f).resolve()) for f in test_files)
        jest_cmd.append(f"--testPathPattern={test_pattern}")

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
    # Set codeflash output file for the jest helper to write timing/behavior data (SQLite format)
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "behavior"

    logger.debug(f"Running Jest tests with command: {' '.join(jest_cmd)}")

    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=timeout or 600, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        logger.debug(f"Jest result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest tests timed out after {timeout}s")
        result = subprocess.CompletedProcess(args=jest_cmd, returncode=-1, stdout="", stderr="Test execution timed out")
    except FileNotFoundError:
        logger.error("Jest not found. Make sure Jest is installed (npm install jest)")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found. Run: npm install jest jest-junit"
        )

    return result_file_path, result, None, None


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout."""
    logger.debug(f"executing test run with command: {' '.join(cmd_list)}")
    with custom_addopts():
        run_args = get_cross_platform_subprocess_run_args(
            cwd=cwd, env=env, timeout=timeout, check=False, text=True, capture_output=True
        )
        return subprocess.run(cmd_list, **run_args)  # noqa: PLW1510


def run_behavioral_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    *,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    enable_coverage: bool = False,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """Run behavioral tests with optional coverage."""
    if test_framework == "jest":
        return run_jest_behavioral_tests(test_paths, test_env, cwd, timeout=pytest_timeout)
    if test_framework in {"pytest", "unittest"}:
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type == TestType.REPLAY_TEST:
                # Replay tests need specific test targeting because one file contains tests for multiple functions
                if file.tests_in_file:
                    test_files.extend(
                        [
                            str(file.instrumented_behavior_file_path) + "::" + test.test_function
                            for test in file.tests_in_file
                        ]
                    )
            else:
                test_files.append(str(file.instrumented_behavior_file_path))

        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else [SAFE_SYS_EXECUTABLE, "-m", *shlex.split(pytest_cmd, posix=IS_POSIX)]
        )
        test_files = list(set(test_files))  # remove multiple calls in the same test function

        common_pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
        if pytest_timeout is not None:
            common_pytest_args.append(f"--timeout={pytest_timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

        if enable_coverage:
            coverage_database_file, coverage_config_file = prepare_coverage_files()
            # disable jit for coverage
            pytest_test_env["NUMBA_DISABLE_JIT"] = str(1)
            pytest_test_env["TORCHDYNAMO_DISABLE"] = str(1)
            pytest_test_env["PYTORCH_JIT"] = str(0)
            pytest_test_env["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
            pytest_test_env["TF_ENABLE_ONEDNN_OPTS"] = str(0)
            pytest_test_env["JAX_DISABLE_JIT"] = str(0)

            is_windows = sys.platform == "win32"
            if is_windows:
                # On Windows, delete coverage database file directly instead of using 'coverage erase', to avoid locking issues
                if coverage_database_file.exists():
                    with contextlib.suppress(PermissionError, OSError):
                        coverage_database_file.unlink()
            else:
                cov_erase = execute_test_subprocess(
                    shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env, timeout=30
                )  # this cleanup is necessary to avoid coverage data from previous runs, if there are any, then the current run will be appended to the previous data, which skews the results
                logger.debug(cov_erase)
            coverage_cmd = [
                SAFE_SYS_EXECUTABLE,
                "-m",
                "coverage",
                "run",
                f"--rcfile={coverage_config_file.as_posix()}",
                "-m",
            ]

            if pytest_cmd == "pytest":
                coverage_cmd.extend(["pytest"])
            else:
                coverage_cmd.extend(shlex.split(pytest_cmd, posix=IS_POSIX)[1:])

            blocklist_args = [f"-p no:{plugin}" for plugin in BEHAVIORAL_BLOCKLISTED_PLUGINS if plugin != "cov"]
            results = execute_test_subprocess(
                coverage_cmd + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                timeout=600,
            )
            logger.debug(
                f"Result return code: {results.returncode}, "
                f"{'Result stderr:' + str(results.stderr) if results.stderr else ''}"
            )
        else:
            blocklist_args = [f"-p no:{plugin}" for plugin in BEHAVIORAL_BLOCKLISTED_PLUGINS]

            results = execute_test_subprocess(
                pytest_cmd_list + common_pytest_args + blocklist_args + result_args + test_files,
                cwd=cwd,
                env=pytest_test_env,
                timeout=600,  # TODO: Make this dynamic
            )
            logger.debug(
                f"""Result return code: {results.returncode}, {"Result stderr:" + str(results.stderr) if results.stderr else ""}"""
            )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)

    return (
        result_file_path,
        results,
        coverage_database_file if enable_coverage else None,
        coverage_config_file if enable_coverage else None,
    )


def run_line_profile_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    *,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,  # noqa: ARG001
    pytest_max_loops: int = 100_000,  # noqa: ARG001
) -> tuple[Path, subprocess.CompletedProcess]:
    if test_framework in {"pytest", "unittest"}:  # pytest runs both pytest and unittest tests
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        # Always use file path - pytest discovers all tests including parametrized ones
        test_files: list[str] = list(
            {str(file.benchmarking_file_path) for file in test_paths.test_files}
        )  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
        if pytest_timeout is not None:
            pytest_args.append(f"--timeout={pytest_timeout}")
        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in BENCHMARKING_BLOCKLISTED_PLUGINS]
        pytest_test_env["LINE_PROFILE"] = "1"
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,  # TODO: Make this dynamic
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results


def run_jest_benchmarking_tests(
    test_paths: TestFiles,
    test_env: dict[str, str],
    cwd: Path,
    *,
    timeout: int | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    """Run Jest benchmarking tests.

    Args:
        test_paths: TestFiles object containing test file information.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    result_file_path = get_run_tmp_file(Path("jest_perf_results.xml"))

    # Get performance test files
    test_files = [str(file.benchmarking_file_path) for file in test_paths.test_files if file.benchmarking_file_path]

    # Find the JavaScript project root
    js_project_root = None
    if test_files:
        first_test_file = Path(test_files[0])
        js_project_root = _find_js_project_root(first_test_file)

    effective_cwd = js_project_root if js_project_root else cwd
    logger.debug(f"Jest benchmarking working directory: {effective_cwd}")

    # Build Jest command for performance tests
    jest_cmd = [
        "npx",
        "jest",
        "--reporters=default",
        "--reporters=jest-junit",
        "--runInBand",
        "--forceExit",
    ]

    if test_files:
        test_pattern = "|".join(str(Path(f).resolve()) for f in test_files)
        jest_cmd.append(f"--testPathPattern={test_pattern}")

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
    # Set codeflash output file for the jest helper to write timing data (SQLite format)
    codeflash_sqlite_file = get_run_tmp_file(Path("test_return_values_0.sqlite"))
    jest_env["CODEFLASH_OUTPUT_FILE"] = str(codeflash_sqlite_file)
    jest_env["CODEFLASH_TEST_ITERATION"] = "0"
    jest_env["CODEFLASH_LOOP_INDEX"] = "1"
    jest_env["CODEFLASH_MODE"] = "performance"

    logger.debug(f"Running Jest benchmarking tests: {' '.join(jest_cmd)}")

    try:
        run_args = get_cross_platform_subprocess_run_args(
            cwd=effective_cwd, env=jest_env, timeout=timeout or 600, check=False, text=True, capture_output=True
        )
        result = subprocess.run(jest_cmd, **run_args)  # noqa: PLW1510
        logger.debug(f"Jest benchmarking result: returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Jest benchmarking tests timed out after {timeout}s")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Benchmarking tests timed out"
        )
    except FileNotFoundError:
        logger.error("Jest not found for benchmarking")
        result = subprocess.CompletedProcess(
            args=jest_cmd, returncode=-1, stdout="", stderr="Jest not found"
        )

    return result_file_path, result


def run_benchmarking_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    *,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
) -> tuple[Path, subprocess.CompletedProcess]:
    if test_framework == "jest":
        return run_jest_benchmarking_tests(test_paths, test_env, cwd, timeout=pytest_timeout)
    if test_framework in {"pytest", "unittest"}:  # pytest runs both pytest and unittest tests
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        # Always use file path - pytest discovers all tests including parametrized ones
        test_files: list[str] = list(
            {str(file.benchmarking_file_path) for file in test_paths.test_files}
        )  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            "-q",
            "--codeflash_loops_scope=session",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
            "--codeflash_stability_check=true",
        ]
        if pytest_timeout is not None:
            pytest_args.append(f"--timeout={pytest_timeout}")

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        blocklist_args = [f"-p no:{plugin}" for plugin in BENCHMARKING_BLOCKLISTED_PLUGINS]
        results = execute_test_subprocess(
            pytest_cmd_list + pytest_args + blocklist_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,  # TODO: Make this dynamic
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results
