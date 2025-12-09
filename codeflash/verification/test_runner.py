from __future__ import annotations

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
from codeflash.models.models import TestFiles, TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles

BEHAVIORAL_BLOCKLISTED_PLUGINS = ["benchmark", "codspeed", "xdist", "sugar"]
BENCHMARKING_BLOCKLISTED_PLUGINS = ["codspeed", "cov", "benchmark", "profiling", "xdist", "sugar"]


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """
    Execute a subprocess with the given command list, working directory, environment variables, and timeout.

    On Windows, uses Popen with communicate() and process groups for proper cleanup.
    On other platforms, uses subprocess.run with capture_output.
    """
    is_windows = sys.platform == "win32"

    with custom_addopts():
        try:
            if is_windows:
                # WINDOWS SUBPROCESS FIX:
                # On Windows, running pytest with coverage can hang indefinitely due to multiple issues:
                #
                # Problem 1: Pipe buffer deadlocks
                #   - subprocess.run() with file handles can deadlock when the child process
                #     produces output faster than the parent can read it
                #   - Solution: Use Popen.communicate() which properly drains both stdout/stderr
                #     concurrently using threads internally
                #
                # Problem 2: Child process waiting for stdin
                #   - Some Windows processes (especially pytest) may wait for console input
                #   - Solution: Use stdin=subprocess.DEVNULL to explicitly close stdin
                #
                # Problem 3: Orphaned child processes after timeout
                #   - When killing a process on Windows, child processes may not be terminated
                #   - Solution: Use CREATE_NEW_PROCESS_GROUP to allow proper process tree termination
                #
                # See: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate

                # CREATE_NEW_PROCESS_GROUP: Creates process in new group for proper termination
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

                process = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,  # Capture stdout
                    stderr=subprocess.PIPE,  # Capture stderr
                    stdin=subprocess.DEVNULL,  # CRITICAL: Prevents child from waiting for input
                    cwd=cwd,
                    env=env,
                    text=True,  # Return strings instead of bytes
                    creationflags=creationflags,  # Windows-specific process group handling
                )

                try:
                    # communicate() properly drains stdout/stderr avoiding deadlocks
                    stdout_content, stderr_content = process.communicate(timeout=timeout)
                    returncode = process.returncode
                except subprocess.TimeoutExpired:
                    # On Windows, terminate the entire process tree
                    try:
                        process.kill()
                    except OSError:
                        pass
                    # Drain remaining output after killing
                    stdout_content, stderr_content = process.communicate(timeout=5)
                    raise subprocess.TimeoutExpired(
                        cmd_list, timeout, output=stdout_content, stderr=stderr_content
                    )

                return subprocess.CompletedProcess(cmd_list, returncode, stdout_content, stderr_content)
            else:
                # On Linux/Mac, use subprocess.run (works fine there)
                result = subprocess.run(
                    cmd_list, capture_output=True, cwd=cwd, env=env, text=True, timeout=timeout, check=False
                )
                return result
        except subprocess.TimeoutExpired:
            raise
        except Exception:
            raise


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
    """
    Run behavioral tests with optional coverage.

    On Windows, uses --capture=no to avoid subprocess output deadlocks.
    """
    is_windows = sys.platform == "win32"

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

        # On Windows, use --capture=no to avoid subprocess output deadlocks
        # On other platforms, use --capture=tee-sys to both capture and display output
        capture_mode = "--capture=no" if is_windows else "--capture=tee-sys"

        common_pytest_args = [
            capture_mode,
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

            # On Windows, delete coverage database file directly instead of using 'coverage erase'
            # to avoid file locking issues
            if is_windows:
                try:
                    if coverage_database_file.exists():
                        coverage_database_file.unlink()
                except (PermissionError, Exception):
                    pass
            else:
                cov_erase = execute_test_subprocess(
                    shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env, timeout=30
                )

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

            final_cmd = coverage_cmd + common_pytest_args + blocklist_args + result_args + test_files
            results = execute_test_subprocess(
                final_cmd,
                cwd=cwd,
                env=pytest_test_env,
                timeout=60,
            )
        else:
            blocklist_args = [f"-p no:{plugin}" for plugin in BEHAVIORAL_BLOCKLISTED_PLUGINS]

            final_cmd = pytest_cmd_list + common_pytest_args + blocklist_args + result_args + test_files
            results = execute_test_subprocess(
                final_cmd,
                cwd=cwd,
                env=pytest_test_env,
                timeout=60,  # TODO: Make this dynamic
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
            timeout=60,  # TODO: Make this dynamic
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results


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
            timeout=60,  # TODO: Make this dynamic
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results
