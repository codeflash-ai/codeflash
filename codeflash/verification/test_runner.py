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

    logger.debug(f"execute_test_subprocess: Starting. platform={sys.platform}, timeout={timeout}s, cwd={cwd}")

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
                    logger.warning(f"execute_test_subprocess: Process timed out after {timeout}s, terminating...")
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

                logger.debug(
                    f"execute_test_subprocess: Completed. returncode={returncode}, "
                    f"stdout_len={len(stdout_content) if stdout_content else 0}, "
                    f"stderr_len={len(stderr_content) if stderr_content else 0}"
                )

                # Log output for debugging
                if returncode != 0:
                    logger.warning(f"execute_test_subprocess: Non-zero return code: {returncode}")
                    if stderr_content:
                        logger.warning(f"execute_test_subprocess: stderr: {stderr_content[:2000]}")
                    if stdout_content:
                        logger.info(f"execute_test_subprocess: stdout: {stdout_content[:2000]}")
                elif stdout_content or stderr_content:
                    # Log a brief summary even on success for debugging
                    if stderr_content:
                        logger.debug(f"execute_test_subprocess: stderr preview: {stderr_content[:500]}")

                return subprocess.CompletedProcess(cmd_list, returncode, stdout_content, stderr_content)
            else:
                # On Linux/Mac, use subprocess.run (works fine there)
                result = subprocess.run(
                    cmd_list, capture_output=True, cwd=cwd, env=env, text=True, timeout=timeout, check=False
                )
                logger.debug(
                    f"execute_test_subprocess: Completed. returncode={result.returncode}, "
                    f"stdout_len={len(result.stdout) if result.stdout else 0}, "
                    f"stderr_len={len(result.stderr) if result.stderr else 0}"
                )
                return result
        except subprocess.TimeoutExpired:
            logger.warning(f"execute_test_subprocess: TimeoutExpired after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"execute_test_subprocess: Unexpected exception: {type(e).__name__}: {e}")
            raise


def run_behavioral_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    *,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    pytest_target_runtime_seconds: int = TOTAL_LOOPING_TIME_EFFECTIVE,
    enable_coverage: bool = False,
) -> tuple[Path, subprocess.CompletedProcess, Path | None, Path | None]:
    """
    Run behavioral tests with optional coverage.

    On Windows, uses --capture=no to avoid subprocess output deadlocks.
    """
    is_windows = sys.platform == "win32"

    logger.debug(f"run_behavioral_tests: framework={test_framework}, enable_coverage={enable_coverage}")

    if test_framework == "pytest":
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type == TestType.REPLAY_TEST:
                # TODO: Does this work for unittest framework?
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
            f"--timeout={pytest_timeout}",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]

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
                except PermissionError as e:
                    logger.warning(f"run_behavioral_tests: PermissionError deleting coverage database: {e}")
                except Exception as e:
                    logger.warning(f"run_behavioral_tests: Exception deleting coverage database: {e}")
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
    elif test_framework == "unittest":
        if enable_coverage:
            msg = "Coverage is not supported yet for unittest framework"
            raise ValueError(msg)
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_files = [file.instrumented_behavior_file_path for file in test_paths.test_files]
        result_file_path, results = run_unittest_tests(
            verbose=verbose, test_file_paths=test_files, test_env=test_env, cwd=cwd
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
    verbose: bool = False,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,  # noqa: ARG001
    pytest_max_loops: int = 100_000,  # noqa: ARG001
    line_profiler_output_file: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    if test_framework == "pytest":
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type in {TestType.REPLAY_TEST, TestType.EXISTING_UNIT_TEST} and file.tests_in_file:
                test_files.extend(
                    [
                        str(file.benchmarking_file_path)
                        + "::"
                        + (test.test_class + "::" if test.test_class else "")
                        + (test.test_function.split("[", 1)[0] if "[" in test.test_function else test.test_function)
                        for test in file.tests_in_file
                    ]
                )
            else:
                test_files.append(str(file.benchmarking_file_path))
        test_files = list(set(test_files))  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            f"--timeout={pytest_timeout}",
            "-q",
            "--codeflash_loops_scope=session",
            "--codeflash_min_loops=1",
            "--codeflash_max_loops=1",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
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
    elif test_framework == "unittest":
        test_env["CODEFLASH_LOOP_INDEX"] = "1"
        test_env["LINE_PROFILE"] = "1"
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type in {TestType.REPLAY_TEST, TestType.EXISTING_UNIT_TEST} and file.tests_in_file:
                test_files.extend(
                    [
                        str(file.benchmarking_file_path)
                        + "::"
                        + (test.test_class + "::" if test.test_class else "")
                        + (test.test_function.split("[", 1)[0] if "[" in test.test_function else test.test_function)
                        for test in file.tests_in_file
                    ]
                )
            else:
                test_files.append(str(file.benchmarking_file_path))
        test_files = list(set(test_files))  # remove multiple calls in the same test function
        line_profiler_output_file, results = run_unittest_tests(
            verbose=verbose, test_file_paths=[Path(file) for file in test_files], test_env=test_env, cwd=cwd
        )
        logger.debug(
            f"""Result return code: {results.returncode}, {"Result stderr:" + str(results.stderr) if results.stderr else ""}"""
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return line_profiler_output_file, results


def run_benchmarking_tests(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    test_framework: str,
    *,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME_EFFECTIVE,
    verbose: bool = False,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
) -> tuple[Path, subprocess.CompletedProcess]:
    if test_framework == "pytest":
        pytest_cmd_list = (
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m pytest", posix=IS_POSIX)
            if pytest_cmd == "pytest"
            else shlex.split(pytest_cmd)
        )
        test_files: list[str] = []
        for file in test_paths.test_files:
            if file.test_type in {TestType.REPLAY_TEST, TestType.EXISTING_UNIT_TEST} and file.tests_in_file:
                test_files.extend(
                    [
                        str(file.benchmarking_file_path)
                        + "::"
                        + (test.test_class + "::" if test.test_class else "")
                        + (test.test_function.split("[", 1)[0] if "[" in test.test_function else test.test_function)
                        for test in file.tests_in_file
                    ]
                )
            else:
                test_files.append(str(file.benchmarking_file_path))
        test_files = list(set(test_files))  # remove multiple calls in the same test function
        pytest_args = [
            "--capture=tee-sys",
            f"--timeout={pytest_timeout}",
            "-q",
            "--codeflash_loops_scope=session",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
        ]
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
    elif test_framework == "unittest":
        test_files = [file.benchmarking_file_path for file in test_paths.test_files]
        result_file_path, results = run_unittest_tests(
            verbose=verbose, test_file_paths=test_files, test_env=test_env, cwd=cwd
        )
    else:
        msg = f"Unsupported test framework: {test_framework}"
        raise ValueError(msg)
    return result_file_path, results


def run_unittest_tests(
    *, verbose: bool, test_file_paths: list[Path], test_env: dict[str, str], cwd: Path
) -> tuple[Path, subprocess.CompletedProcess]:
    result_file_path = get_run_tmp_file(Path("unittest_results.xml"))
    unittest_cmd_list = [SAFE_SYS_EXECUTABLE, "-m", "xmlrunner"]
    log_level = ["-v"] if verbose else []
    files = [str(file) for file in test_file_paths]
    output_file = ["--output-file", str(result_file_path)]
    results = execute_test_subprocess(
        unittest_cmd_list + log_level + files + output_file, cwd=cwd, env=test_env, timeout=60
    )
    return result_file_path, results
