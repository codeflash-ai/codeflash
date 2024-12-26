from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.compat import IS_POSIX, SAFE_SYS_EXECUTABLE
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME
from codeflash.code_utils.coverage_utils import prepare_coverage_files
from codeflash.models.models import TestFiles
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout."""
    logger.debug(f"executing test run with command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, capture_output=True, cwd=cwd, env=env, text=True, timeout=timeout, check=False)


def run_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    only_run_these_test_functions: dict[Path, str] | None = None,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
    enable_coverage: bool = False,
) -> tuple[Path, subprocess.CompletedProcess, Path | None]:
    assert test_framework in ["pytest", "unittest"]
    coverage_out_file = None
    if test_framework == "pytest":
        run_behavioral_tests_pytest(
            test_paths=test_paths,
            test_framework=test_framework,
            test_env=test_env,
            cwd=cwd,
            pytest_timeout=pytest_timeout,
            pytest_cmd=pytest_cmd,
            verbose=verbose,
            only_run_these_test_functions=only_run_these_test_functions,
            enable_coverage=enable_coverage,
        )
        run_benchmarking_tests_pytest()
    elif test_framework == "unittest":
        result_file_path = get_run_tmp_file(Path("unittest_results.xml"))
        unittest_cmd_list = [SAFE_SYS_EXECUTABLE, "-m", "xmlrunner"]
        log_level = ["-v"] if verbose else []
        files = [str(file.instrumented_file_path) for file in test_paths.test_files]
        output_file = ["--output-file", str(result_file_path)]

        results = execute_test_subprocess(
            unittest_cmd_list + log_level + files + output_file, cwd=cwd, env=test_env, timeout=600
        )

    else:
        msg = "Invalid test framework -- I only support Pytest and Unittest currently."
        raise ValueError(msg)

    return result_file_path, results, coverage_out_file if enable_coverage else None


def run_behavioral_tests_pytest(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    cwd: Path,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    pytest_target_runtime_seconds: int = TOTAL_LOOPING_TIME,
    only_run_these_test_functions: dict[Path, str] | None = None,
    enable_coverage: bool = False,
) -> tuple[Path, subprocess.CompletedProcess, Path | None]:
    assert test_framework == "pytest"
    test_files = []
    for file in test_paths.test_files:
        if file.test_type == TestType.REPLAY_TEST:
            test_files.append(
                str(file.instrumented_file_path) + "::" + only_run_these_test_functions[file.instrumented_file_path]
            )
        else:
            test_files.append(str(file.instrumented_file_path))
    pytest_cmd_list = shlex.split(pytest_cmd, posix=IS_POSIX)

    common_pytest_args = [
        "--capture=tee-sys",
        f"--timeout={pytest_timeout}",
        "-q",
        "--codeflash_loops_scope=session",
        "--codeflash_min_loops=1",
        "--codeflash_max_loops=1",
        f"--codeflash_seconds={pytest_target_runtime_seconds}",  # TODO :This is unnecessary, update the plugin to not ask for this
    ]

    result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
    result_args = [f"--junitxml={result_file_path.as_posix()}", "-o", "junit_logging=all"]

    pytest_test_env = test_env.copy()
    pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"

    if enable_coverage:
        coverage_out_file, coveragercfile = prepare_coverage_files()

        cov_erase = execute_test_subprocess(
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage erase"), cwd=cwd, env=pytest_test_env
        )  # this cleanup is necessary to avoid coverage data from previous runs, if there are any, then the current run will be appended to the previous data, which skews the results
        logger.debug(cov_erase)

        results = execute_test_subprocess(
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage run --rcfile={coveragercfile.as_posix()} -m")
            + pytest_cmd_list
            + common_pytest_args
            + result_args
            + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,
        )
        logger.debug(results)

        cov_report = execute_test_subprocess(
            shlex.split(f"{SAFE_SYS_EXECUTABLE} -m coverage json --rcfile={coveragercfile.as_posix()}"),
            cwd=cwd,
            env=pytest_test_env,
        )  # this will generate a json file with the coverage data
        logger.debug(cov_report)
        if "No data to report." in cov_report.stdout:
            logger.warning("No coverage data to report. Check if the tests are running correctly.")
            console.rule()
            coverage_out_file = None
    else:
        results = execute_test_subprocess(
            pytest_cmd_list + common_pytest_args + result_args + test_files,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,  # TODO: Make this dynamic
        )

    return result_file_path, results, coverage_out_file if enable_coverage else None


def run_benchmarking_tests_pytest(
    test_paths: TestFiles,
    pytest_cmd: str,
    test_env: dict[str, str],
    cwd: Path,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME,
    only_run_these_test_functions: dict[Path, str] | None = None,
    pytest_timeout: int | None = None,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
):
    pytest_cmd_list = shlex.split(pytest_cmd, posix=IS_POSIX)
    test_files = []
    # TODO: For existing unit tests, also add only the specific tests not entire test files
    for file in test_paths.test_files:
        if file.test_type == TestType.REPLAY_TEST:
            test_files.append(
                str(file.benchmarking_file_path)
                + "::"
                + only_run_these_test_functions[
                    file.instrumented_file_path
                ]  # the lookup key should be made more general
            )
        else:
            test_files.append(str(file.benchmarking_file_path))
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
    results = execute_test_subprocess(
        pytest_cmd_list + test_files + pytest_args + result_args,
        cwd=cwd,
        env=pytest_test_env,
        timeout=600,  # TODO: Make this dynamic
    )
    return result_file_path, results
