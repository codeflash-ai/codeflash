from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME
from codeflash.code_utils.coverage_utils import CoverageData, prepare_coverage_files
from codeflash.models.models import CodeOptimizationContext, TestFiles
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles

is_posix = os.name != "nt"


def execute_test_subprocess(
    cmd_list: list[str], cwd: Path | None, env: dict[str, str] | None, timeout: int = 600
) -> subprocess.CompletedProcess:
    """Execute a subprocess with the given command list, working directory, environment variables, and timeout."""
    logger.debug(f"executing test run with command: {' '.join(cmd_list)}")
    return subprocess.run(cmd_list, capture_output=True, cwd=cwd, env=env, text=True, timeout=timeout, check=False)


def run_tests(
    test_paths: TestFiles,
    test_framework: str,
    test_env: dict[str, str],
    function_name: str | None,
    source_file: Path | None,
    cwd: Path | None = None,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    only_run_these_test_functions: list[str | None] | None = None,
    pytest_target_runtime_seconds: float = TOTAL_LOOPING_TIME,
    pytest_min_loops: int = 5,
    pytest_max_loops: int = 100_000,
    enable_coverage: bool = False,
    code_context: CodeOptimizationContext | None = None,
    project_root: Path | None = None,
) -> tuple[Path, subprocess.CompletedProcess]:
    assert test_framework in ["pytest", "unittest"]

    if test_framework == "pytest":
        test_files = []
        for file in test_paths.test_files:
            if file.test_type == TestType.REPLAY_TEST:
                test_files.append(
                    str(file.instrumented_file_path) + "::" + only_run_these_test_functions[file.instrumented_file_path]
                )
            else:
                test_files.append(str(file.instrumented_file_path))

        if enable_coverage:
            assert project_root is not None, "project_root must be provided for coverage analysis"

            coverage_out_file, coveragercfile = prepare_coverage_files(project_root)

            pytest_ignore_files = [
                "--ignore-glob=build/*",
                "--ignore-glob=dist/*",
                "--ignore-glob=*.egg-info/*",
            ]  # --ignore-glob=path, let's use this to ignore leftover build artifacts from setuptools for local installs https://pip.pypa.io/en/stable/topics/local-project-installs/#build-artifacts

            pytest_test_env = test_env.copy()

            pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
            pytest_args = [
                f"--timeout={pytest_timeout* 2}",
                f"--codeflash_seconds={pytest_target_runtime_seconds * 2}",
                "--codeflash_min_loops=1",
                "--codeflash_max_loops=3",
                "--codeflash_loops_scope=session",
            ]

            cov_erase = execute_test_subprocess(
                shlex.split(f"{sys.executable} -m coverage erase"), cwd=cwd, env=pytest_test_env
            )
            logger.debug(cov_erase)

            files = [str(file.instrumented_file_path) for file in test_paths.test_files]

            cov_run = execute_test_subprocess(
                shlex.split(f"{sys.executable} -m coverage run --rcfile={coveragercfile} -m pytest")
                + files
                + pytest_args
                + pytest_ignore_files,
                cwd=cwd,
                env=pytest_test_env,
            )
            logger.debug(cov_run)

            cov_report = execute_test_subprocess(
                shlex.split(f"{sys.executable} -m coverage json --rcfile={coveragercfile}"),
                cwd=cwd,
                env=pytest_test_env,
            )
            logger.debug(cov_report)

            coveragepy_coverage = CoverageData.load_from_coverage_file(
                coverage_out_file, source_file, function_name, code_context=code_context
            )
            coveragepy_coverage.log_coverage()

        result_file_path = get_run_tmp_file(Path("pytest_results.xml"))
        pytest_cmd_list = shlex.split(pytest_cmd, posix=is_posix)
        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        pytest_args = [
            "--capture=tee-sys",
            f"--timeout={pytest_timeout}",
            "-q",
            f"--junitxml={result_file_path}",
            "-o",
            "junit_logging=all",
            f"--codeflash_seconds={pytest_target_runtime_seconds}",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            "--codeflash_loops_scope=session",
        ]

        results = execute_test_subprocess(
            pytest_cmd_list + test_files + pytest_args,
            cwd=cwd,
            env=pytest_test_env,
            timeout=600,  # TODO: Make this dynamic
        )
    elif test_framework == "unittest":
        result_file_path = get_run_tmp_file(Path("unittest_results.xml"))
        unittest_cmd = ["python", "-m", "unittest"]
        verbosity = ["-v"] if verbose else []
        test_files = [str(file.instrumented_file_path) for file in test_paths.test_files]
        output_file_command = ["--output-file", str(result_file_path)]
        results = execute_test_subprocess(
            unittest_cmd + verbosity + test_files + output_file_command, cwd=cwd, env=test_env
        )
    else:
        raise ValueError("Invalid test framework -- I only support Pytest and Unittest currently.")
    return result_file_path, results
