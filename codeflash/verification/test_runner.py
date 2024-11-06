from __future__ import annotations

import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from codeflash.cli_cmds.console import code_print, console, logger
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import TOTAL_LOOPING_TIME
from codeflash.models.models import TestFiles
from codeflash.verification.parse_test_output import CoverageData
from codeflash.verification.test_results import TestType

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext, TestFiles

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
        is_posix = os.name != "nt"

        # Future plans , don't implement now
        # Loop 1 - only runs coverage and gets the binary and xml files
        # will be a slower "analysis" run. No looping

        # Loop 2 - n - Runs only the performance benchmarking loops - very little overhead
        # no coverage, no pickle files, no binary or sqlite files.
        # performance data put on xml and stdout files.

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

            coverage_out_file = get_run_tmp_file(Path("coverage.json"))
            coveragercfile = get_run_tmp_file(Path(".coveragerc"))
            coveragerc_content = (
                "[run]\n"
                f"source = {project_root.as_posix()}\n"
                "branch = True\n"
                "[json]\n"
                f"output = {coverage_out_file.as_posix()}\n"
            )
            coveragercfile.write_text(coveragerc_content)

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
            assert source_file is not None, "source_file must be provided for coverage analysis"
            assert function_name is not None, "function_name must be provided for coverage analysis"
            assert code_context is not None, "code_context must be provided for coverage analysis"
            assert coverage_out_file.exists(), "coverage_out_file must exist for coverage analysis [run_tests]"
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
            f"--timeout={pytest_timeout * 2}",
            "-q",
            f"--junitxml={result_file_path}",
            "-o",
            "junit_logging=all",
            f"--codeflash_seconds={pytest_target_runtime_seconds * 2}",
            f"--codeflash_min_loops={pytest_min_loops}",
            f"--codeflash_max_loops={pytest_max_loops}",
            "--codeflash_loops_scope=session",
        ]

        results = subprocess.run(
            pytest_cmd_list + test_files + pytest_args,
            capture_output=True,
            cwd=cwd,
            env=pytest_test_env,
            text=True,
            timeout=600 * 2,  # TODO: Make this dynamic
            check=False,
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
