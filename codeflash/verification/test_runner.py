from __future__ import annotations

import os
import shlex
import subprocess

from codeflash.code_utils.code_utils import get_run_tmp_file


def run_tests(
    test_path: str,
    test_framework: str,
    cwd: str | None = None,
    test_env: dict[str, str] | None = None,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    only_run_this_test_function: str | None = None,
) -> tuple[str, subprocess.CompletedProcess]:
    assert test_framework in ["pytest", "unittest"]
    if only_run_this_test_function and "__replay_test" in test_path:
        test_path = test_path + "::" + only_run_this_test_function

    if test_framework == "pytest":
        result_file_path = get_run_tmp_file("pytest_results.xml")
        pytest_cmd_list = shlex.split(pytest_cmd, posix=os.name != "nt")

        results = subprocess.run(
            pytest_cmd_list
            + [
                test_path,
                "--capture=tee-sys",
                f"--timeout={pytest_timeout}",
                "-q",
                f"--junitxml={result_file_path}",
                "-o",
                "junit_logging=all",
            ],
            capture_output=True,
            cwd=cwd,
            env=test_env,
            text=True,
            timeout=600,
            check=False,
        )
    elif test_framework == "unittest":
        result_file_path = get_run_tmp_file("unittest_results.xml")
        results = subprocess.run(
            ["python", "-m", "xmlrunner"]
            + (["-v"] if verbose else [])
            + [test_path]
            + ["--output-file", result_file_path],
            capture_output=True,
            cwd=cwd,
            env=test_env,
            text=True,
            timeout=600,
            check=False,
        )
    else:
        raise ValueError("Invalid test framework -- I only support Pytest and Unittest currently.")
    return result_file_path, results
