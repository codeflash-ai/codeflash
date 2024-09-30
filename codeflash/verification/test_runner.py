from __future__ import annotations

import os
import shlex
import subprocess

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.config_consts import REPEAT_COUNT


def run_tests(
    test_paths: list[str],
    test_framework: str,
    cwd: str | None = None,
    test_env: dict[str, str] | None = None,
    pytest_timeout: int | None = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    only_run_these_test_functions: list[str | None] | None = None,
    count: int = REPEAT_COUNT,
) -> tuple[str, subprocess.CompletedProcess]:
    assert test_framework in ["pytest", "unittest"]
    # TODO: Make this work for replay tests
    for i, test_path in enumerate(test_paths):
        if only_run_these_test_functions and "__replay_test" in test_path:
            test_paths[i] = test_path + "::" + only_run_these_test_functions

    if test_framework == "pytest":
        result_file_path = get_run_tmp_file("pytest_results.xml")
        pytest_cmd_list = shlex.split(pytest_cmd, posix=os.name != "nt")

        pytest_test_env = test_env.copy()
        pytest_test_env["PYTEST_PLUGINS"] = "codeflash.verification.pytest_plugin"
        pytest_test_env["CODEFLASH_LOOP_ID"] = "1"

        results = subprocess.run(
            pytest_cmd_list
            + test_paths
            + [
                "--capture=tee-sys",
                f"--timeout={pytest_timeout}",
                "-q",
                f"--junitxml={result_file_path}",
                "-o",
                "junit_logging=all",
                f"--seconds={1}",
                f"--min_loops={5}",
                f"--max_loops={100_000}",
                "--loops-scope=session",
            ],
            capture_output=True,
            cwd=cwd,
            env=pytest_test_env,
            text=True,
            timeout=600,  # TODO: Make this dynamic
            check=False,
        )
    elif test_framework == "unittest":
        result_file_path = get_run_tmp_file("unittest_results.xml")
        results = subprocess.run(
            ["python", "-m", "xmlrunner"]
            + (["-v"] if verbose else [])
            + test_paths
            + ["--output-file", result_file_path],
            capture_output=True,
            cwd=cwd,
            env=test_env,
            text=True,
            timeout=600,
            check=False,
        )
    else:
        raise ValueError(
            "Invalid test framework -- I only support Pytest and Unittest currently.",
        )
    return result_file_path, results
