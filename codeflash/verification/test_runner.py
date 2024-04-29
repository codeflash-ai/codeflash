import subprocess
from typing import Optional, Tuple

from codeflash.code_utils.code_utils import get_run_tmp_file


def run_tests(
    test_path,
    test_framework: str,
    cwd: Optional[str] = None,
    test_env: Optional[dict[str, str]] = None,
    pytest_timeout: Optional[int] = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
    test_function: Optional[str] = None,
) -> Tuple[str, subprocess.CompletedProcess]:
    assert test_framework in ["pytest", "unittest"]
    if test_function and "__replay_test" in test_path:
        test_path = test_path + "::" + test_function

    if test_framework == "pytest":
        result_file_path = get_run_tmp_file("pytest_results.xml")
        pytest_cmd_list = [chunk for chunk in pytest_cmd.split(" ") if chunk != ""]
        results = subprocess.run(
            pytest_cmd_list
            + [
                test_path,
                "-q",
                f"--timeout={pytest_timeout}",
                f"--junitxml={result_file_path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=test_env,
            check=False,
            timeout=600,
        )
    elif test_framework == "unittest":
        result_file_path = get_run_tmp_file("unittest_results.xml")
        results = subprocess.run(
            ["python", "-m", "xmlrunner"]
            + (["-v"] if verbose else [])
            + [test_path]
            + ["--output-file", result_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=test_env, check=False,
        )
    else:
        raise ValueError("Invalid test framework -- I only support Pytest and Unittest currently.")
    return result_file_path, results
