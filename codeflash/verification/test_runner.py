import subprocess

from codeflash.code_utils.code_utils import get_run_tmp_file


def run_tests(
    test_path,
    test_framework: str,
    cwd: str = None,
    test_env=None,
    pytest_timeout: int = None,
    pytest_cmd: str = "pytest",
    verbose: bool = False,
) -> (str, subprocess.CompletedProcess):
    assert test_framework in ["pytest", "unittest"]
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
            env=test_env,
        )
    else:
        raise ValueError("Invalid test framework, we only support Pytest and Unittest currently.")
    return result_file_path, results
