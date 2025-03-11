import logging
import os
import pathlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoverageExpectation:
    function_name: str
    expected_coverage: float = 100.0
    expected_lines: list[int] = field(default_factory=list)  # Field with default list


@dataclass
class TestConfig:
    # Make file_path optional when trace_mode is True
    file_path: Optional[pathlib.Path] = None
    function_name: Optional[str] = None
    test_framework: Optional[str] = None
    expected_unit_tests: Optional[int] = None
    min_improvement_x: float = 0.1
    trace_mode: bool = False
    trace_load: str = "workload"
    coverage_expectations: list[CoverageExpectation] = field(default_factory=list)


def clear_directory(directory_path: str | pathlib.Path) -> None:
    """Empties all the files and subdirectories in the given directory to avoid errors in count of functions to be tested during retry."""
    dir_path = pathlib.Path(directory_path)
    if not dir_path.exists():
        print(f"The directory {directory_path} does not exist.")
        return

    for item in dir_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()  # Remove the file or symbolic link
            elif item.is_dir():
                shutil.rmtree(item)  # Remove the subdirectory
        except Exception as e:
            print(f"Failed to delete {item}. Reason: {e}")


def validate_coverage(stdout: str, expectations: list[CoverageExpectation]) -> bool:
    if not expectations:
        return True

    assert "CoverageData(" in stdout, "Failed to find CoverageData in stdout"

    for expect in expectations:
        pattern = rf"""
        (?:main|dependent)_func_coverage=FunctionCoverage\(
        \s+name='{expect.function_name}',
        \s+coverage=([\d.]+),
        \s+executed_lines=\[(.+?)\],
        """

        coverage_match = re.search(pattern, stdout, re.VERBOSE)
        assert coverage_match, f"Failed to find coverage data for {expect.function_name}"

        coverage = float(coverage_match.group(1))
        assert coverage == expect.expected_coverage, (
            f"Coverage was {coverage} instead of {expect.expected_coverage} for function: {expect.function_name}"
        )

        executed_lines = list(map(int, coverage_match.group(2).split(", ")))
        assert executed_lines == expect.expected_lines, (
            f"Executed lines were {executed_lines} instead of {expect.expected_lines} for function: {expect.function_name}"
        )

    return True


def run_codeflash_command(
    cwd: pathlib.Path, config: TestConfig, expected_improvement_pct: int, expected_in_stdout: list[str] = None
) -> bool:
    logging.basicConfig(level=logging.INFO)
    if config.trace_mode:
        return run_trace_test(cwd, config, expected_improvement_pct)

    path_to_file = cwd / config.file_path
    file_contents = path_to_file.read_text("utf-8")
    test_root = cwd / "tests" / (config.test_framework or "")
    command = build_command(cwd, config, test_root)

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd), env=os.environ.copy()
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)

    validated = validate_output(stdout, return_code, expected_improvement_pct, config)
    if not validated:
        # Write original file contents back to file
        path_to_file.write_text(file_contents, "utf-8")
        logging.info("Codeflash run did not meet expected requirements for testing, reverting file changes.")
        return False

    if expected_in_stdout:
        stdout_validated = validate_stdout_in_candidate(stdout, expected_in_stdout)
        if not stdout_validated:
            logging.error("Failed to find expected output in candidate output")
            validated = False
        logging.info(f"Success: Expected output found in candidate output")

    return validated


def build_command(cwd: pathlib.Path, config: TestConfig, test_root: pathlib.Path) -> list[str]:
    python_path = "../../../codeflash/main.py" if "code_directories" in str(cwd) else "../codeflash/main.py"

    base_command = ["python", python_path, "--file", config.file_path, "--no-pr"]

    if config.function_name:
        base_command.extend(["--function", config.function_name])
    if config.test_framework:
        base_command.extend(
            ["--test-framework", config.test_framework, "--tests-root", str(test_root), "--module-root", str(cwd)]
        )

    return base_command


def validate_output(stdout: str, return_code: int, expected_improvement_pct: int, config: TestConfig) -> bool:
    if return_code != 0:
        logging.error(f"Command returned exit code {return_code} instead of 0")
        return False

    if "⚡️ Optimization successful! 📄 " not in stdout:
        logging.error("Failed to find performance improvement message")
        return False

    improvement_match = re.search(r"📈 ([\d,]+)% improvement", stdout)
    if not improvement_match:
        logging.error("Could not find improvement percentage in output")
        return False

    improvement_pct = int(improvement_match.group(1).replace(",", ""))
    improvement_x = float(improvement_pct) / 100

    print("Performance improvement:", improvement_pct, "; Performance improvement rate:", improvement_x)
    if improvement_pct <= expected_improvement_pct:
        logging.error(f"Performance improvement {improvement_pct}% not above {expected_improvement_pct}%")
        return False

    if improvement_x <= config.min_improvement_x:
        logging.error(f"Performance improvement rate {improvement_x}x not above {config.min_improvement_x}x")
        return False

    if config.expected_unit_tests is not None:
        unit_test_match = re.search(r"Discovered (\d+) existing unit tests", stdout)
        if not unit_test_match:
            logging.error("Could not find unit test count")
            return False

        num_tests = int(unit_test_match.group(1))
        if num_tests != config.expected_unit_tests:
            logging.error(f"Expected {config.expected_unit_tests} unit tests, found {num_tests}")
            return False

    if config.coverage_expectations:
        validate_coverage(stdout, config.coverage_expectations)

    logging.info(f"Success: Performance improvement is {improvement_pct}%")
    return True


def validate_stdout_in_candidate(stdout: str, expected_in_stdout: list[str]) -> bool:
    candidate_output = stdout[stdout.find("INFO     Best candidate") : stdout.find("Best Candidate Explanation")]
    return all(expected in candidate_output for expected in expected_in_stdout)


def run_trace_test(cwd: pathlib.Path, config: TestConfig, expected_improvement_pct: int) -> bool:
    # First command: Run the tracer
    test_root = cwd / "tests" / (config.test_framework or "")
    clear_directory(test_root)

    trace_script = "workload.py" if config.trace_load == "workload" else "testbench.py"
    expected_traced_functions = 3 if config.trace_load == "workload" else 5

    command = ["python", "-m", "codeflash.tracer", "-o", "codeflash.trace", trace_script]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd), env=os.environ.copy()
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)

    if return_code != 0:
        logging.error(f"Tracer command returned exit code {return_code}")
        return False

    functions_traced = re.search(r"Traced (\d+) function calls successfully and replay test created at - (.*)$", stdout)
    if not functions_traced or int(functions_traced.group(1)) != expected_traced_functions:
        logging.error(f"Expected {expected_traced_functions} traced functions")
        return False

    replay_test_path = pathlib.Path(functions_traced.group(2))
    if not replay_test_path.exists():
        logging.error(f"Replay test file missing at {replay_test_path}")
        return False

    # Second command: Run optimization
    command = ["python", "../../../codeflash/main.py", "--replay-test", str(replay_test_path), "--no-pr"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd), env=os.environ.copy()
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)

    return validate_output(stdout, return_code, expected_improvement_pct, config)


def run_with_retries(test_func, *args, **kwargs) -> bool:
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    retry_delay = int(os.getenv("RETRY_DELAY", 5))
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    for attempt in range(1, max_retries + 1):
        logging.info(f"\n=== Attempt {attempt} of {max_retries} ===")

        if test_func(*args, **kwargs):
            logging.info(f"Test passed on attempt {attempt}")
            return 0

        logging.error(f"Test failed on attempt {attempt}")

        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logging.error("Test failed after all retries")
            return 1

    return 1
