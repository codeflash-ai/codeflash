import logging
import os
import pathlib
import re
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
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    test_framework: Optional[str] = None
    expected_unit_tests: Optional[int] = None
    min_improvement_x: float = 0.1
    trace_mode: bool = False
    coverage_expectations: list[CoverageExpectation] = field(default_factory=list)


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
        assert coverage == expect.expected_coverage, f"Coverage was {coverage} instead of {expect.expected_coverage}"

        executed_lines = list(map(int, coverage_match.group(2).split(", ")))
        assert (
            executed_lines == expect.expected_lines
        ), f"Executed lines were {executed_lines} instead of {expect.expected_lines}"

    return True


def run_codeflash_command(cwd: pathlib.Path, config: TestConfig, expected_improvement_pct: int) -> bool:
    logging.basicConfig(level=logging.INFO)
    logging.info("If you see this, logging is working!")
    # print("Debug")
    if config.trace_mode:
        return run_trace_test(cwd, config, expected_improvement_pct)

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

    return validate_output(stdout, return_code, expected_improvement_pct, config)


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


def run_trace_test(cwd: pathlib.Path, config: TestConfig, expected_improvement_pct: int) -> bool:
    # First command: Run the tracer
    command = ["python", "-m", "codeflash.tracer", "-o", "codeflash.trace", "workload.py"]
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
    if not functions_traced or int(functions_traced.group(1)) != 3:
        logging.error("Expected 3 traced functions")
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
