"""Java test runner for JUnit 5 with Maven.

This module provides functionality to run JUnit 5 tests using Maven Surefire,
supporting both behavioral testing and benchmarking modes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import TestResult
from codeflash.languages.java.build_tools import (
    find_maven_executable,
    find_test_root,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class JavaTestRunResult:
    """Result of running Java tests."""

    success: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    test_results: list[TestResult]
    sqlite_db_path: Path | None
    junit_xml_path: Path | None
    stdout: str
    stderr: str
    returncode: int


def run_behavioral_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, Any, Path | None, Path | None]:
    """Run behavioral tests for Java code.

    This runs tests and captures behavior (inputs/outputs) for verification.

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        enable_coverage: Whether to collect coverage information.
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_file_path, subprocess_result, coverage_path, config_path).

    """
    project_root = project_root or cwd

    # Generate unique result file path
    result_id = uuid.uuid4().hex[:8]
    result_file = Path(tempfile.gettempdir()) / f"codeflash_java_behavior_{result_id}.db"

    # Set environment variables for CodeFlash runtime
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_RESULT_FILE"] = str(result_file)
    run_env["CODEFLASH_MODE"] = "behavior"

    # Run Maven tests
    result = _run_maven_tests(
        project_root,
        test_paths,
        run_env,
        timeout=timeout or 300,
    )

    return result_file, result, None, None


def run_benchmarking_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100_000,
    target_duration_seconds: float = 10.0,
) -> tuple[Path, Any]:
    """Run benchmarking tests for Java code.

    This runs tests with performance measurement.

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        min_loops: Minimum number of loops for benchmarking.
        max_loops: Maximum number of loops for benchmarking.
        target_duration_seconds: Target duration for benchmarking in seconds.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    project_root = project_root or cwd

    # Generate unique result file path
    result_id = uuid.uuid4().hex[:8]
    result_file = Path(tempfile.gettempdir()) / f"codeflash_java_benchmark_{result_id}.db"

    # Set environment variables
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_RESULT_FILE"] = str(result_file)
    run_env["CODEFLASH_MODE"] = "benchmark"
    run_env["CODEFLASH_MIN_LOOPS"] = str(min_loops)
    run_env["CODEFLASH_MAX_LOOPS"] = str(max_loops)
    run_env["CODEFLASH_TARGET_DURATION"] = str(target_duration_seconds)

    # Run Maven tests
    result = _run_maven_tests(
        project_root,
        test_paths,
        run_env,
        timeout=timeout or 600,  # Longer timeout for benchmarks
    )

    return result_file, result


def _run_maven_tests(
    project_root: Path,
    test_paths: Any,
    env: dict[str, str],
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run Maven tests with Surefire.

    Args:
        project_root: Root directory of the Maven project.
        test_paths: Test files or classes to run.
        env: Environment variables.
        timeout: Maximum execution time in seconds.

    Returns:
        CompletedProcess with test results.

    """
    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found")
        return subprocess.CompletedProcess(
            args=["mvn"],
            returncode=-1,
            stdout="",
            stderr="Maven not found",
        )

    # Build test filter
    test_filter = _build_test_filter(test_paths)

    # Build Maven command
    cmd = [mvn, "test", "-fae"]  # Fail at end to run all tests

    if test_filter:
        cmd.append(f"-Dtest={test_filter}")

    try:
        result = subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result

    except subprocess.TimeoutExpired:
        logger.error("Maven test execution timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-2,
            stdout="",
            stderr=f"Test execution timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception("Maven test execution failed: %s", e)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
        )


def _build_test_filter(test_paths: Any) -> str:
    """Build a Maven Surefire test filter from test paths.

    Args:
        test_paths: Test files, classes, or methods to include.

    Returns:
        Surefire test filter string.

    """
    if not test_paths:
        return ""

    # Handle different input types
    if isinstance(test_paths, (list, tuple)):
        filters = []
        for path in test_paths:
            if isinstance(path, Path):
                # Convert file path to class name
                class_name = _path_to_class_name(path)
                if class_name:
                    filters.append(class_name)
            elif isinstance(path, str):
                filters.append(path)
        return ",".join(filters) if filters else ""

    # Handle TestFiles object (has test_files attribute)
    if hasattr(test_paths, "test_files"):
        return _build_test_filter(list(test_paths.test_files))

    return ""


def _path_to_class_name(path: Path) -> str | None:
    """Convert a test file path to a Java class name.

    Args:
        path: Path to the test file.

    Returns:
        Fully qualified class name, or None if unable to determine.

    """
    if not path.suffix == ".java":
        return None

    # Try to extract package from path
    # e.g., src/test/java/com/example/CalculatorTest.java -> com.example.CalculatorTest
    parts = path.parts

    # Find 'java' in the path and take everything after
    try:
        java_idx = parts.index("java")
        class_parts = parts[java_idx + 1 :]
        # Remove .java extension from last part
        class_parts = list(class_parts)
        class_parts[-1] = class_parts[-1].replace(".java", "")
        return ".".join(class_parts)
    except ValueError:
        # No 'java' directory, just use the file name
        return path.stem


def run_tests(
    test_files: list[Path],
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[list[TestResult], Path]:
    """Run tests and return results.

    Args:
        test_files: Paths to test files to run.
        cwd: Working directory for test execution.
        env: Environment variables.
        timeout: Maximum execution time in seconds.

    Returns:
        Tuple of (list of TestResults, path to JUnit XML).

    """
    # Run Maven tests
    result = _run_maven_tests(cwd, test_files, env, timeout)

    # Parse JUnit XML results
    surefire_dir = cwd / "target" / "surefire-reports"
    test_results = parse_surefire_results(surefire_dir)

    # Return first XML file path
    junit_files = list(surefire_dir.glob("TEST-*.xml")) if surefire_dir.exists() else []
    junit_path = junit_files[0] if junit_files else cwd / "target" / "surefire-reports" / "test-results.xml"

    return test_results, junit_path


def parse_test_results(junit_xml_path: Path, stdout: str) -> list[TestResult]:
    """Parse test results from JUnit XML and stdout.

    Args:
        junit_xml_path: Path to JUnit XML results file.
        stdout: Standard output from test execution.

    Returns:
        List of TestResult objects.

    """
    return parse_surefire_results(junit_xml_path.parent)


def parse_surefire_results(surefire_dir: Path) -> list[TestResult]:
    """Parse Maven Surefire XML reports into TestResult objects.

    Args:
        surefire_dir: Directory containing Surefire XML reports.

    Returns:
        List of TestResult objects.

    """
    results: list[TestResult] = []

    if not surefire_dir.exists():
        return results

    for xml_file in surefire_dir.glob("TEST-*.xml"):
        results.extend(_parse_surefire_xml(xml_file))

    return results


def _parse_surefire_xml(xml_file: Path) -> list[TestResult]:
    """Parse a single Surefire XML file.

    Args:
        xml_file: Path to the XML file.

    Returns:
        List of TestResult objects for tests in this file.

    """
    results: list[TestResult] = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get test class info
        class_name = root.get("name", "")

        # Process each test case
        for testcase in root.findall(".//testcase"):
            test_name = testcase.get("name", "")
            test_time = float(testcase.get("time", "0"))
            runtime_ns = int(test_time * 1_000_000_000)

            # Check for failure/error
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            passed = failure is None and error is None and skipped is None
            error_message = None

            if failure is not None:
                error_message = failure.get("message", "")
                if failure.text:
                    error_message += "\n" + failure.text

            if error is not None:
                error_message = error.get("message", "")
                if error.text:
                    error_message += "\n" + error.text

            # Get stdout/stderr from system-out/system-err elements
            stdout = ""
            stderr = ""
            stdout_elem = testcase.find("system-out")
            if stdout_elem is not None and stdout_elem.text:
                stdout = stdout_elem.text
            stderr_elem = testcase.find("system-err")
            if stderr_elem is not None and stderr_elem.text:
                stderr = stderr_elem.text

            results.append(
                TestResult(
                    test_name=test_name,
                    test_file=xml_file,
                    passed=passed,
                    runtime_ns=runtime_ns,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=error_message,
                )
            )

    except ET.ParseError as e:
        logger.warning("Failed to parse Surefire report %s: %s", xml_file, e)

    return results


def get_test_run_command(
    project_root: Path,
    test_classes: list[str] | None = None,
) -> list[str]:
    """Get the command to run Java tests.

    Args:
        project_root: Root directory of the Maven project.
        test_classes: Optional list of test class names to run.

    Returns:
        Command as list of strings.

    """
    mvn = find_maven_executable() or "mvn"

    cmd = [mvn, "test"]

    if test_classes:
        cmd.append(f"-Dtest={','.join(test_classes)}")

    return cmd
