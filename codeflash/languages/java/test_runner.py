"""Java test runner for JUnit 5 with Maven.

This module provides functionality to run JUnit 5 tests using Maven Surefire,
supporting both behavioral testing and benchmarking modes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.code_utils.code_utils import get_run_tmp_file
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
    For Java, test results are written to a SQLite database via CodeflashHelper,
    and JUnit test pass/fail results serve as the primary verification mechanism.

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        enable_coverage: Whether to collect coverage information.
        candidate_index: Index of the candidate being tested.

    Returns:
        Tuple of (result_xml_path, subprocess_result, sqlite_db_path, None).

    """
    project_root = project_root or cwd

    # Create SQLite database path for behavior capture - use standard path that parse_test_results expects
    sqlite_db_path = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))

    # Set environment variables for timing instrumentation and behavior capture
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_LOOP_INDEX"] = "1"  # Single loop for behavior tests
    run_env["CODEFLASH_MODE"] = "behavior"
    run_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    run_env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_db_path)  # SQLite output path

    # Run Maven tests
    result = _run_maven_tests(
        project_root,
        test_paths,
        run_env,
        timeout=timeout or 300,
        mode="behavior",
    )

    # Find or create the JUnit XML results file
    surefire_dir = project_root / "target" / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, candidate_index)

    # Return sqlite_db_path as the third element (was None before)
    return result_xml_path, result, sqlite_db_path, None


def run_benchmarking_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100,
    target_duration_seconds: float = 10.0,
) -> tuple[Path, Any]:
    """Run benchmarking tests for Java code.

    This runs tests multiple times with performance measurement.
    The instrumented tests print timing markers that are parsed from stdout:
      Start: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
      End:   !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!

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
        Tuple of (result_file_path, subprocess_result with aggregated stdout).

    """
    import time

    project_root = project_root or cwd

    # Collect stdout from all loops
    all_stdout = []
    all_stderr = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    # Run multiple loops until we hit target duration or max loops
    for loop_idx in range(1, max_loops + 1):
        # Set environment variables for this loop
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"

        # Run Maven tests for this loop
        result = _run_maven_tests(
            project_root,
            test_paths,
            run_env,
            timeout=timeout or 120,  # Per-loop timeout
            mode="performance",
        )

        last_result = result
        loop_count = loop_idx

        # Collect stdout/stderr
        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        # Check if we've hit the target duration
        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug(
                "Stopping benchmark after %d loops (%.2fs elapsed, target: %.2fs)",
                loop_idx,
                elapsed,
                target_duration_seconds,
            )
            break

        # Check if tests failed - don't continue looping
        if result.returncode != 0:
            logger.warning("Tests failed in loop %d, stopping benchmark", loop_idx)
            break

    # Create a combined result with all stdout
    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)

    logger.debug(
        "Completed %d benchmark loops in %.2fs",
        loop_count,
        time.time() - total_start_time,
    )

    # Create a combined subprocess result
    combined_result = subprocess.CompletedProcess(
        args=last_result.args if last_result else ["mvn", "test"],
        returncode=last_result.returncode if last_result else -1,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )

    # Find or create the JUnit XML results file (from last run)
    surefire_dir = project_root / "target" / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, -1)  # Use -1 for benchmark

    return result_xml_path, combined_result


def _get_combined_junit_xml(surefire_dir: Path, candidate_index: int) -> Path:
    """Get or create a combined JUnit XML file from Surefire reports.

    Args:
        surefire_dir: Directory containing Surefire reports.
        candidate_index: Index for unique naming.

    Returns:
        Path to the combined JUnit XML file.

    """
    # Create a temp file for the combined results
    result_id = uuid.uuid4().hex[:8]
    result_xml_path = Path(tempfile.gettempdir()) / f"codeflash_java_results_{candidate_index}_{result_id}.xml"

    if not surefire_dir.exists():
        # Create an empty results file
        _write_empty_junit_xml(result_xml_path)
        return result_xml_path

    # Find all TEST-*.xml files
    xml_files = list(surefire_dir.glob("TEST-*.xml"))

    if not xml_files:
        _write_empty_junit_xml(result_xml_path)
        return result_xml_path

    if len(xml_files) == 1:
        # Copy the single file
        shutil.copy(xml_files[0], result_xml_path)
        return result_xml_path

    # Combine multiple XML files into one
    _combine_junit_xml_files(xml_files, result_xml_path)
    return result_xml_path


def _write_empty_junit_xml(path: Path) -> None:
    """Write an empty JUnit XML results file."""
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="NoTests" tests="0" failures="0" errors="0" skipped="0" time="0">
</testsuite>
'''
    path.write_text(xml_content, encoding="utf-8")


def _combine_junit_xml_files(xml_files: list[Path], output_path: Path) -> None:
    """Combine multiple JUnit XML files into one.

    Args:
        xml_files: List of XML files to combine.
        output_path: Path for the combined output.

    """
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0
    all_testcases = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get testsuite attributes
            total_tests += int(root.get("tests", 0))
            total_failures += int(root.get("failures", 0))
            total_errors += int(root.get("errors", 0))
            total_skipped += int(root.get("skipped", 0))
            total_time += float(root.get("time", 0))

            # Collect all testcases
            for testcase in root.findall(".//testcase"):
                all_testcases.append(testcase)

        except Exception as e:
            logger.warning("Failed to parse %s: %s", xml_file, e)

    # Create combined XML
    combined_root = ET.Element("testsuite")
    combined_root.set("name", "CombinedTests")
    combined_root.set("tests", str(total_tests))
    combined_root.set("failures", str(total_failures))
    combined_root.set("errors", str(total_errors))
    combined_root.set("skipped", str(total_skipped))
    combined_root.set("time", str(total_time))

    for testcase in all_testcases:
        combined_root.append(testcase)

    tree = ET.ElementTree(combined_root)
    tree.write(output_path, encoding="unicode", xml_declaration=True)


def _run_maven_tests(
    project_root: Path,
    test_paths: Any,
    env: dict[str, str],
    timeout: int = 300,
    mode: str = "behavior",
) -> subprocess.CompletedProcess:
    """Run Maven tests with Surefire.

    Args:
        project_root: Root directory of the Maven project.
        test_paths: Test files or classes to run.
        env: Environment variables.
        timeout: Maximum execution time in seconds.
        mode: Testing mode - "behavior" or "performance".

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
    test_filter = _build_test_filter(test_paths, mode=mode)

    # Build Maven command
    cmd = [mvn, "test", "-fae"]  # Fail at end to run all tests

    if test_filter:
        cmd.append(f"-Dtest={test_filter}")

    logger.debug("Running Maven command: %s in %s", " ".join(cmd), project_root)

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


def _build_test_filter(test_paths: Any, mode: str = "behavior") -> str:
    """Build a Maven Surefire test filter from test paths.

    Args:
        test_paths: Test files, classes, or methods to include.
        mode: Testing mode - "behavior" or "performance".

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
        filters = []
        for test_file in test_paths.test_files:
            # For performance mode, use benchmarking_file_path
            if mode == "performance":
                if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                    class_name = _path_to_class_name(test_file.benchmarking_file_path)
                    if class_name:
                        filters.append(class_name)
            else:
                # For behavior mode, use instrumented_behavior_file_path
                if hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
                    class_name = _path_to_class_name(test_file.instrumented_behavior_file_path)
                    if class_name:
                        filters.append(class_name)
        return ",".join(filters) if filters else ""

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
    parts = list(path.parts)

    # Look for standard Maven/Gradle source directories
    # Find 'java' that comes after 'main' or 'test'
    java_idx = None
    for i, part in enumerate(parts):
        if part == "java" and i > 0 and parts[i - 1] in ("main", "test"):
            java_idx = i
            break

    # If no standard Maven structure, find the last 'java' in path
    if java_idx is None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "java":
                java_idx = i
                break

    if java_idx is not None:
        class_parts = parts[java_idx + 1:]
        # Remove .java extension from last part
        class_parts[-1] = class_parts[-1].replace(".java", "")
        return ".".join(class_parts)

    # Fallback: just use the file name
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
