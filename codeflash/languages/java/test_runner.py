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
from typing import Any

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.languages.base import TestResult
from codeflash.languages.java.build_tools import (
    add_jacoco_plugin_to_pom,
    find_maven_executable,
    get_jacoco_xml_path,
    is_jacoco_configured,
)

logger = logging.getLogger(__name__)


def _find_multi_module_root(project_root: Path, test_paths: Any) -> tuple[Path, str | None]:
    """Find the multi-module Maven parent root if tests are in a different module.

    For multi-module Maven projects, tests may be in a separate module from the source code.
    This function detects this situation and returns the parent project root along with
    the module containing the tests.

    Args:
        project_root: The current project root (typically the source module).
        test_paths: TestFiles object or list of test file paths.

    Returns:
        Tuple of (maven_root, test_module_name) where:
        - maven_root: The directory to run Maven from (parent if multi-module, else project_root)
        - test_module_name: The name of the test module if different from project_root, else None

    """
    # Get test file paths - try both benchmarking and behavior paths
    test_file_paths: list[Path] = []
    if hasattr(test_paths, "test_files"):
        for test_file in test_paths.test_files:
            # Prefer benchmarking_file_path for performance mode
            if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                test_file_paths.append(test_file.benchmarking_file_path)
            elif hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
                test_file_paths.append(test_file.instrumented_behavior_file_path)
    elif isinstance(test_paths, (list, tuple)):
        test_file_paths = [Path(p) if isinstance(p, str) else p for p in test_paths]

    if not test_file_paths:
        return project_root, None

    # Check if any test file is outside the project_root
    test_outside_project = False
    test_dir: Path | None = None
    for test_path in test_file_paths:
        try:
            test_path.relative_to(project_root)
        except ValueError:
            # Test is outside project_root
            test_outside_project = True
            test_dir = test_path.parent
            break

    if not test_outside_project:
        # Check if project_root itself is a multi-module project
        # and the test file is in a submodule (e.g., test/src/...)
        pom_path = project_root / "pom.xml"
        if pom_path.exists():
            try:
                content = pom_path.read_text(encoding="utf-8")
                if "<modules>" in content:
                    # This is a multi-module project root
                    # Extract modules from pom.xml
                    import re
                    modules = re.findall(r"<module>([^<]+)</module>", content)
                    # Check if test file is in one of the modules
                    for test_path in test_file_paths:
                        try:
                            rel_path = test_path.relative_to(project_root)
                            # Get the first component of the relative path
                            first_component = rel_path.parts[0] if rel_path.parts else None
                            if first_component and first_component in modules:
                                logger.debug(
                                    "Detected multi-module Maven project. Root: %s, Test module: %s",
                                    project_root,
                                    first_component,
                                )
                                return project_root, first_component
                        except ValueError:
                            pass
            except Exception:
                pass
        return project_root, None

    # Find common parent that contains both project_root and test files
    # and has a pom.xml with <modules> section
    current = project_root.parent
    while current != current.parent:
        pom_path = current / "pom.xml"
        if pom_path.exists():
            # Check if this is a multi-module pom
            try:
                content = pom_path.read_text(encoding="utf-8")
                if "<modules>" in content:
                    # Found multi-module parent
                    # Get the relative module name for the test directory
                    if test_dir:
                        try:
                            test_module = test_dir.relative_to(current)
                            # Get the top-level module name (first component)
                            test_module_name = test_module.parts[0] if test_module.parts else None
                            logger.debug(
                                "Detected multi-module Maven project. Root: %s, Test module: %s",
                                current,
                                test_module_name,
                            )
                            return current, test_module_name
                        except ValueError:
                            pass
            except Exception:
                pass
        current = current.parent

    return project_root, None


def _get_test_module_target_dir(maven_root: Path, test_module: str | None) -> Path:
    """Get the target directory for the test module.

    Args:
        maven_root: The Maven project root.
        test_module: The test module name, or None if not a multi-module project.

    Returns:
        Path to the target directory where surefire reports will be.

    """
    if test_module:
        return maven_root / test_module / "target"
    return maven_root / "target"


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
        Tuple of (result_xml_path, subprocess_result, sqlite_db_path, coverage_xml_path).

    """
    project_root = project_root or cwd

    # Detect multi-module Maven projects where tests are in a different module
    maven_root, test_module = _find_multi_module_root(project_root, test_paths)

    # Create SQLite database path for behavior capture - use standard path that parse_test_results expects
    sqlite_db_path = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))

    # Set environment variables for timing instrumentation and behavior capture
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_LOOP_INDEX"] = "1"  # Single loop for behavior tests
    run_env["CODEFLASH_MODE"] = "behavior"
    run_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    run_env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_db_path)  # SQLite output path

    # If coverage is enabled, ensure JaCoCo is configured
    # For multi-module projects, add JaCoCo to the test module's pom.xml (where tests run)
    coverage_xml_path: Path | None = None
    if enable_coverage:
        # Determine which pom.xml to configure JaCoCo in
        if test_module:
            # Multi-module project: add JaCoCo to test module
            test_module_pom = maven_root / test_module / "pom.xml"
            if test_module_pom.exists():
                if not is_jacoco_configured(test_module_pom):
                    logger.info(f"Adding JaCoCo plugin to test module pom.xml: {test_module_pom}")
                    add_jacoco_plugin_to_pom(test_module_pom)
                coverage_xml_path = get_jacoco_xml_path(maven_root / test_module)
        else:
            # Single module project
            pom_path = project_root / "pom.xml"
            if pom_path.exists():
                if not is_jacoco_configured(pom_path):
                    logger.info("Adding JaCoCo plugin to pom.xml for coverage collection")
                    add_jacoco_plugin_to_pom(pom_path)
                coverage_xml_path = get_jacoco_xml_path(project_root)

    # Run Maven tests from the appropriate root
    # Use a minimum timeout of 60s for Java builds (120s when coverage is enabled due to verify phase)
    min_timeout = 120 if enable_coverage else 60
    effective_timeout = max(timeout or 300, min_timeout)
    result = _run_maven_tests(
        maven_root,
        test_paths,
        run_env,
        timeout=effective_timeout,
        mode="behavior",
        enable_coverage=enable_coverage,
        test_module=test_module,
    )

    # Find or create the JUnit XML results file
    # For multi-module projects, look in the test module's target directory
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, candidate_index)

    # Return coverage_xml_path as the fourth element when coverage is enabled
    return result_xml_path, result, sqlite_db_path, coverage_xml_path


def _compile_tests(
    project_root: Path,
    env: dict[str, str],
    test_module: str | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Compile test code using Maven (without running tests).

    Args:
        project_root: Root directory of the Maven project.
        env: Environment variables.
        test_module: For multi-module projects, the module containing tests.
        timeout: Maximum execution time in seconds.

    Returns:
        CompletedProcess with compilation results.

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

    cmd = [mvn, "test-compile", "-q"]  # Quiet mode for faster output

    if test_module:
        cmd.extend(["-pl", test_module, "-am"])

    logger.debug("Compiling tests: %s in %s", " ".join(cmd), project_root)

    try:
        return subprocess.run(
            cmd,
            check=False,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("Maven compilation timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-2,
            stdout="",
            stderr=f"Compilation timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception("Maven compilation failed: %s", e)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
        )


def _get_test_classpath(
    project_root: Path,
    env: dict[str, str],
    test_module: str | None = None,
    timeout: int = 60,
) -> str | None:
    """Get the test classpath from Maven.

    Args:
        project_root: Root directory of the Maven project.
        env: Environment variables.
        test_module: For multi-module projects, the module containing tests.
        timeout: Maximum execution time in seconds.

    Returns:
        Classpath string, or None if failed.

    """
    mvn = find_maven_executable()
    if not mvn:
        return None

    # Create temp file for classpath output
    cp_file = project_root / ".codeflash_classpath.txt"

    cmd = [
        mvn,
        "dependency:build-classpath",
        "-DincludeScope=test",
        f"-Dmdep.outputFile={cp_file}",
        "-q",
    ]

    if test_module:
        cmd.extend(["-pl", test_module])

    logger.debug("Getting classpath: %s", " ".join(cmd))

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

        if result.returncode != 0:
            logger.error("Failed to get classpath: %s", result.stderr)
            return None

        if not cp_file.exists():
            logger.error("Classpath file not created")
            return None

        classpath = cp_file.read_text(encoding="utf-8").strip()

        # Add compiled classes directories to classpath
        # For multi-module, we need to find the correct target directories
        if test_module:
            module_path = project_root / test_module
        else:
            module_path = project_root

        test_classes = module_path / "target" / "test-classes"
        main_classes = module_path / "target" / "classes"

        cp_parts = [classpath]
        if test_classes.exists():
            cp_parts.append(str(test_classes))
        if main_classes.exists():
            cp_parts.append(str(main_classes))

        return os.pathsep.join(cp_parts)

    except subprocess.TimeoutExpired:
        logger.error("Getting classpath timed out")
        return None
    except Exception as e:
        logger.exception("Failed to get classpath: %s", e)
        return None
    finally:
        # Clean up temp file
        if cp_file.exists():
            cp_file.unlink()


def _run_tests_direct(
    classpath: str,
    test_classes: list[str],
    env: dict[str, str],
    working_dir: Path,
    timeout: int = 60,
    reports_dir: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run JUnit tests directly using java command (bypassing Maven).

    This is much faster than Maven invocation (~500ms vs ~5-10s overhead).

    Args:
        classpath: Full classpath including test dependencies.
        test_classes: List of fully qualified test class names to run.
        env: Environment variables.
        working_dir: Working directory for execution.
        timeout: Maximum execution time in seconds.
        reports_dir: Optional directory for JUnit XML reports.

    Returns:
        CompletedProcess with test results.

    """
    # Find java executable
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java = Path(java_home) / "bin" / "java"
        if not java.exists():
            java = "java"
    else:
        java = "java"

    # Build command using JUnit Platform Console Launcher
    # The launcher is included in junit-platform-console-standalone or junit-jupiter
    cmd = [
        str(java),
        "-cp",
        classpath,
        "org.junit.platform.console.ConsoleLauncher",
        "--disable-banner",
        "--disable-ansi-colors",
        # Use 'none' details to avoid duplicate output
        # Timing markers are captured in XML via stdout capture config
        "--details=none",
        # Enable stdout/stderr capture in XML reports
        # This ensures timing markers are included in the XML system-out element
        "--config=junit.platform.output.capture.stdout=true",
        "--config=junit.platform.output.capture.stderr=true",
    ]

    # Add reports directory if specified (for XML output)
    if reports_dir:
        reports_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--reports-dir", str(reports_dir)])

    # Add test classes to select
    for test_class in test_classes:
        cmd.extend(["--select-class", test_class])

    logger.debug("Running tests directly: java -cp ... ConsoleLauncher --select-class %s", test_classes)

    try:
        return subprocess.run(
            cmd,
            check=False,
            cwd=working_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error("Direct test execution timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-2,
            stdout="",
            stderr=f"Test execution timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception("Direct test execution failed: %s", e)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
        )


def _get_test_class_names(test_paths: Any, mode: str = "performance") -> list[str]:
    """Extract fully qualified test class names from test paths.

    Args:
        test_paths: TestFiles object or list of test file paths.
        mode: Testing mode - "behavior" or "performance".

    Returns:
        List of fully qualified class names.

    """
    class_names = []

    if hasattr(test_paths, "test_files"):
        for test_file in test_paths.test_files:
            if mode == "performance":
                if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                    class_name = _path_to_class_name(test_file.benchmarking_file_path)
                    if class_name:
                        class_names.append(class_name)
            elif hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
                class_name = _path_to_class_name(test_file.instrumented_behavior_file_path)
                if class_name:
                    class_names.append(class_name)
    elif isinstance(test_paths, (list, tuple)):
        for path in test_paths:
            if isinstance(path, Path):
                class_name = _path_to_class_name(path)
                if class_name:
                    class_names.append(class_name)
            elif isinstance(path, str):
                class_names.append(path)

    return class_names


def _get_empty_result(maven_root: Path, test_module: str | None) -> tuple[Path, Any]:
    """Return an empty result for when no tests can be run.

    Args:
        maven_root: Maven project root.
        test_module: Optional test module name.

    Returns:
        Tuple of (empty_xml_path, empty_result).

    """
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, -1)

    empty_result = subprocess.CompletedProcess(
        args=["java", "-cp", "...", "ConsoleLauncher"],
        returncode=-1,
        stdout="",
        stderr="No test classes found",
    )
    return result_xml_path, empty_result


def _run_benchmarking_tests_maven(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None,
    project_root: Path | None,
    min_loops: int,
    max_loops: int,
    target_duration_seconds: float,
    inner_iterations: int,
) -> tuple[Path, Any]:
    """Fallback: Run benchmarking tests using Maven (slower but more reliable).

    This is used when direct JVM execution fails (e.g., classpath issues).

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        min_loops: Minimum number of outer loops.
        max_loops: Maximum number of outer loops.
        target_duration_seconds: Target duration for benchmarking.
        inner_iterations: Number of inner loop iterations.

    Returns:
        Tuple of (result_file_path, subprocess_result with aggregated stdout).

    """
    import time

    project_root = project_root or cwd
    maven_root, test_module = _find_multi_module_root(project_root, test_paths)

    all_stdout = []
    all_stderr = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    per_loop_timeout = timeout or max(120, 60 + inner_iterations)

    logger.debug("Using Maven-based benchmarking (fallback mode)")

    for loop_idx in range(1, max_loops + 1):
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

        result = _run_maven_tests(
            maven_root,
            test_paths,
            run_env,
            timeout=per_loop_timeout,
            mode="performance",
            test_module=test_module,
        )

        last_result = result
        loop_count = loop_idx

        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug(
                "Stopping Maven benchmark after %d loops (%.2fs elapsed)",
                loop_idx,
                elapsed,
            )
            break

        if result.returncode != 0:
            logger.warning("Tests failed in Maven loop %d, stopping", loop_idx)
            break

    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)

    total_iterations = loop_count * inner_iterations
    logger.debug(
        "Maven fallback: %d loops x %d iterations = %d total in %.2fs",
        loop_count,
        inner_iterations,
        total_iterations,
        time.time() - total_start_time,
    )

    combined_result = subprocess.CompletedProcess(
        args=last_result.args if last_result else ["mvn", "test"],
        returncode=last_result.returncode if last_result else -1,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )

    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, -1)

    return result_xml_path, combined_result


def run_benchmarking_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 1,
    max_loops: int = 3,
    target_duration_seconds: float = 10.0,
    inner_iterations: int = 100,
) -> tuple[Path, Any]:
    """Run benchmarking tests for Java code with compile-once-run-many optimization.

    This compiles tests once, then runs them multiple times directly via JVM,
    bypassing Maven overhead (~500ms vs ~5-10s per invocation).

    The instrumented tests run CODEFLASH_INNER_ITERATIONS iterations per JVM invocation,
    printing timing markers that are parsed from stdout:
      Start: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
      End:   !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!

    Where iterationId is the inner iteration number (0, 1, 2, ..., inner_iterations-1).

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        min_loops: Minimum number of outer loops (JVM invocations). Default: 1.
        max_loops: Maximum number of outer loops (JVM invocations). Default: 3.
        target_duration_seconds: Target duration for benchmarking in seconds.
        inner_iterations: Number of inner loop iterations per JVM invocation. Default: 100.

    Returns:
        Tuple of (result_file_path, subprocess_result with aggregated stdout).

    """
    import time

    project_root = project_root or cwd

    # Detect multi-module Maven projects where tests are in a different module
    maven_root, test_module = _find_multi_module_root(project_root, test_paths)

    # Get test class names
    test_classes = _get_test_class_names(test_paths, mode="performance")
    if not test_classes:
        logger.error("No test classes found")
        return _get_empty_result(maven_root, test_module)

    # Step 1: Compile tests once using Maven
    compile_env = os.environ.copy()
    compile_env.update(test_env)

    logger.debug("Step 1: Compiling tests (one-time Maven overhead)")
    compile_start = time.time()
    compile_result = _compile_tests(maven_root, compile_env, test_module, timeout=120)
    compile_time = time.time() - compile_start

    if compile_result.returncode != 0:
        logger.error("Test compilation failed: %s", compile_result.stderr)
        # Fall back to Maven-based execution
        logger.warning("Falling back to Maven-based test execution")
        return _run_benchmarking_tests_maven(
            test_paths, test_env, cwd, timeout, project_root,
            min_loops, max_loops, target_duration_seconds, inner_iterations
        )

    logger.debug("Compilation completed in %.2fs", compile_time)

    # Step 2: Get classpath from Maven
    logger.debug("Step 2: Getting classpath")
    classpath = _get_test_classpath(maven_root, compile_env, test_module, timeout=60)

    if not classpath:
        logger.warning("Failed to get classpath, falling back to Maven-based execution")
        return _run_benchmarking_tests_maven(
            test_paths, test_env, cwd, timeout, project_root,
            min_loops, max_loops, target_duration_seconds, inner_iterations
        )

    # Step 3: Run tests multiple times directly via JVM
    logger.debug("Step 3: Running tests directly (bypassing Maven)")

    all_stdout = []
    all_stderr = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    # Calculate timeout per loop
    per_loop_timeout = timeout or max(60, 30 + inner_iterations // 10)

    # Determine working directory for test execution
    if test_module:
        working_dir = maven_root / test_module
    else:
        working_dir = maven_root

    # Create reports directory for JUnit XML output (in Surefire-compatible location)
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    reports_dir = target_dir / "surefire-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for loop_idx in range(1, max_loops + 1):
        # Set environment variables for this loop
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

        # Run tests directly with XML report generation
        loop_start = time.time()
        result = _run_tests_direct(
            classpath,
            test_classes,
            run_env,
            working_dir,
            timeout=per_loop_timeout,
            reports_dir=reports_dir,
        )
        loop_time = time.time() - loop_start

        last_result = result
        loop_count = loop_idx

        # Collect stdout/stderr
        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        logger.debug("Loop %d completed in %.2fs (returncode=%d)", loop_idx, loop_time, result.returncode)

        # Check if JUnit Console Launcher is not available (JUnit 4 projects)
        # Fall back to Maven-based execution in this case
        if (
            loop_idx == 1
            and result.returncode != 0
            and result.stderr
            and "ConsoleLauncher" in result.stderr
        ):
            logger.debug("JUnit Console Launcher not available, falling back to Maven-based execution")
            return _run_benchmarking_tests_maven(
                test_paths,
                test_env,
                cwd,
                timeout,
                project_root,
                min_loops,
                max_loops,
                target_duration_seconds,
                inner_iterations,
            )

        # Check if we've hit the target duration
        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug(
                "Stopping benchmark after %d loops (%.2fs elapsed, target: %.2fs, %d inner iterations each)",
                loop_idx,
                elapsed,
                target_duration_seconds,
                inner_iterations,
            )
            break

        # Check if tests failed - don't continue looping
        if result.returncode != 0:
            logger.warning("Tests failed in loop %d, stopping benchmark", loop_idx)
            break

    # Create a combined result with all stdout
    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)

    total_time = time.time() - total_start_time
    total_iterations = loop_count * inner_iterations
    logger.debug(
        "Completed %d loops x %d inner iterations = %d total iterations in %.2fs (compile: %.2fs)",
        loop_count,
        inner_iterations,
        total_iterations,
        total_time,
        compile_time,
    )

    # Create a combined subprocess result
    combined_result = subprocess.CompletedProcess(
        args=last_result.args if last_result else ["mvn", "test"],
        returncode=last_result.returncode if last_result else -1,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )

    # Find or create the JUnit XML results file (from last run)
    # For multi-module projects, look in the test module's target directory
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
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
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="NoTests" tests="0" failures="0" errors="0" skipped="0" time="0">
</testsuite>
"""
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
    enable_coverage: bool = False,
    test_module: str | None = None,
) -> subprocess.CompletedProcess:
    """Run Maven tests with Surefire.

    Args:
        project_root: Root directory of the Maven project.
        test_paths: Test files or classes to run.
        env: Environment variables.
        timeout: Maximum execution time in seconds.
        mode: Testing mode - "behavior" or "performance".
        enable_coverage: Whether to enable JaCoCo coverage collection.
        test_module: For multi-module projects, the module containing tests.

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
    # When coverage is enabled, use 'verify' phase to ensure JaCoCo report runs after tests
    # JaCoCo's report goal is bound to the verify phase to get post-test execution data
    maven_goal = "verify" if enable_coverage else "test"
    cmd = [mvn, maven_goal, "-fae"]  # Fail at end to run all tests

    # When coverage is enabled, continue build even if tests fail so JaCoCo report is generated
    if enable_coverage:
        cmd.append("-Dmaven.test.failure.ignore=true")

    # For multi-module projects, specify which module to test
    if test_module:
        # -am = also make dependencies
        # -DfailIfNoTests=false allows dependency modules without tests to pass
        # -DskipTests=false overrides any skipTests=true in pom.xml
        cmd.extend(["-pl", test_module, "-am", "-DfailIfNoTests=false", "-DskipTests=false"])

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
            # For behavior mode, use instrumented_behavior_file_path
            elif hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
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
