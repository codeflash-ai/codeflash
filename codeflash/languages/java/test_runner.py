"""Java test runner for JUnit 5 with Maven.

This module provides functionality to run JUnit 5 tests using Maven Surefire,
supporting both behavioral testing and benchmarking modes.
"""

from __future__ import annotations

import logging
import os
import re
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


def _extract_modules_from_pom_content(content: str) -> list[str]:
    """Extract module names from Maven POM XML content using proper XML parsing.

    Handles both namespaced and non-namespaced POMs.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        logger.debug("Failed to parse POM XML for module extraction")
        return []

    ns = {"m": "http://maven.apache.org/POM/4.0.0"}

    modules_elem = root.find("m:modules", ns)
    if modules_elem is None:
        modules_elem = root.find("modules")

    if modules_elem is None:
        return []

    return [m.text for m in modules_elem if m.text]


# Regex pattern for valid Java class names (package.ClassName format)
_VALID_JAVA_CLASS_NAME = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$.]*$")


def _validate_java_class_name(class_name: str) -> bool:
    """Validate that a string is a valid Java class name."""
    return bool(_VALID_JAVA_CLASS_NAME.match(class_name))


def _validate_test_filter(test_filter: str) -> str:
    """Validate and sanitize a test filter string for Maven."""
    patterns = [p.strip() for p in test_filter.split(",")]

    for pattern in patterns:
        name_to_validate = pattern.replace("*", "A")

        if not _validate_java_class_name(name_to_validate):
            msg = (
                f"Invalid test class name or pattern: '{pattern}'. "
                f"Test names must follow Java identifier rules."
            )
            raise ValueError(msg)

    return test_filter


def _find_multi_module_root(project_root: Path, test_paths: Any) -> tuple[Path, str | None]:
    """Find the multi-module Maven parent root if tests are in a different module."""
    test_file_paths: list[Path] = []
    if hasattr(test_paths, "test_files"):
        for test_file in test_paths.test_files:
            if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                test_file_paths.append(test_file.benchmarking_file_path)
            elif hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
                test_file_paths.append(test_file.instrumented_behavior_file_path)
    elif isinstance(test_paths, (list, tuple)):
        test_file_paths = [Path(p) if isinstance(p, str) else p for p in test_paths]

    if not test_file_paths:
        return project_root, None

    test_outside_project = False
    test_dir: Path | None = None
    for test_path in test_file_paths:
        try:
            test_path.relative_to(project_root)
        except ValueError:
            test_outside_project = True
            test_dir = test_path.parent
            break

    if not test_outside_project:
        pom_path = project_root / "pom.xml"
        if pom_path.exists():
            try:
                content = pom_path.read_text(encoding="utf-8")
                if "<modules>" in content:
                    modules = _extract_modules_from_pom_content(content)
                    for test_path in test_file_paths:
                        try:
                            rel_path = test_path.relative_to(project_root)
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

    current = project_root.parent
    while current != current.parent:
        pom_path = current / "pom.xml"
        if pom_path.exists():
            try:
                content = pom_path.read_text(encoding="utf-8")
                if "<modules>" in content:
                    if test_dir:
                        try:
                            test_module = test_dir.relative_to(current)
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
    """Get the target directory for the test module."""
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
    """Run behavioral tests for Java code."""
    project_root = project_root or cwd

    maven_root, test_module = _find_multi_module_root(project_root, test_paths)

    sqlite_db_path = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))

    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_LOOP_INDEX"] = "1"
    run_env["CODEFLASH_MODE"] = "behavior"
    run_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    run_env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_db_path)

    coverage_xml_path: Path | None = None
    if enable_coverage:
        if test_module:
            test_module_pom = maven_root / test_module / "pom.xml"
            if test_module_pom.exists():
                if not is_jacoco_configured(test_module_pom):
                    logger.info(f"Adding JaCoCo plugin to test module pom.xml: {test_module_pom}")
                    add_jacoco_plugin_to_pom(test_module_pom)
                coverage_xml_path = get_jacoco_xml_path(maven_root / test_module)
        else:
            pom_path = project_root / "pom.xml"
            if pom_path.exists():
                if not is_jacoco_configured(pom_path):
                    logger.info("Adding JaCoCo plugin to pom.xml for coverage collection")
                    add_jacoco_plugin_to_pom(pom_path)
                coverage_xml_path = get_jacoco_xml_path(project_root)

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

    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, candidate_index)

    return result_xml_path, result, sqlite_db_path, coverage_xml_path


def _compile_tests(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Compile test code using Maven (without running tests)."""
    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found")
        return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

    cmd = [mvn, "test-compile", "-e"]

    # Add Maven profiles if configured
    maven_profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
    if maven_profiles:
        cmd.extend(["-P", maven_profiles])

    if test_module:
        cmd.extend(["-pl", test_module, "-am"])

    logger.debug("Compiling tests: %s in %s", " ".join(cmd), project_root)

    try:
        return subprocess.run(
            cmd, check=False, cwd=project_root, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.exception("Maven compilation timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd, returncode=-2, stdout="", stderr=f"Compilation timed out after {timeout} seconds"
        )
    except Exception as e:
        logger.exception("Maven compilation failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))


def _get_test_classpath(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 60
) -> str | None:
    """Get the test classpath from Maven."""
    mvn = find_maven_executable()
    if not mvn:
        return None

    cp_file = project_root / ".codeflash_classpath.txt"

    cmd = [mvn, "dependency:build-classpath", "-DincludeScope=test", f"-Dmdep.outputFile={cp_file}", "-q"]

    if test_module:
        cmd.extend(["-pl", test_module])

    logger.debug("Getting classpath: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, check=False, cwd=project_root, env=env, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            logger.error("Failed to get classpath: %s", result.stderr)
            return None

        if not cp_file.exists():
            logger.error("Classpath file not created")
            return None

        classpath = cp_file.read_text(encoding="utf-8").strip()

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
        logger.exception("Getting classpath timed out")
        return None
    except Exception as e:
        logger.exception("Failed to get classpath: %s", e)
        return None
    finally:
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
    """Run JUnit tests directly using java command (bypassing Maven)."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java = Path(java_home) / "bin" / "java"
        if not java.exists():
            java = "java"
    else:
        java = "java"

    cmd = [
        str(java),
        "-cp",
        classpath,
        "org.junit.platform.console.ConsoleLauncher",
        "--disable-banner",
        "--disable-ansi-colors",
        "--details=none",
        "--config=junit.platform.output.capture.stdout=true",
        "--config=junit.platform.output.capture.stderr=true",
    ]

    if reports_dir:
        reports_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--reports-dir", str(reports_dir)])

    for test_class in test_classes:
        cmd.extend(["--select-class", test_class])

    logger.debug("Running tests directly: java -cp ... ConsoleLauncher --select-class %s", test_classes)

    try:
        return subprocess.run(
            cmd, check=False, cwd=working_dir, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.exception("Direct test execution timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd, returncode=-2, stdout="", stderr=f"Test execution timed out after {timeout} seconds"
        )
    except Exception as e:
        logger.exception("Direct test execution failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))


def _get_test_class_names(test_paths: Any, mode: str = "performance") -> list[str]:
    """Extract fully qualified test class names from test paths."""
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
    """Return an empty result for when no tests can be run."""
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    surefire_dir = target_dir / "surefire-reports"
    result_xml_path = _get_combined_junit_xml(surefire_dir, -1)

    empty_result = subprocess.CompletedProcess(
        args=["java", "-cp", "...", "ConsoleLauncher"], returncode=-1, stdout="", stderr="No test classes found"
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
    """Fallback: Run benchmarking tests using Maven (slower but more reliable)."""
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
            maven_root, test_paths, run_env, timeout=per_loop_timeout, mode="performance", test_module=test_module
        )

        last_result = result
        loop_count = loop_idx

        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug("Stopping Maven benchmark after %d loops (%.2fs elapsed)", loop_idx, elapsed)
            break

        if result.returncode != 0:
            timing_pattern = re.compile(r"!######[^:]*:[^:]*:[^:]*:[^:]*:[^:]+:[^:]+######!")
            has_timing_markers = bool(timing_pattern.search(result.stdout or ""))
            if not has_timing_markers:
                logger.warning("Tests failed in Maven loop %d with no timing markers, stopping", loop_idx)
                break
            logger.debug("Some tests failed in Maven loop %d but timing markers present, continuing", loop_idx)

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
    inner_iterations: int = 10,
) -> tuple[Path, Any]:
    """Run benchmarking tests for Java code with compile-once-run-many optimization."""
    import time

    project_root = project_root or cwd

    maven_root, test_module = _find_multi_module_root(project_root, test_paths)

    test_classes = _get_test_class_names(test_paths, mode="performance")
    if not test_classes:
        logger.error("No test classes found")
        return _get_empty_result(maven_root, test_module)

    compile_env = os.environ.copy()
    compile_env.update(test_env)

    logger.debug("Step 1: Compiling tests (one-time Maven overhead)")
    compile_start = time.time()
    compile_result = _compile_tests(maven_root, compile_env, test_module, timeout=120)
    compile_time = time.time() - compile_start

    if compile_result.returncode != 0:
        logger.error(
            "Test compilation failed (rc=%d):\nstdout: %s\nstderr: %s",
            compile_result.returncode,
            compile_result.stdout,
            compile_result.stderr,
        )
        logger.warning("Falling back to Maven-based test execution")
        return _run_benchmarking_tests_maven(
            test_paths, test_env, cwd, timeout, project_root,
            min_loops, max_loops, target_duration_seconds, inner_iterations,
        )

    logger.debug("Compilation completed in %.2fs", compile_time)

    logger.debug("Step 2: Getting classpath")
    classpath = _get_test_classpath(maven_root, compile_env, test_module, timeout=60)

    if not classpath:
        logger.warning("Failed to get classpath, falling back to Maven-based execution")
        return _run_benchmarking_tests_maven(
            test_paths, test_env, cwd, timeout, project_root,
            min_loops, max_loops, target_duration_seconds, inner_iterations,
        )

    logger.debug("Step 3: Running tests directly (bypassing Maven)")

    all_stdout = []
    all_stderr = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    per_loop_timeout = timeout or max(60, 30 + inner_iterations // 10)

    if test_module:
        working_dir = maven_root / test_module
    else:
        working_dir = maven_root

    target_dir = _get_test_module_target_dir(maven_root, test_module)
    reports_dir = target_dir / "surefire-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for loop_idx in range(1, max_loops + 1):
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

        loop_start = time.time()
        result = _run_tests_direct(
            classpath, test_classes, run_env, working_dir, timeout=per_loop_timeout, reports_dir=reports_dir
        )
        loop_time = time.time() - loop_start

        last_result = result
        loop_count = loop_idx

        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        logger.debug("Loop %d completed in %.2fs (returncode=%d)", loop_idx, loop_time, result.returncode)

        if loop_idx == 1 and result.returncode != 0 and result.stderr and "ConsoleLauncher" in result.stderr:
            logger.debug("JUnit Console Launcher not available, falling back to Maven-based execution")
            return _run_benchmarking_tests_maven(
                test_paths, test_env, cwd, timeout, project_root,
                min_loops, max_loops, target_duration_seconds, inner_iterations,
            )

        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug(
                "Stopping benchmark after %d loops (%.2fs elapsed, target: %.2fs, %d inner iterations each)",
                loop_idx, elapsed, target_duration_seconds, inner_iterations,
            )
            break

        if result.returncode != 0:
            timing_pattern = re.compile(r"!######[^:]*:[^:]*:[^:]*:[^:]*:[^:]+:[^:]+######!")
            has_timing_markers = bool(timing_pattern.search(result.stdout or ""))
            if not has_timing_markers:
                logger.warning("Tests failed in loop %d with no timing markers, stopping benchmark", loop_idx)
                break
            logger.debug("Some tests failed in loop %d but timing markers present, continuing", loop_idx)

    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)

    total_time = time.time() - total_start_time
    total_iterations = loop_count * inner_iterations
    logger.debug(
        "Completed %d loops x %d inner iterations = %d total iterations in %.2fs (compile: %.2fs)",
        loop_count, inner_iterations, total_iterations, total_time, compile_time,
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


def _get_combined_junit_xml(surefire_dir: Path, candidate_index: int) -> Path:
    """Get or create a combined JUnit XML file from Surefire reports."""
    result_id = uuid.uuid4().hex[:8]
    result_xml_path = Path(tempfile.gettempdir()) / f"codeflash_java_results_{candidate_index}_{result_id}.xml"

    if not surefire_dir.exists():
        _write_empty_junit_xml(result_xml_path)
        return result_xml_path

    xml_files = list(surefire_dir.glob("TEST-*.xml"))

    if not xml_files:
        _write_empty_junit_xml(result_xml_path)
        return result_xml_path

    if len(xml_files) == 1:
        shutil.copy(xml_files[0], result_xml_path)
        return result_xml_path

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
    """Combine multiple JUnit XML files into one."""
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

            total_tests += int(root.get("tests", 0))
            total_failures += int(root.get("failures", 0))
            total_errors += int(root.get("errors", 0))
            total_skipped += int(root.get("skipped", 0))
            total_time += float(root.get("time", 0))

            for testcase in root.findall(".//testcase"):
                all_testcases.append(testcase)

        except Exception as e:
            logger.warning("Failed to parse %s: %s", xml_file, e)

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
    """Run Maven tests with Surefire."""
    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found")
        return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

    test_filter = _build_test_filter(test_paths, mode=mode)
    logger.debug(f"Built test filter for mode={mode}: '{test_filter}' (empty={not test_filter})")
    logger.debug(f"test_paths type: {type(test_paths)}, has test_files: {hasattr(test_paths, 'test_files')}")
    if hasattr(test_paths, "test_files"):
        logger.debug(f"Number of test files: {len(test_paths.test_files)}")
        for i, tf in enumerate(test_paths.test_files[:3]):
            logger.debug(
                f"  TestFile[{i}]: behavior={tf.instrumented_behavior_file_path},"
                f" bench={tf.benchmarking_file_path}"
            )

    maven_goal = "verify" if enable_coverage else "test"
    cmd = [mvn, maven_goal, "-fae"]

    # Add Maven profiles if configured via environment variable
    maven_profiles = os.environ.get("CODEFLASH_MAVEN_PROFILES", "").strip()
    if maven_profiles:
        cmd.extend(["-P", maven_profiles])
        logger.debug("Using Maven profiles: %s", maven_profiles)

    if enable_coverage:
        cmd.append("-Dmaven.test.failure.ignore=true")

    if test_module:
        cmd.extend(["-pl", test_module, "-am", "-DfailIfNoTests=false", "-DskipTests=false"])

    if test_filter:
        validated_filter = _validate_test_filter(test_filter)
        cmd.append(f"-Dtest={validated_filter}")
        logger.debug(f"Added -Dtest={validated_filter} to Maven command")
    else:
        error_msg = (
            f"Test filter is EMPTY for mode={mode}! "
            f"Maven will run ALL tests instead of the specified tests. "
            f"This indicates a problem with test file instrumentation or path resolution."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("Running Maven command: %s in %s", " ".join(cmd), project_root)

    try:
        return subprocess.run(
            cmd, check=False, cwd=project_root, env=env, capture_output=True, text=True, timeout=timeout
        )

    except subprocess.TimeoutExpired:
        logger.exception("Maven test execution timed out after %d seconds", timeout)
        return subprocess.CompletedProcess(
            args=cmd, returncode=-2, stdout="", stderr=f"Test execution timed out after {timeout} seconds"
        )
    except Exception as e:
        logger.exception("Maven test execution failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))


def _build_test_filter(test_paths: Any, mode: str = "behavior") -> str:
    """Build a Maven Surefire test filter from test paths."""
    if not test_paths:
        logger.debug("_build_test_filter: test_paths is empty/None")
        return ""

    if isinstance(test_paths, (list, tuple)):
        filters = []
        for path in test_paths:
            if isinstance(path, Path):
                class_name = _path_to_class_name(path)
                if class_name:
                    filters.append(class_name)
                else:
                    logger.debug(f"_build_test_filter: Could not convert path to class name: {path}")
            elif isinstance(path, str):
                filters.append(path)
        result = ",".join(filters) if filters else ""
        logger.debug(f"_build_test_filter (list/tuple): {len(filters)} filters -> '{result}'")
        return result

    if hasattr(test_paths, "test_files"):
        filters = []
        skipped = 0
        skipped_reasons = []

        for test_file in test_paths.test_files:
            if mode == "performance":
                if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                    class_name = _path_to_class_name(test_file.benchmarking_file_path)
                    if class_name:
                        filters.append(class_name)
                    else:
                        reason = (
                            "Could not convert benchmarking path to class name:"
                            f" {test_file.benchmarking_file_path}"
                        )
                        logger.debug(f"_build_test_filter: {reason}")
                        skipped += 1
                        skipped_reasons.append(reason)
                else:
                    reason = (
                        "TestFile has no benchmarking_file_path"
                        f" (original: {test_file.original_file_path})"
                    )
                    logger.warning(f"_build_test_filter: {reason}")
                    skipped += 1
                    skipped_reasons.append(reason)
            elif (
                hasattr(test_file, "instrumented_behavior_file_path")
                and test_file.instrumented_behavior_file_path
            ):
                class_name = _path_to_class_name(test_file.instrumented_behavior_file_path)
                if class_name:
                    filters.append(class_name)
                else:
                    reason = (
                        "Could not convert behavior path to class name:"
                        f" {test_file.instrumented_behavior_file_path}"
                    )
                    logger.debug(f"_build_test_filter: {reason}")
                    skipped += 1
                    skipped_reasons.append(reason)
            else:
                reason = (
                    "TestFile has no instrumented_behavior_file_path"
                    f" (original: {test_file.original_file_path})"
                )
                logger.warning(f"_build_test_filter: {reason}")
                skipped += 1
                skipped_reasons.append(reason)

        result = ",".join(filters) if filters else ""
        logger.debug(
            f"_build_test_filter (TestFiles): {len(filters)} filters, {skipped} skipped -> '{result}'"
        )

        if not filters and skipped > 0:
            logger.error(
                f"All {skipped} test files were skipped in _build_test_filter! "
                f"Mode: {mode}. This will cause an empty test filter. "
                f"Reasons: {skipped_reasons[:5]}"
            )

        return result

    logger.debug(f"_build_test_filter: Unknown test_paths type: {type(test_paths)}")
    return ""


def _path_to_class_name(path: Path, source_dirs: list[str] | None = None) -> str | None:
    """Convert a test file path to a Java class name.

    Args:
        path: Path to the test file.
        source_dirs: Optional list of custom source directory suffixes to try
            (e.g., ["src/main/custom", "app/java"]). These are matched against
            the path before standard Maven directories.

    Returns:
        Fully qualified class name, or None if unable to determine.

    """
    if path.suffix != ".java":
        return None

    # Step 1: Try matching against provided custom source directories
    if source_dirs:
        path_str = str(path).replace("\\", "/")
        for src_dir in source_dirs:
            normalized = src_dir.replace("\\", "/").rstrip("/") + "/"
            idx = path_str.find(normalized)
            if idx != -1:
                remainder = path_str[idx + len(normalized) :]
                remainder = remainder.removesuffix(".java")
                return remainder.replace("/", ".")

    # Step 2: Try standard Maven/Gradle source directories
    parts = path.parts

    java_idx = None
    for i, part in enumerate(parts):
        if part == "java" and i > 0 and parts[i - 1] in ("main", "test"):
            java_idx = i
            break

    if java_idx is not None:
        class_parts = list(parts[java_idx + 1 :])
        class_parts[-1] = class_parts[-1].replace(".java", "")
        return ".".join(class_parts)

    # Step 3: Find the last 'java' in path as a fallback heuristic
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "java":
            class_parts = list(parts[i + 1 :])
            class_parts[-1] = class_parts[-1].replace(".java", "")
            return ".".join(class_parts)

    return path.stem


def _extract_source_dirs_from_pom(project_root: Path) -> list[str]:
    """Extract custom source and test source directories from pom.xml."""
    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return []

    try:
        content = pom_path.read_text(encoding="utf-8")
        root = ET.fromstring(content)
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        source_dirs: list[str] = []
        standard_dirs = {
            "src/main/java",
            "src/test/java",
            "${project.basedir}/src/main/java",
            "${project.basedir}/src/test/java",
        }

        for build in [root.find("m:build", ns), root.find("build")]:
            if build is not None:
                for tag in ("sourceDirectory", "testSourceDirectory"):
                    for elem in [build.find(f"m:{tag}", ns), build.find(tag)]:
                        if elem is not None and elem.text:
                            dir_text = elem.text.strip()
                            if dir_text not in standard_dirs:
                                source_dirs.append(dir_text)

        return source_dirs
    except ET.ParseError:
        logger.debug("Failed to parse pom.xml for source directories")
        return []
    except Exception:
        logger.debug("Error reading pom.xml for source directories")
        return []


def run_tests(
    test_files: list[Path], cwd: Path, env: dict[str, str], timeout: int
) -> tuple[list[TestResult], Path]:
    """Run tests and return results."""
    result = _run_maven_tests(cwd, test_files, env, timeout)

    surefire_dir = cwd / "target" / "surefire-reports"
    test_results = parse_surefire_results(surefire_dir)

    junit_files = list(surefire_dir.glob("TEST-*.xml")) if surefire_dir.exists() else []
    junit_path = (
        junit_files[0] if junit_files else cwd / "target" / "surefire-reports" / "test-results.xml"
    )

    return test_results, junit_path


def parse_test_results(junit_xml_path: Path, stdout: str) -> list[TestResult]:
    """Parse test results from JUnit XML and stdout."""
    return parse_surefire_results(junit_xml_path.parent)


def parse_surefire_results(surefire_dir: Path) -> list[TestResult]:
    """Parse Maven Surefire XML reports into TestResult objects."""
    results: list[TestResult] = []

    if not surefire_dir.exists():
        return results

    for xml_file in surefire_dir.glob("TEST-*.xml"):
        results.extend(_parse_surefire_xml(xml_file))

    return results


def _parse_surefire_xml(xml_file: Path) -> list[TestResult]:
    """Parse a single Surefire XML file."""
    results: list[TestResult] = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        class_name = root.get("name", "")  # noqa: F841

        for testcase in root.findall(".//testcase"):
            test_name = testcase.get("name", "")
            test_time = float(testcase.get("time", "0"))
            runtime_ns = int(test_time * 1_000_000_000)

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
    project_root: Path, test_classes: list[str] | None = None
) -> list[str]:
    """Get the command to run Java tests."""
    mvn = find_maven_executable() or "mvn"

    cmd = [mvn, "test"]

    if test_classes:
        validated_classes = []
        for test_class in test_classes:
            if not _validate_java_class_name(test_class):
                msg = (
                    f"Invalid test class name: '{test_class}'."
                    " Test names must follow Java identifier rules."
                )
                raise ValueError(msg)
            validated_classes.append(test_class)

        cmd.append(f"-Dtest={','.join(validated_classes)}")

    return cmd
