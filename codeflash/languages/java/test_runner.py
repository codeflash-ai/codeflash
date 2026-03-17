"""Java test runner for JUnit 5 with build tool strategy.

This module provides functionality to run JUnit 5 tests, supporting both
behavioral testing and benchmarking modes. Build-tool-specific operations
(compilation, classpath, etc.) are delegated to BuildToolStrategy implementations.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.languages.base import TestResult

_result_counter = itertools.count(1)


class CompilationCache:
    _instance: CompilationCache | None = None

    def __init__(self) -> None:
        self.tests_compiled: set[tuple[Path, str | None]] = set()
        self.last_compiled_candidate: dict[tuple[Path, str | None], int] = {}
        self.failed: set[tuple[Path, str | None]] = set()

    @classmethod
    def get(cls) -> CompilationCache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def clear(cls) -> None:
        cls._instance = None

    def should_skip(self, build_root: Path, test_module: str | None, candidate_index: int) -> bool:
        return self.last_compiled_candidate.get((build_root, test_module)) == candidate_index

    def needs_source_only(self, build_root: Path, test_module: str | None) -> bool:
        return (build_root, test_module) in self.tests_compiled

    def is_failed(self, build_root: Path, test_module: str | None) -> bool:
        return (build_root, test_module) in self.failed

    def mark_failed(self, build_root: Path, test_module: str | None) -> None:
        self.failed.add((build_root, test_module))

    def mark_compiled(
        self, build_root: Path, test_module: str | None, candidate_index: int, *, tests: bool = False
    ) -> None:
        key = (build_root, test_module)
        self.last_compiled_candidate[key] = candidate_index
        if tests:
            self.tests_compiled.add(key)


_MAVEN_NS = "http://maven.apache.org/POM/4.0.0"

_M_MODULES_TAG = f"{{{_MAVEN_NS}}}modules"

logger = logging.getLogger(__name__)

# Regex pattern for valid Java class names (package.ClassName format)
# Allows: letters, digits, underscores, dots, and dollar signs (inner classes)
_VALID_JAVA_CLASS_NAME = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$.]*$")


def _run_cmd_kill_pg_on_timeout(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, killing its entire process group on timeout (POSIX only).

    On POSIX systems this function uses start_new_session=True so the child
    process gets its own process group.  When the timeout fires we send SIGTERM
    (then SIGKILL) to the whole process group, not just the process itself.
    This is critical for Maven, which forks child JVM processes (Maven Surefire
    forks) that would otherwise become orphaned when the Maven parent is killed
    by a plain subprocess.run() timeout.  Orphaned JVMs keep SQLite
    file-handles open, causing "database is locked" errors.

    On Windows, process groups work differently (no POSIX signals / killpg), so
    we fall back to plain subprocess.run() which kills only the parent process.
    """
    if sys.platform == "win32":
        try:
            return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=text, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2, stdout="", stderr=f"Process timed out after {timeout}s"
            )

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, start_new_session=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
        try:
            stdout_data = proc.stdout.read() if proc.stdout else ""
            stderr_data = proc.stderr.read() if proc.stderr else ""
        except Exception:
            stdout_data, stderr_data = "", ""
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=-2,
            stdout=stdout_data,
            stderr=f"Process group killed after timeout ({timeout}s): {stderr_data}",
        )


def _validate_java_class_name(class_name: str) -> bool:
    """Validate that a string is a valid Java class name.

    This prevents command injection when passing test class names to Maven.
    """
    return bool(_VALID_JAVA_CLASS_NAME.match(class_name))


def _extract_modules_from_pom_content(content: str) -> list[str]:
    """Extract module names from Maven POM XML content using proper XML parsing.

    Handles both namespaced and non-namespaced POMs.
    """
    if "modules" not in content:
        return []

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        logger.debug("Failed to parse POM XML for module extraction")
        return []

    modules_elem = root.find(_M_MODULES_TAG)
    if modules_elem is None:
        modules_elem = root.find("modules")

    if modules_elem is None:
        return []

    return [m.text for m in modules_elem if m.text]


def _validate_test_filter(test_filter: str) -> str:
    """Validate and sanitize a test filter string for Maven.

    Test filters can contain commas (multiple classes) and wildcards (*).
    This function validates the format to prevent command injection.
    """
    # Iterate over comma-separated parts without building an intermediate list.
    for raw in test_filter.split(","):
        pattern = raw.strip()

        # Avoid allocating a new string if there is no wildcard.
        if "*" in pattern:
            name_to_validate = pattern.replace("*", "A")
        else:
            name_to_validate = pattern

        if _VALID_JAVA_CLASS_NAME.match(name_to_validate) is None:
            msg = (
                f"Invalid test class name or pattern: '{pattern}'. "
                f"Test names must follow Java identifier rules (letters, digits, underscores, dots, dollar signs)."
            )
            raise ValueError(msg)

    return test_filter


def _extract_modules_from_settings_gradle(content: str) -> list[str]:
    """Extract module names from settings.gradle(.kts) content.

    Looks for include directives like:
        include("module-a", "module-b")   // Kotlin DSL
        include 'module-a', 'module-b'    // Groovy DSL
    Module names may be prefixed with ':' which is stripped.
    """
    modules: list[str] = []
    for match in re.findall(r"""include\s*\(?[^)\n]*\)?""", content):
        for name in re.findall(r"""['"]([^'"]+)['"]""", match):
            modules.append(name.lstrip(":"))
    return modules


def _detect_modules(directory: Path) -> list[str]:
    """Detect sub-modules in a directory for both Maven and Gradle projects."""
    pom_path = directory / "pom.xml"
    if pom_path.exists():
        try:
            content = pom_path.read_text(encoding="utf-8")
            modules = _extract_modules_from_pom_content(content)
            if modules:
                return modules
        except Exception:
            pass

    for settings_name in ("settings.gradle.kts", "settings.gradle"):
        settings_path = directory / settings_name
        if settings_path.exists():
            try:
                content = settings_path.read_text(encoding="utf-8")
                modules = _extract_modules_from_settings_gradle(content)
                if modules:
                    return modules
            except Exception:
                pass

    return []


def _is_build_root(directory: Path) -> bool:
    """Check if a directory is a build root (has a build config file)."""
    return (
        (directory / "pom.xml").exists()
        or (directory / "build.gradle").exists()
        or (directory / "build.gradle.kts").exists()
    )


def _find_multi_module_root(project_root: Path, test_paths: Any) -> tuple[Path, str | None]:
    """Find the multi-module parent root if tests are in a different module.

    Works for both Maven and Gradle multi-module projects.

    Returns:
        Tuple of (build_root, test_module_name) where:
        - build_root: The directory to run the build tool from (parent if multi-module, else project_root)
        - test_module_name: The name of the test module if different from project_root, else None

    """
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
        modules = _detect_modules(project_root)
        if modules:
            for test_path in test_file_paths:
                try:
                    rel_path = test_path.relative_to(project_root)
                    first_component = rel_path.parts[0] if rel_path.parts else None
                    if first_component and first_component in modules:
                        logger.debug(
                            "Detected multi-module project. Root: %s, Test module: %s", project_root, first_component
                        )
                        return project_root, first_component
                except ValueError:
                    pass
        return project_root, None

    current = project_root.parent
    while current != current.parent:
        if _is_build_root(current):
            modules = _detect_modules(current)
            if modules and test_dir:
                try:
                    test_module = test_dir.relative_to(current)
                    test_module_name = test_module.parts[0] if test_module.parts else None
                    logger.debug("Detected multi-module project. Root: %s, Test module: %s", current, test_module_name)
                    return current, test_module_name
                except ValueError:
                    pass
        current = current.parent

    return project_root, None


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


# ---------------------------------------------------------------------------
# Entry points — use BuildToolStrategy for build-tool-specific operations
# ---------------------------------------------------------------------------


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

    Returns:
        Tuple of (result_xml_path, subprocess_result, coverage_xml_path, None).

    """
    from codeflash.languages.java.build_tool_strategy import get_strategy

    project_root = project_root or cwd
    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    strategy = get_strategy(build_root)

    # Ensure codeflash-runtime is installed and added as dependency before compilation
    strategy.ensure_runtime(build_root, test_module)

    # Pre-install multi-module deps
    base_env = os.environ.copy()
    base_env.update(test_env)
    strategy.install_multi_module_deps(build_root, test_module, base_env)

    # Create SQLite database path for behavior capture
    sqlite_db_path = get_run_tmp_file(Path(f"test_return_values_{candidate_index}.sqlite"))

    # Set environment variables for timing instrumentation and behavior capture
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_LOOP_INDEX"] = "1"
    run_env["CODEFLASH_MODE"] = "behavior"
    run_env["CODEFLASH_TEST_ITERATION"] = str(candidate_index)
    run_env["CODEFLASH_OUTPUT_FILE"] = str(sqlite_db_path)

    # Coverage handling
    coverage_xml_path: Path | None = None
    if enable_coverage:
        coverage_xml_path = strategy.setup_coverage(build_root, test_module, project_root)

    min_timeout = 300 if enable_coverage else 60
    effective_timeout = max(timeout or 300, min_timeout)

    if enable_coverage:
        # Coverage MUST use build tool — JaCoCo runs as a plugin during the verify phase
        result, result_xml_path, coverage_xml_path = strategy.run_tests_with_coverage(
            build_root, test_module, test_paths, run_env, effective_timeout, candidate_index
        )
    else:
        # Direct JVM execution (fast path — bypasses build tool overhead)
        result, result_xml_path = _run_direct_or_fallback(
            strategy,
            build_root,
            test_module,
            test_paths,
            run_env,
            effective_timeout,
            mode="behavior",
            candidate_index=candidate_index,
        )

    # Debug: Log coverage file status
    if enable_coverage:
        logger.info("%s verify completed with return code: %s", strategy.name, result.returncode)
        if result.returncode != 0:
            logger.warning(
                "%s verify had non-zero return code: %s. Coverage data may be incomplete.",
                strategy.name,
                result.returncode,
            )

    if enable_coverage and coverage_xml_path:
        target_dir = strategy.get_build_output_dir(build_root, test_module)
        jacoco_exec_path = target_dir / "jacoco.exec"
        logger.info("Coverage paths - target_dir: %s, coverage_xml_path: %s", target_dir, coverage_xml_path)
        if jacoco_exec_path.exists():
            logger.info("JaCoCo exec file exists: %s (%s bytes)", jacoco_exec_path, jacoco_exec_path.stat().st_size)
        else:
            logger.warning("JaCoCo exec file not found: %s - JaCoCo agent may not have run", jacoco_exec_path)
        if coverage_xml_path.exists():
            file_size = coverage_xml_path.stat().st_size
            logger.info("JaCoCo XML report exists: %s (%s bytes)", coverage_xml_path, file_size)
            if file_size == 0:
                logger.warning("JaCoCo XML report is empty - report generation may have failed")
        else:
            logger.warning("JaCoCo XML report not found: %s - verify phase may not have completed", coverage_xml_path)

    return result_xml_path, result, coverage_xml_path, None


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
    """Run benchmarking tests for Java code with compile-once-run-many optimization.

    This compiles tests once, then runs them multiple times directly via JVM,
    bypassing build tool overhead (~500ms vs ~5-10s per invocation).

    Returns:
        Tuple of (result_file_path, subprocess_result with aggregated stdout).

    """
    import time

    from codeflash.languages.java.build_tool_strategy import get_strategy

    project_root = project_root or cwd
    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    strategy = get_strategy(build_root)

    # Ensure codeflash-runtime is installed and added as dependency before compilation
    strategy.ensure_runtime(build_root, test_module)

    # Pre-install multi-module deps
    base_env = os.environ.copy()
    base_env.update(test_env)
    strategy.install_multi_module_deps(build_root, test_module, base_env)

    # Get test class names
    test_classes = _get_test_class_names(test_paths, mode="performance")
    if not test_classes:
        logger.error("No test classes found")
        return _get_empty_result(strategy, build_root, test_module)

    # Step 1: Compile tests once using build tool (skip if already compiled for this candidate)
    compile_env = os.environ.copy()
    compile_env.update(test_env)

    candidate_index = int(test_env.get("CODEFLASH_TEST_ITERATION", -1))
    compile_time = 0.0
    cache = CompilationCache.get()
    if cache.is_failed(build_root, test_module):
        logger.debug("Step 1: Skipping — test compilation previously failed, skipping all candidates")
        return _get_empty_result(strategy, build_root, test_module)
    if cache.should_skip(build_root, test_module, candidate_index):
        logger.debug("Step 1: Skipping compilation — already compiled for candidate %d", candidate_index)
    elif cache.needs_source_only(build_root, test_module):
        logger.debug("Step 1: Compiling source only (tests already compiled)")
        compile_start = time.time()
        compile_result = strategy.compile_source_only(build_root, compile_env, test_module, timeout=120)
        if compile_result.returncode != 0:
            logger.warning("Source compilation failed for candidate %d, skipping", candidate_index)
            return _get_empty_result(strategy, build_root, test_module)
        cache.mark_compiled(build_root, test_module, candidate_index)
        compile_time = time.time() - compile_start
        logger.debug("Source compilation completed in %.2fs", compile_time)
    else:
        logger.debug("Step 1: Compiling tests + source (one-time %s overhead)", strategy.name)
        compile_start = time.time()
        compile_result = strategy.compile_tests(build_root, compile_env, test_module, timeout=120)
        compile_time = time.time() - compile_start

        if compile_result.returncode != 0:
            logger.error(
                "Test compilation failed (rc=%d):\nstdout: %s\nstderr: %s",
                compile_result.returncode,
                compile_result.stdout,
                compile_result.stderr,
            )
            cache.mark_failed(build_root, test_module)
            return _get_empty_result(strategy, build_root, test_module)

        logger.debug("Compilation completed in %.2fs", compile_time)
        cache.mark_compiled(build_root, test_module, candidate_index, tests=True)

    # Step 2: Get classpath from build tool
    logger.debug("Step 2: Getting classpath")
    classpath = strategy.get_classpath(build_root, compile_env, test_module, timeout=60)

    if not classpath:
        logger.warning("Failed to get classpath, falling back to %s-based execution", strategy.name)
        return strategy.run_benchmarking_via_build_tool(
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

    # Step 3: Run tests multiple times directly via JVM
    logger.debug("Step 3: Running tests directly (bypassing %s)", strategy.name)

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    per_loop_timeout = timeout or max(60, 30 + inner_iterations // 10)

    working_dir = build_root / test_module if test_module else build_root
    reports_dir = strategy.get_reports_dir(build_root, test_module)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for loop_idx in range(1, max_loops + 1):
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        if "CODEFLASH_INNER_ITERATIONS" not in run_env:
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

        if result.returncode != 0 and result.stderr:
            logger.debug("Direct JVM stderr: %s", result.stderr[:500])

        # Check if direct JVM execution failed on the first loop — fall back to build tool
        should_fallback = False
        if loop_idx == 1 and result.returncode != 0:
            combined_output = (result.stderr or "") + (result.stdout or "")
            fallback_indicators = [
                "ConsoleLauncher",
                "ClassNotFoundException",
                "No tests were executed",
                "Unable to locate a Java Runtime",
                "No tests found",
            ]
            should_fallback = any(indicator in combined_output for indicator in fallback_indicators)
            if not should_fallback:
                has_markers = bool(re.search(r"!######", result.stdout or ""))
                if not has_markers and result.returncode != 0:
                    should_fallback = True
                    logger.debug("Direct execution failed with no timing markers, likely JUnit version mismatch")

        if should_fallback:
            logger.debug("Direct JVM execution failed, falling back to %s-based execution", strategy.name)
            return strategy.run_benchmarking_via_build_tool(
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
        loop_count,
        inner_iterations,
        total_iterations,
        total_time,
        compile_time,
    )

    combined_result = subprocess.CompletedProcess(
        args=last_result.args if last_result else ["java", "-cp", "..."],
        returncode=last_result.returncode if last_result else -1,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )

    result_xml_path = _get_combined_junit_xml(reports_dir, -1)

    return result_xml_path, combined_result


def run_line_profile_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    line_profile_output_file: Path | None = None,
    javaagent_arg: str | None = None,
) -> tuple[Path, Any]:
    """Run tests with the profiler agent attached.

    The agent instruments bytecode at class-load time — no source modification needed.
    Profiling results are written to line_profile_output_file on JVM exit.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    from codeflash.languages.java.build_tool_strategy import get_strategy

    project_root = project_root or cwd
    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    strategy = get_strategy(build_root)

    # Ensure codeflash-runtime is installed
    strategy.ensure_runtime(build_root, test_module)

    # Pre-install multi-module deps
    base_env = os.environ.copy()
    base_env.update(test_env)
    strategy.install_multi_module_deps(build_root, test_module, base_env)

    # Set up environment with profiling mode
    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_MODE"] = "line_profile"
    if line_profile_output_file:
        run_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    min_timeout = 120
    effective_timeout = max(timeout or min_timeout, min_timeout)
    logger.debug("Running line profiling tests (single run) with timeout=%ds", effective_timeout)
    result = strategy.run_tests_via_build_tool(
        build_root,
        test_paths,
        run_env,
        timeout=effective_timeout,
        mode="line_profile",
        test_module=test_module,
        javaagent_arg=javaagent_arg,
    )

    reports_dir = strategy.get_reports_dir(build_root, test_module)
    result_xml_path = _get_combined_junit_xml(reports_dir, -1)

    return result_xml_path, result


# ---------------------------------------------------------------------------
# Shared helpers — used by both entry points and strategy implementations
# ---------------------------------------------------------------------------


def _run_direct_or_fallback(
    strategy: Any,
    build_root: Path,
    test_module: str | None,
    test_paths: Any,
    run_env: dict[str, str],
    timeout: int,
    mode: str,
    candidate_index: int = -1,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Compile once, then run tests directly via JVM. Falls back to build tool on failure."""
    test_classes = _get_test_class_names(test_paths, mode=mode)
    if not test_classes:
        logger.warning("No test classes found for mode=%s, returning empty result", mode)
        result_xml_path, empty_result = _get_empty_result(strategy, build_root, test_module)
        return empty_result, result_xml_path

    # Step 1: Compile (4-tier: failed / skip / source-only / full)
    cache = CompilationCache.get()
    if cache.is_failed(build_root, test_module):
        logger.debug("Step 1: Skipping — test compilation previously failed, skipping all candidates")
        result_xml_path, empty_result = _get_empty_result(strategy, build_root, test_module)
        return empty_result, result_xml_path
    if cache.should_skip(build_root, test_module, candidate_index):
        logger.debug("Step 1: Skipping compilation — already compiled for candidate %d", candidate_index)
    elif cache.needs_source_only(build_root, test_module):
        logger.debug("Step 1: Compiling source only (tests already compiled)")
        compile_result = strategy.compile_source_only(build_root, run_env, test_module, timeout=120)
        if compile_result.returncode != 0:
            logger.warning("Source compilation failed for candidate %d, skipping", candidate_index)
            result_xml_path, empty_result = _get_empty_result(strategy, build_root, test_module)
            return empty_result, result_xml_path
        cache.mark_compiled(build_root, test_module, candidate_index)
    else:
        logger.debug("Step 1: Compiling tests + source (first time)")
        compile_result = strategy.compile_tests(build_root, run_env, test_module, timeout=120)
        if compile_result.returncode != 0:
            logger.warning("Compilation failed (rc=%d), skipping all candidates", compile_result.returncode)
            cache.mark_failed(build_root, test_module)
            result_xml_path, empty_result = _get_empty_result(strategy, build_root, test_module)
            return empty_result, result_xml_path
        cache.mark_compiled(build_root, test_module, candidate_index, tests=True)

    # Step 2: Get classpath (cached after first call)
    logger.debug("Step 2: Getting classpath")
    classpath = strategy.get_classpath(build_root, run_env, test_module, timeout=60)
    if not classpath:
        logger.warning("Failed to get classpath, falling back to %s-based execution", strategy.name)
        result = strategy.run_tests_via_build_tool(
            build_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module
        )
        reports_dir = strategy.get_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    # Step 3: Run tests directly via JVM
    working_dir = build_root / test_module if test_module else build_root
    reports_dir = strategy.get_reports_dir(build_root, test_module)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Step 3: Running %s tests directly (bypassing %s)", mode, strategy.name)
    result = _run_tests_direct(classpath, test_classes, run_env, working_dir, timeout=timeout, reports_dir=reports_dir)

    # Check for fallback indicators on failure
    if result.returncode != 0:
        combined_output = (result.stderr or "") + (result.stdout or "")
        fallback_indicators = [
            "ConsoleLauncher",
            "ClassNotFoundException",
            "No tests were executed",
            "Unable to locate a Java Runtime",
            "No tests found",
        ]
        if any(indicator in combined_output for indicator in fallback_indicators):
            logger.debug("Direct JVM execution failed, falling back to %s-based execution", strategy.name)
            result = strategy.run_tests_via_build_tool(
                build_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module
            )

    result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
    return result, result_xml_path


def _get_empty_result(strategy: Any, build_root: Path, test_module: str | None) -> tuple[Path, Any]:
    """Return an empty result for when no tests can be run."""
    reports_dir = strategy.get_reports_dir(build_root, test_module)
    result_xml_path = _get_combined_junit_xml(reports_dir, -1)

    empty_result = subprocess.CompletedProcess(
        args=["java", "-cp", "...", "ConsoleLauncher"], returncode=-1, stdout="", stderr="No test classes found"
    )
    return result_xml_path, empty_result


def _find_junit_console_standalone() -> Path | None:
    """Find the JUnit Platform Console Standalone JAR in the local Maven repository.

    This JAR contains ConsoleLauncher which is required for direct JVM test execution
    with JUnit 5.
    """
    m2_base = Path.home() / ".m2" / "repository" / "org" / "junit" / "platform" / "junit-platform-console-standalone"
    if not m2_base.exists():
        mvn = shutil.which("mvn")
        if mvn:
            logger.debug("Console standalone not found in cache, downloading via Maven")
            with contextlib.suppress(subprocess.TimeoutExpired, Exception):
                subprocess.run(
                    [
                        mvn,
                        "dependency:get",
                        "-Dartifact=org.junit.platform:junit-platform-console-standalone:1.10.0",
                        "-q",
                        "-B",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        if not m2_base.exists():
            return None

    try:
        versions = sorted(
            [d for d in m2_base.iterdir() if d.is_dir()],
            key=lambda d: tuple(int(x) for x in d.name.split(".") if x.isdigit()),
            reverse=True,
        )
        for version_dir in versions:
            jar = version_dir / f"junit-platform-console-standalone-{version_dir.name}.jar"
            if jar.exists():
                return jar
    except Exception:
        pass

    return None


def _run_tests_direct(
    classpath: str,
    test_classes: list[str],
    env: dict[str, str],
    working_dir: Path,
    timeout: int = 60,
    reports_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run JUnit tests directly using java command (bypassing build tool).

    This is much faster than build tool invocation (~500ms vs ~5-10s overhead).
    """
    from codeflash.languages.java.comparator import _find_java_executable

    java = _find_java_executable() or "java"

    has_junit5_tests = "junit-jupiter" in classpath
    has_console_launcher = "console-standalone" in classpath or "ConsoleLauncher" in classpath
    is_junit4 = not has_console_launcher
    if is_junit4:
        logger.debug("JUnit 4 project, no ConsoleLauncher available, using JUnitCore")
    elif has_junit5_tests:
        logger.debug("JUnit 5 project, using ConsoleLauncher")
    else:
        logger.debug("JUnit 4 project, using ConsoleLauncher (via vintage engine)")

    # Common --add-opens flags for Java 16+ module system
    add_opens = [
        "--add-opens",
        "java.base/java.util=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.lang=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.io=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.math=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.net=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.util.zip=ALL-UNNAMED",
    ]

    if is_junit4:
        if reports_dir:
            logger.debug(
                "JUnitCore does not support XML report generation; reports_dir=%s ignored. "
                "XML reports require ConsoleLauncher.",
                reports_dir,
            )
        cmd = [str(java), *add_opens, "-cp", classpath, "org.junit.runner.JUnitCore"]
        cmd.extend(test_classes)
    else:
        cmd = [
            str(java),
            *add_opens,
            "-cp",
            classpath,
            "org.junit.platform.console.ConsoleLauncher",
            "--disable-banner",
            "--disable-ansi-colors",
            "--details=none",
            "--config=junit.platform.output.capture.stdout=true",
            "--config=junit.platform.output.capture.stderr=true",
            f"--config=junit.jupiter.execution.timeout.default={max(timeout // 3, 10)}s",
        ]

        if reports_dir:
            reports_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--reports-dir", str(reports_dir)])

        for test_class in test_classes:
            cmd.extend(["--select-class", test_class])

    if is_junit4:
        logger.debug("Running tests directly: java -cp ... JUnitCore %s", test_classes)
    else:
        logger.debug("Running tests directly: java -cp ... ConsoleLauncher --select-class %s", test_classes)

    try:
        return _run_cmd_kill_pg_on_timeout(cmd, cwd=working_dir, env=env, timeout=timeout)
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


def _get_combined_junit_xml(surefire_dir: Path, candidate_index: int) -> Path:
    """Get or create a combined JUnit XML file from test reports."""
    result_id = next(_result_counter)
    result_xml_path = get_run_tmp_file(Path(f"java_results_{candidate_index}_{result_id}.xml"))

    if not surefire_dir.exists():
        _copy_empty_junit_template(result_xml_path)
        return result_xml_path

    xml_files = list(surefire_dir.glob("TEST-*.xml"))

    if not xml_files:
        _copy_empty_junit_template(result_xml_path)
        return result_xml_path

    if len(xml_files) == 1:
        shutil.copy(xml_files[0], result_xml_path)
        Path(xml_files[0]).unlink(missing_ok=True)
        return result_xml_path

    _combine_junit_xml_files(xml_files, result_xml_path)
    for xml_file in xml_files:
        Path(xml_file).unlink(missing_ok=True)
    return result_xml_path


def _copy_empty_junit_template(result_xml_path: Path) -> None:
    """Create a result file with empty JUnit XML content."""
    _write_empty_junit_xml(result_xml_path)


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

            all_testcases.extend(root.findall(".//testcase"))

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
                    logger.debug("_build_test_filter: Could not convert path to class name: %s", path)
            elif isinstance(path, str):
                filters.append(path)
        result = ",".join(filters) if filters else ""
        logger.debug("_build_test_filter (list/tuple): %s filters -> '%s'", len(filters), result)
        return result

    if hasattr(test_paths, "test_files"):
        filters = []
        skipped = 0
        skipped_reasons: list[str] = []

        for test_file in test_paths.test_files:
            if mode == "performance":
                if hasattr(test_file, "benchmarking_file_path") and test_file.benchmarking_file_path:
                    class_name = _path_to_class_name(test_file.benchmarking_file_path)
                    if class_name:
                        filters.append(class_name)
                    else:
                        reason = (
                            f"Could not convert benchmarking path to class name: {test_file.benchmarking_file_path}"
                        )
                        logger.debug("_build_test_filter: %s", reason)
                        skipped += 1
                        skipped_reasons.append(reason)
                else:
                    reason = f"TestFile has no benchmarking_file_path (original: {test_file.original_file_path})"
                    logger.warning("_build_test_filter: %s", reason)
                    skipped += 1
                    skipped_reasons.append(reason)
            elif hasattr(test_file, "instrumented_behavior_file_path") and test_file.instrumented_behavior_file_path:
                class_name = _path_to_class_name(test_file.instrumented_behavior_file_path)
                if class_name:
                    filters.append(class_name)
                else:
                    reason = (
                        f"Could not convert behavior path to class name: {test_file.instrumented_behavior_file_path}"
                    )
                    logger.debug("_build_test_filter: %s", reason)
                    skipped += 1
                    skipped_reasons.append(reason)
            else:
                reason = f"TestFile has no instrumented_behavior_file_path (original: {test_file.original_file_path})"
                logger.warning("_build_test_filter: %s", reason)
                skipped += 1
                skipped_reasons.append(reason)

        result = ",".join(filters) if filters else ""
        logger.debug("_build_test_filter (TestFiles): %s filters, %s skipped -> '%s'", len(filters), skipped, result)

        if not filters and skipped > 0:
            logger.error(
                "All %s test files were skipped in _build_test_filter! "
                "Mode: %s. This will cause an empty test filter. "
                "Reasons: %s",
                skipped,
                mode,
                skipped_reasons[:5],
            )

        return result

    logger.debug("_build_test_filter: Unknown test_paths type: %s", type(test_paths))
    return ""


def _path_to_class_name(path: Path, source_dirs: list[str] | None = None) -> str | None:
    """Convert a test file path to a Java class name."""
    if path.suffix != ".java":
        return None

    path_str = path.as_posix()
    parts = list(path.parts)

    if source_dirs:
        for src_dir in source_dirs:
            normalized = src_dir.rstrip("/")
            if normalized in path_str:
                idx = path_str.index(normalized) + len(normalized)
                remainder = path_str[idx:].lstrip("/")
                if remainder:
                    return remainder.replace("/", ".").removesuffix(".java")

    # Look for standard Maven/Gradle source directories
    java_idx = None
    for i, part in enumerate(parts):
        if part == "java" and i > 0 and parts[i - 1] in ("main", "test"):
            java_idx = i
            break

    if java_idx is None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "java":
                java_idx = i
                break

    if java_idx is not None:
        class_parts = parts[java_idx + 1 :]
        class_parts[-1] = class_parts[-1].replace(".java", "")
        return ".".join(class_parts)

    # For non-standard source directories, read the package declaration
    try:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("package "):
                    package = line[8:].rstrip(";").strip()
                    return f"{package}.{path.stem}"
                if line and not line.startswith("//") and not line.startswith("/*") and not line.startswith("*"):
                    break
    except Exception:
        pass

    return path.stem


def run_tests(test_files: list[Path], cwd: Path, env: dict[str, str], timeout: int) -> tuple[list[TestResult], Path]:
    """Run tests and return results."""
    from codeflash.languages.java.build_tool_strategy import get_strategy

    strategy = get_strategy(cwd)
    result = strategy.run_tests_via_build_tool(cwd, test_files, env, timeout, mode="behavior", test_module=None)

    reports_dir = strategy.get_reports_dir(cwd, None)
    test_results = parse_surefire_results(reports_dir)

    junit_files = list(reports_dir.glob("TEST-*.xml")) if reports_dir.exists() else []
    junit_path = junit_files[0] if junit_files else reports_dir / "test-results.xml"

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


def _extract_custom_source_dirs(project_root: Path) -> list[str]:
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


def get_test_run_command(project_root: Path, test_classes: list[str] | None = None) -> list[str]:
    """Get the command to run Java tests. Delegates to the appropriate build tool strategy."""
    from codeflash.languages.java.build_tool_strategy import get_strategy

    return get_strategy(project_root).get_test_run_command(project_root, test_classes)
