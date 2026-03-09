"""Java test runner for JUnit 5 with Maven and Gradle.

This module provides functionality to run JUnit 5 tests using Maven Surefire
or Gradle, supporting both behavioral testing and benchmarking modes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.languages.base import TestResult
from codeflash.languages.java.build_tools import (
    BuildTool,
    add_codeflash_dependency_to_pom,
    add_jacoco_plugin_to_pom,
    create_codeflash_gradle_init_script,
    detect_build_tool,
    extract_modules_from_settings_gradle,
    find_gradle_executable,
    find_maven_executable,
    get_gradle_test_reports_dir,
    get_jacoco_xml_path,
    get_jacoco_xml_path_gradle,
    install_codeflash_runtime,
    install_codeflash_runtime_to_m2,
    is_jacoco_configured,
)

_MAVEN_NS = "http://maven.apache.org/POM/4.0.0"

_M_MODULES_TAG = f"{{{_MAVEN_NS}}}modules"

logger = logging.getLogger(__name__)

# Cache for classpath strings — keyed on (maven_root, test_module).
# Dependencies don't change between candidates (only source code under test changes),
# so we avoid calling `mvn dependency:build-classpath` (~2-3s) repeatedly.
_classpath_cache: dict[tuple[Path, str | None], str] = {}

# Cache for multi-module dependency installs — keyed on (maven_root, test_module).
# After pre-installing deps to .m2 once, subsequent Maven invocations can skip -am.
_multimodule_deps_installed: set[tuple[Path, str]] = set()

# Regex pattern for valid Java class names (package.ClassName format)
# Allows: letters, digits, underscores, dots, and dollar signs (inner classes)
_VALID_JAVA_CLASS_NAME = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_$.]*$")

# Skip validation/analysis plugins that reject generated instrumented files
# (e.g. Apache Rat rejects missing license headers, Checkstyle rejects naming, etc.)
_MAVEN_VALIDATION_SKIP_FLAGS = [
    "-Drat.skip=true",
    "-Dcheckstyle.skip=true",
    "-Dspotbugs.skip=true",
    "-Dpmd.skip=true",
    "-Denforcer.skip=true",
    "-Djapicmp.skip=true",
]


class BuildToolStrategy(ABC):
    """Strategy interface for build-tool-specific operations (Maven vs Gradle)."""

    @abstractmethod
    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool: ...

    @abstractmethod
    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> None: ...

    @abstractmethod
    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def get_classpath_cached(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None: ...

    @abstractmethod
    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path: ...

    @abstractmethod
    def run_tests_fallback(
        self,
        build_root: Path,
        test_paths: Any,
        env: dict[str, str],
        timeout: int,
        mode: str,
        test_module: str | None,
        javaagent_arg: str | None = None,
    ) -> subprocess.CompletedProcess: ...

    @abstractmethod
    def run_benchmarking_fallback(
        self,
        test_paths: Any,
        test_env: dict[str, str],
        cwd: Path,
        timeout: int | None,
        project_root: Path | None,
        min_loops: int,
        max_loops: int,
        target_duration_seconds: float,
        inner_iterations: int,
    ) -> tuple[Path, Any]: ...

    @abstractmethod
    def run_tests_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess, Path]: ...

    @abstractmethod
    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None: ...

    @property
    @abstractmethod
    def default_cmd_name(self) -> str: ...


class MavenStrategy(BuildToolStrategy):
    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        return _ensure_codeflash_runtime(build_root, test_module)

    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> None:
        ensure_multi_module_deps_installed(build_root, test_module, env)

    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        return _compile_tests(build_root, env, test_module, timeout)

    def get_classpath_cached(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        return _get_test_classpath_cached(build_root, env, test_module, timeout)

    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        target_dir = _get_test_module_target_dir(build_root, test_module)
        return target_dir / "surefire-reports"

    def run_tests_fallback(
        self,
        build_root: Path,
        test_paths: Any,
        env: dict[str, str],
        timeout: int,
        mode: str,
        test_module: str | None,
        javaagent_arg: str | None = None,
    ) -> subprocess.CompletedProcess:
        return _run_maven_tests(
            build_root,
            test_paths,
            env,
            timeout=timeout,
            mode=mode,
            test_module=test_module,
            javaagent_arg=javaagent_arg,
        )

    def run_benchmarking_fallback(
        self,
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

    def run_tests_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess, Path]:
        result = _run_maven_tests(
            build_root,
            test_paths,
            run_env,
            timeout=timeout,
            mode="behavior",
            enable_coverage=True,
            test_module=test_module,
        )
        reports_dir = self.get_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        if test_module:
            test_module_pom = build_root / test_module / "pom.xml"
            if test_module_pom.exists():
                if not is_jacoco_configured(test_module_pom):
                    logger.info("Adding JaCoCo plugin to test module pom.xml: %s", test_module_pom)
                    add_jacoco_plugin_to_pom(test_module_pom)
                return get_jacoco_xml_path(build_root / test_module)
        else:
            pom_path = project_root / "pom.xml"
            if pom_path.exists():
                if not is_jacoco_configured(pom_path):
                    logger.info("Adding JaCoCo plugin to pom.xml for coverage collection")
                    add_jacoco_plugin_to_pom(pom_path)
                return get_jacoco_xml_path(project_root)
        return None

    @property
    def default_cmd_name(self) -> str:
        return "mvn"


class GradleStrategy(BuildToolStrategy):
    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        return _ensure_codeflash_runtime_gradle(build_root)

    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> None:
        pass  # Gradle handles dependencies via its own resolution

    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess:
        return _compile_tests_gradle(build_root, env, test_module, timeout)

    def get_classpath_cached(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        return _get_test_classpath_gradle_cached(build_root, env, test_module, timeout)

    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        return get_gradle_test_reports_dir(build_root, test_module)

    def run_tests_fallback(
        self,
        build_root: Path,
        test_paths: Any,
        env: dict[str, str],
        timeout: int,
        mode: str,
        test_module: str | None,
        javaagent_arg: str | None = None,
    ) -> subprocess.CompletedProcess:
        return _run_gradle_tests(build_root, test_paths, env, timeout=timeout, mode=mode, test_module=test_module)

    def run_benchmarking_fallback(
        self,
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
        return _run_benchmarking_tests_gradle(
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

    def run_tests_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess, Path]:
        return _run_gradle_tests_coverage(build_root, test_module, test_paths, run_env, timeout, candidate_index)

    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        return get_jacoco_xml_path_gradle(build_root, test_module)

    @property
    def default_cmd_name(self) -> str:
        return "gradle"


def _get_strategy(build_tool: BuildTool) -> BuildToolStrategy:
    if build_tool == BuildTool.GRADLE:
        return _GRADLE_STRATEGY
    return _MAVEN_STRATEGY


def _run_cmd_kill_pg_on_timeout(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess:
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

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Seconds to wait before killing the process group.
        text: If True, decode stdout/stderr as text.

    Returns:
        CompletedProcess.  On timeout, returncode is -2 and stderr contains a
        human-readable explanation.

    """
    if sys.platform == "win32":
        # Windows does not have POSIX process groups / killpg.  Fall back to
        # the standard subprocess.run() behaviour (kills parent only).
        try:
            return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=text, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2, stdout="", stderr=f"Process timed out after {timeout}s"
            )

    # POSIX path: start in its own process group so we can kill the tree.
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,  # puts proc in its own process group
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        # Kill the entire process group so Maven's forked Surefire JVMs don't
        # become orphans that keep the SQLite database locked.
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.kill()
        # Give processes a few seconds to shut down gracefully before SIGKILL.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
        # Drain pipes so we don't leave zombie pipe buffers.
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

    Args:
        class_name: The class name to validate (e.g., "com.example.MyTest").

    Returns:
        True if valid, False otherwise.

    """
    return bool(_VALID_JAVA_CLASS_NAME.match(class_name))


def _find_runtime_jar() -> Path | None:
    """Find the codeflash-runtime JAR file.

    Checks local Maven repo, package resources, and development build directory.
    """
    # Check local Maven repository first (fastest)
    m2_jar = (
        Path.home()
        / ".m2"
        / "repository"
        / "com"
        / "codeflash"
        / "codeflash-runtime"
        / "1.0.0"
        / "codeflash-runtime-1.0.0.jar"
    )
    if m2_jar.exists():
        return m2_jar

    # Check bundled JAR in package resources
    resources_jar = Path(__file__).parent / "resources" / "codeflash-runtime-1.0.0.jar"
    if resources_jar.exists():
        return resources_jar

    # Check development build directory
    dev_jar = (
        Path(__file__).parent.parent.parent.parent / "codeflash-java-runtime" / "target" / "codeflash-runtime-1.0.0.jar"
    )
    if dev_jar.exists():
        return dev_jar

    return None


def _ensure_codeflash_runtime(maven_root: Path, test_module: str | None) -> bool:
    """Ensure codeflash-runtime JAR is installed and added as a dependency.

    This must be called before running any Maven tests that use generated
    instrumented test code, since the generated tests import
    com.codeflash.CodeflashHelper from the codeflash-runtime JAR.

    Args:
        maven_root: Root directory of the Maven project.
        test_module: For multi-module projects, the test module name.

    Returns:
        True if runtime is available, False otherwise.

    """
    runtime_jar = _find_runtime_jar()
    if runtime_jar is None:
        logger.error("codeflash-runtime JAR not found. Generated tests will fail to compile.")
        return False

    # Install to local Maven repo if not already there
    m2_jar = (
        Path.home()
        / ".m2"
        / "repository"
        / "com"
        / "codeflash"
        / "codeflash-runtime"
        / "1.0.0"
        / "codeflash-runtime-1.0.0.jar"
    )
    if not m2_jar.exists():
        logger.info("Installing codeflash-runtime JAR to local Maven repository")
        if not install_codeflash_runtime(maven_root, runtime_jar):
            logger.error("Failed to install codeflash-runtime to local Maven repository")
            return False

    # Add dependency to the appropriate pom.xml
    if test_module:
        pom_path = maven_root / test_module / "pom.xml"
    else:
        pom_path = maven_root / "pom.xml"

    if pom_path.exists():
        if not add_codeflash_dependency_to_pom(pom_path):
            logger.error("Failed to add codeflash-runtime dependency to %s", pom_path)
            return False
    else:
        logger.warning("pom.xml not found at %s, cannot add codeflash-runtime dependency", pom_path)
        return False

    return True


def ensure_multi_module_deps_installed(maven_root: Path, test_module: str | None, env: dict[str, str]) -> bool:
    """Pre-install multi-module dependencies to the local Maven repository.

    In multi-module Maven projects (like Guava), Maven compiler plugin 3.15.0's
    JDK-8318913 workaround patches module-info.class timestamps after compilation.
    When a subsequent Maven invocation uses -am (also-make), the compiler detects
    "changed source code" and recompiles dependency modules — which fails because
    module-path resolution doesn't work in a partial reactor rebuild.

    This function runs `mvn install -DskipTests -pl <module> -am` once to put all
    dependency JARs into ~/.m2.  After that, test-running commands can use
    `-pl <module>` without `-am`, resolving deps from .m2 instead.

    Skipped for single-module projects (test_module is None) and cached so it only
    runs once per (maven_root, test_module) pair within a session.
    """
    if not test_module:
        return True

    cache_key = (maven_root, test_module)
    if cache_key in _multimodule_deps_installed:
        logger.debug("Multi-module deps already installed for %s:%s, skipping", maven_root, test_module)
        return True

    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found — cannot pre-install multi-module dependencies")
        return False

    cmd = [mvn, "install", "-DskipTests", "-B", "-pl", test_module, "-am"]
    cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

    logger.info("Pre-installing multi-module dependencies: %s (module: %s)", maven_root, test_module)
    logger.debug("Running: %s", " ".join(cmd))

    try:
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=maven_root, env=env, timeout=300)
        if result.returncode != 0:
            logger.error(
                "Failed to pre-install multi-module deps (exit %d).\nstdout: %s\nstderr: %s",
                result.returncode,
                result.stdout[-2000:] if result.stdout else "",
                result.stderr[-2000:] if result.stderr else "",
            )
            return False
    except Exception:
        logger.exception("Exception during multi-module dependency install")
        return False

    _multimodule_deps_installed.add(cache_key)
    logger.info("Multi-module dependencies installed successfully for %s:%s", maven_root, test_module)
    return True


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

    Args:
        test_filter: The test filter string (e.g., "MyTest", "MyTest,OtherTest", "My*Test").

    Returns:
        The sanitized test filter.

    Raises:
        ValueError: If the test filter contains invalid characters.

    """
    # Split by comma for multiple test patterns
    patterns = [p.strip() for p in test_filter.split(",")]

    for pattern in patterns:
        # Remove wildcards for validation (they're allowed in test filters)
        name_to_validate = pattern.replace("*", "A")  # Replace * with a valid char

        if not _validate_java_class_name(name_to_validate):
            msg = (
                f"Invalid test class name or pattern: '{pattern}'. "
                f"Test names must follow Java identifier rules (letters, digits, underscores, dots, dollar signs)."
            )
            raise ValueError(msg)

    return test_filter


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
        # and the test file is in a submodule
        modules = _get_project_modules(project_root)
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

        # For Gradle projects, walk up to find the actual root with settings.gradle/gradlew
        current = project_root.parent
        while current != current.parent:
            settings_gradle = current / "settings.gradle.kts"
            if not settings_gradle.exists():
                settings_gradle = current / "settings.gradle"
            if settings_gradle.exists():
                try:
                    module_name = project_root.relative_to(current).parts[0]
                    logger.debug("Detected Gradle multi-module project. Root: %s, Module: %s", current, module_name)
                    return current, module_name
                except (ValueError, IndexError):
                    pass
            current = current.parent

        return project_root, None

    # Find common parent that contains both project_root and test files
    current = project_root.parent
    while current != current.parent:
        modules = _get_project_modules(current)
        if modules:
            if test_dir:
                try:
                    test_module = test_dir.relative_to(current)
                    test_module_name = test_module.parts[0] if test_module.parts else None
                    logger.debug("Detected multi-module project. Root: %s, Test module: %s", current, test_module_name)
                    return current, test_module_name
                except ValueError:
                    pass
        current = current.parent

    return project_root, None


def _get_project_modules(project_root: Path) -> list[str]:
    """Get submodule names from Maven pom.xml or Gradle settings."""
    # Try Maven first
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            content = pom_path.read_text(encoding="utf-8")
            if "<modules>" in content:
                return _extract_modules_from_pom_content(content)
        except Exception:
            pass

    # Try Gradle settings
    gradle_modules = extract_modules_from_settings_gradle(project_root)
    if gradle_modules:
        return gradle_modules

    return []


def _get_test_module_target_dir(build_root: Path, test_module: str | None) -> Path:
    """Get the target/build output directory for the test module.

    Returns the correct directory based on build tool (Maven=target, Gradle=build).
    """
    build_tool = detect_build_tool(build_root)
    if build_tool == BuildTool.GRADLE:
        base = build_root / test_module if test_module else build_root
        return base / "build"
    # Maven
    if test_module:
        return build_root / test_module / "target"
    return build_root / "target"


def _get_test_reports_dir(build_root: Path, test_module: str | None) -> Path:
    """Get the directory containing test XML reports (Surefire or Gradle test-results)."""
    build_tool = detect_build_tool(build_root)
    if build_tool == BuildTool.GRADLE:
        return get_gradle_test_reports_dir(build_root, test_module)
    # Maven surefire reports
    target_dir = _get_test_module_target_dir(build_root, test_module)
    return target_dir / "surefire-reports"


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
    build_tool = detect_build_tool(project_root)
    strategy = _get_strategy(build_tool)

    # Detect multi-module projects where tests are in a different module
    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    # Ensure codeflash-runtime is installed and multi-module deps are available
    strategy.ensure_runtime(build_root, test_module)
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

    # Configure coverage (JaCoCo)
    coverage_xml_path: Path | None = None
    if enable_coverage:
        coverage_xml_path = strategy.setup_coverage(build_root, test_module, project_root)

    min_timeout = 300 if enable_coverage else 60
    effective_timeout = max(timeout or 300, min_timeout)

    if enable_coverage:
        result, result_xml_path = strategy.run_tests_coverage(
            build_root, test_module, test_paths, run_env, effective_timeout, candidate_index
        )
    else:
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

    if enable_coverage:
        logger.info("Build verify completed with return code: %s", result.returncode)
        if result.returncode != 0:
            logger.warning("Verify had non-zero return code: %s. Coverage data may be incomplete.", result.returncode)

    if enable_coverage and coverage_xml_path:
        target_dir = _get_test_module_target_dir(build_root, test_module)
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


def _compile_tests(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 120
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
        return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

    cmd = [mvn, "test-compile", "-e", "-B"]  # Show errors but not verbose output; -B for batch mode (no ANSI colors)
    cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

    if test_module:
        cmd.extend(["-pl", test_module])

    logger.debug("Compiling tests: %s in %s", " ".join(cmd), project_root)

    try:
        return _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)
    except Exception as e:
        logger.exception("Maven compilation failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))


def _get_test_classpath(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 60
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

    cmd = [mvn, "dependency:build-classpath", "-DincludeScope=test", f"-Dmdep.outputFile={cp_file}", "-q", "-B"]

    if test_module:
        cmd.extend(["-pl", test_module])

    logger.debug("Getting classpath: %s", " ".join(cmd))

    try:
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)

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

        # For multi-module projects, also include target/classes from all modules
        # This is needed because the test module may depend on other modules
        if test_module:
            # Find all target/classes directories in sibling modules
            for module_dir in project_root.iterdir():
                if module_dir.is_dir() and module_dir.name != test_module:
                    module_classes = module_dir / "target" / "classes"
                    if module_classes.exists():
                        logger.debug("Adding multi-module classpath: %s", module_classes)
                        cp_parts.append(str(module_classes))

        # Add JUnit Platform Console Standalone JAR if not already on classpath.
        # This is required for direct JVM execution with ConsoleLauncher,
        # which is NOT included in the standard junit-jupiter dependency tree.
        if "console-standalone" not in classpath and "ConsoleLauncher" not in classpath:
            console_jar = _find_junit_console_standalone()
            if console_jar:
                logger.debug("Adding JUnit Console Standalone to classpath: %s", console_jar)
                cp_parts.append(str(console_jar))

        return os.pathsep.join(cp_parts)

    except Exception as e:
        logger.exception("Failed to get classpath: %s", e)
        return None
    finally:
        # Clean up temp file
        if cp_file.exists():
            cp_file.unlink()


def _get_test_classpath_cached(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 60
) -> str | None:
    key = (project_root, test_module)
    cached = _classpath_cache.get(key)
    if cached is not None:
        logger.debug("Using cached classpath for (%s, %s)", project_root, test_module)
        return cached
    result = _get_test_classpath(project_root, env, test_module, timeout)
    if result is not None:
        _classpath_cache[key] = result
    return result


def _find_junit_console_standalone() -> Path | None:
    """Find the JUnit Platform Console Standalone JAR in the local Maven repository.

    This JAR contains ConsoleLauncher which is required for direct JVM test execution
    with JUnit 5. It is NOT included in the standard junit-jupiter dependency tree.

    Returns:
        Path to the console standalone JAR, or None if not found.

    """
    m2_base = Path.home() / ".m2" / "repository" / "org" / "junit" / "platform" / "junit-platform-console-standalone"
    if not m2_base.exists():
        # Try to download it via Maven
        mvn = find_maven_executable()
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

    # Find the latest version available
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
    javaagent_arg: str | None = None,
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
    # Find java executable (reuse comparator's robust finder for macOS compatibility)
    from codeflash.languages.java.comparator import _find_java_executable

    java = _find_java_executable() or "java"

    # Detect JUnit version from the classpath string.
    # We check for junit-jupiter (the JUnit 5 test API) as the indicator of JUnit 5 tests.
    # Note: console-standalone and junit-platform are NOT reliable indicators because
    # we inject console-standalone ourselves in _get_test_classpath(), so it's always present.
    # ConsoleLauncher can run both JUnit 5 and JUnit 4 tests (via vintage engine),
    # so we prefer it when available and only fall back to JUnitCore for pure JUnit 4
    # projects without ConsoleLauncher on the classpath.
    has_junit5_tests = "junit-jupiter" in classpath
    has_console_launcher = "console-standalone" in classpath or "ConsoleLauncher" in classpath
    # Use ConsoleLauncher if available (works for both JUnit 4 via vintage and JUnit 5).
    # Only use JUnitCore when ConsoleLauncher is not on the classpath at all.
    is_junit4 = not has_console_launcher
    if is_junit4:
        logger.debug("JUnit 4 project, no ConsoleLauncher available, using JUnitCore")
    elif has_junit5_tests:
        logger.debug("JUnit 5 project, using ConsoleLauncher")
    else:
        logger.debug("JUnit 4 project, using ConsoleLauncher (via vintage engine)")

    # Collect extra JVM flags (e.g. -javaagent for line profiler)
    extra_jvm_flags: list[str] = []
    if javaagent_arg:
        extra_jvm_flags.append(javaagent_arg)

    if is_junit4:
        if reports_dir:
            logger.debug(
                "JUnitCore does not support XML report generation; reports_dir=%s ignored. "
                "XML reports require ConsoleLauncher.",
                reports_dir,
            )
        # Use JUnit 4's JUnitCore runner
        cmd = [
            str(java),
            *extra_jvm_flags,
            # Java 16+ module system: Kryo needs reflective access to internal JDK classes
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
            "-cp",
            classpath,
            "org.junit.runner.JUnitCore",
        ]
        # Add test classes
        cmd.extend(test_classes)
    else:
        # Build command using JUnit Platform Console Launcher (JUnit 5)
        # The launcher is included in junit-platform-console-standalone or junit-jupiter
        cmd = [
            str(java),
            *extra_jvm_flags,
            # Java 16+ module system: Kryo needs reflective access to internal JDK classes
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
        args=["java", "-cp", "...", "ConsoleLauncher"], returncode=-1, stdout="", stderr="No test classes found"
    )
    return result_xml_path, empty_result


def _run_direct_or_fallback(
    strategy: BuildToolStrategy,
    build_root: Path,
    test_module: str | None,
    test_paths: Any,
    run_env: dict[str, str],
    timeout: int,
    mode: str,
    candidate_index: int = -1,
    javaagent_arg: str | None = None,
) -> tuple[subprocess.CompletedProcess, Path]:
    """Compile once, then run tests directly via JVM. Falls back to build tool on failure."""
    test_classes = _get_test_class_names(test_paths, mode=mode)
    if not test_classes:
        logger.warning("No test classes found for mode=%s, returning empty result", mode)
        result_xml_path, empty_result = _get_empty_result(build_root, test_module)
        return empty_result, result_xml_path

    # Step 1: Compile tests
    logger.debug("Step 1: Compiling tests for %s mode", mode)
    compile_result = strategy.compile_tests(build_root, run_env, test_module, timeout=120)
    if compile_result.returncode != 0:
        logger.warning("Compilation failed (rc=%d), falling back to build-tool execution", compile_result.returncode)
        result = strategy.run_tests_fallback(build_root, test_paths, run_env, timeout, mode, test_module)
        reports_dir = strategy.get_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    # Step 2: Get classpath
    logger.debug("Step 2: Getting classpath")
    classpath = strategy.get_classpath_cached(build_root, run_env, test_module, timeout=60)
    if not classpath:
        logger.warning("Failed to get classpath, falling back to build-tool execution")
        result = strategy.run_tests_fallback(build_root, test_paths, run_env, timeout, mode, test_module)
        reports_dir = strategy.get_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    # Step 3: Run tests directly via JVM
    working_dir = build_root / test_module if test_module else build_root
    reports_dir = strategy.get_reports_dir(build_root, test_module)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Step 3: Running %s tests directly (bypassing build tool)", mode)
    result = _run_tests_direct(
        classpath,
        test_classes,
        run_env,
        working_dir,
        timeout=timeout,
        reports_dir=reports_dir,
        javaagent_arg=javaagent_arg,
    )

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
            logger.debug("Direct JVM execution failed, falling back to build-tool execution")
            result = strategy.run_tests_fallback(build_root, test_paths, run_env, timeout, mode, test_module)

    result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
    return result, result_xml_path


def _run_direct_or_fallback_maven(
    maven_root: Path,
    test_module: str | None,
    test_paths: Any,
    run_env: dict[str, str],
    timeout: int,
    mode: str,
    candidate_index: int = -1,
) -> tuple[subprocess.CompletedProcess, Path]:
    """Compile once, then run tests directly via JVM. Falls back to Maven on failure.

    This mirrors the compile-once-run-many pattern from run_benchmarking_tests but
    for single-run modes (behavioral without coverage, line-profile).
    """
    test_classes = _get_test_class_names(test_paths, mode=mode)
    if not test_classes:
        logger.warning("No test classes found for mode=%s, returning empty result", mode)
        result_xml_path, empty_result = _get_empty_result(maven_root, test_module)
        return empty_result, result_xml_path

    # Step 1: Compile tests (still Maven — needed for dependency resolution)
    logger.debug("Step 1: Compiling tests for %s mode", mode)
    compile_result = _compile_tests(maven_root, run_env, test_module, timeout=120)
    if compile_result.returncode != 0:
        logger.warning("Compilation failed (rc=%d), falling back to Maven-based execution", compile_result.returncode)
        result = _run_maven_tests(maven_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module)
        target_dir = _get_test_module_target_dir(maven_root, test_module)
        surefire_dir = target_dir / "surefire-reports"
        result_xml_path = _get_combined_junit_xml(surefire_dir, candidate_index)
        return result, result_xml_path

    # Step 2: Get classpath (cached after first call)
    logger.debug("Step 2: Getting classpath")
    classpath = _get_test_classpath_cached(maven_root, run_env, test_module, timeout=60)
    if not classpath:
        logger.warning("Failed to get classpath, falling back to Maven-based execution")
        result = _run_maven_tests(maven_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module)
        target_dir = _get_test_module_target_dir(maven_root, test_module)
        surefire_dir = target_dir / "surefire-reports"
        result_xml_path = _get_combined_junit_xml(surefire_dir, candidate_index)
        return result, result_xml_path

    # Step 3: Run tests directly via JVM
    working_dir = maven_root / test_module if test_module else maven_root
    target_dir = _get_test_module_target_dir(maven_root, test_module)
    reports_dir = target_dir / "surefire-reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Step 3: Running %s tests directly (bypassing Maven)", mode)
    result = _run_tests_direct(classpath, test_classes, run_env, working_dir, timeout=timeout, reports_dir=reports_dir)

    # Check for fallback indicators on failure (same checks as benchmarking)
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
            logger.debug("Direct JVM execution failed, falling back to Maven-based execution")
            result = _run_maven_tests(
                maven_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module
            )

    result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
    return result, result_xml_path


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

    per_loop_timeout = max(timeout or 0, 120, 60 + inner_iterations)

    logger.debug("Using Maven-based benchmarking (fallback mode)")

    for loop_idx in range(1, max_loops + 1):
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        if "CODEFLASH_INNER_ITERATIONS" not in run_env:
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

        # Check if we have timing markers even if some tests failed
        # We should continue looping if we're getting valid timing data
        if result.returncode != 0:
            import re

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
    build_tool = detect_build_tool(project_root)
    strategy = _get_strategy(build_tool)

    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    strategy.ensure_runtime(build_root, test_module)
    base_env = os.environ.copy()
    base_env.update(test_env)
    strategy.install_multi_module_deps(build_root, test_module, base_env)

    test_classes = _get_test_class_names(test_paths, mode="performance")
    if not test_classes:
        logger.error("No test classes found")
        return _get_empty_result(build_root, test_module)

    # Step 1: Compile tests once
    compile_env = os.environ.copy()
    compile_env.update(test_env)

    logger.debug("Step 1: Compiling tests (one-time build overhead)")
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
        logger.warning("Falling back to build-tool-based test execution")
        return strategy.run_benchmarking_fallback(
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

    logger.debug("Compilation completed in %.2fs", compile_time)

    # Step 2: Get classpath
    logger.debug("Step 2: Getting classpath")
    classpath = strategy.get_classpath_cached(build_root, compile_env, test_module, timeout=60)

    if not classpath:
        logger.warning("Failed to get classpath, falling back to build-tool execution")
        return strategy.run_benchmarking_fallback(
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
    logger.debug("Step 3: Running tests directly (bypassing build tool)")

    all_stdout = []
    all_stderr = []
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
                import re as _re

                has_markers = bool(_re.search(r"!######", result.stdout or ""))
                if not has_markers and result.returncode != 0:
                    should_fallback = True
                    logger.debug("Direct execution failed with no timing markers, likely JUnit version mismatch")

        if should_fallback:
            logger.debug("Direct JVM execution failed, falling back to build-tool execution")
            return strategy.run_benchmarking_fallback(
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
        args=last_result.args if last_result else [strategy.default_cmd_name, "test"],
        returncode=last_result.returncode if last_result else -1,
        stdout=combined_stdout,
        stderr=combined_stderr,
    )

    reports_dir_final = strategy.get_reports_dir(build_root, test_module)
    result_xml_path = _get_combined_junit_xml(reports_dir_final, -1)

    return result_xml_path, combined_result


def _get_combined_junit_xml(surefire_dir: Path, candidate_index: int) -> Path:
    """Get or create a combined JUnit XML file from test reports (Surefire or Gradle).

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
        Path(xml_files[0]).unlink(missing_ok=True)
        return result_xml_path

    # Combine multiple XML files into one
    _combine_junit_xml_files(xml_files, result_xml_path)
    for xml_file in xml_files:
        Path(xml_file).unlink(missing_ok=True)
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
            all_testcases.extend(root.findall(".//testcase"))

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
    javaagent_arg: str | None = None,
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
        return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

    # Build test filter
    test_filter = _build_test_filter(test_paths, mode=mode)
    logger.debug("Built test filter for mode=%s: '%s' (empty=%s)", mode, test_filter, not test_filter)
    logger.debug("test_paths type: %s, has test_files: %s", type(test_paths), hasattr(test_paths, "test_files"))
    if hasattr(test_paths, "test_files"):
        logger.debug("Number of test files: %s", len(test_paths.test_files))
        for i, tf in enumerate(test_paths.test_files[:3]):  # Log first 3
            logger.debug(
                "  TestFile[%s]: behavior=%s, bench=%s",
                i,
                tf.instrumented_behavior_file_path,
                tf.benchmarking_file_path,
            )

    # Build Maven command
    # When coverage is enabled, use 'verify' phase to ensure JaCoCo report runs after tests
    # JaCoCo's report goal is bound to the verify phase to get post-test execution data
    maven_goal = "verify" if enable_coverage else "test"
    cmd = [mvn, maven_goal, "-fae", "-B"]  # Fail at end to run all tests; -B for batch mode (no ANSI colors)
    cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

    # Add --add-opens flags for Java 16+ module system compatibility.
    # The codeflash-runtime Serializer uses Kryo which needs reflective access to
    # java.base internals for serializing test inputs/outputs to SQLite.
    # These flags are safe no-ops on older Java versions.
    # Note: This overrides JaCoCo's argLine for the forked JVM, but JaCoCo coverage
    # is handled separately via enable_coverage and the verify phase.
    add_opens_flags = (
        "--add-opens java.base/java.util=ALL-UNNAMED"
        " --add-opens java.base/java.lang=ALL-UNNAMED"
        " --add-opens java.base/java.lang.reflect=ALL-UNNAMED"
        " --add-opens java.base/java.io=ALL-UNNAMED"
        " --add-opens java.base/java.math=ALL-UNNAMED"
        " --add-opens java.base/java.net=ALL-UNNAMED"
        " --add-opens java.base/java.util.zip=ALL-UNNAMED"
    )
    if javaagent_arg:
        cmd.append(f"-DargLine={javaagent_arg} {add_opens_flags}")
    else:
        cmd.append(f"-DargLine={add_opens_flags}")

    # For performance mode, disable Surefire's file-based output redirection.
    # By default, Surefire captures System.out.println() to .txt report files,
    # which prevents timing markers from appearing in Maven's stdout.
    if mode == "performance":
        cmd.append("-Dsurefire.useFile=false")

    # When coverage is enabled, continue build even if tests fail so JaCoCo report is generated
    if enable_coverage:
        cmd.append("-Dmaven.test.failure.ignore=true")

    # For multi-module projects, specify which module to test.
    # Dependencies are pre-installed to .m2 by ensure_multi_module_deps_installed(),
    # so we use -pl without -am to avoid recompiling dependency modules (which fails
    # on projects like Guava due to Maven compiler plugin's JDK-8318913 workaround).
    if test_module:
        cmd.extend(
            [
                "-pl",
                test_module,
                "-DfailIfNoTests=false",
                "-Dsurefire.failIfNoSpecifiedTests=false",
                "-DskipTests=false",
            ]
        )

    if test_filter:
        # Validate test filter to prevent command injection
        validated_filter = _validate_test_filter(test_filter)
        cmd.append(f"-Dtest={validated_filter}")
        logger.debug("Added -Dtest=%s to Maven command", validated_filter)
    else:
        # CRITICAL: Empty test filter means Maven will run ALL tests
        # This is almost always a bug - tests should be filtered to relevant ones
        error_msg = (
            f"Test filter is EMPTY for mode={mode}! "
            f"Maven will run ALL tests instead of the specified tests. "
            f"This indicates a problem with test file instrumentation or path resolution."
        )
        logger.error(error_msg)
        # Raise exception to prevent running all tests unintentionally
        # This helps catch bugs early rather than silently running wrong tests
        raise ValueError(error_msg)

    logger.debug("Running Maven command: %s in %s", " ".join(cmd), project_root)

    try:
        # Use _run_cmd_kill_pg_on_timeout instead of subprocess.run so that on
        # timeout we kill the entire Maven process GROUP (including forked Surefire
        # JVMs).  With plain subprocess.run(), only the Maven parent is killed and
        # the child JVMs become orphaned, holding the SQLite result file open and
        # causing "database is locked" errors when Python reads the file immediately
        # after Maven is killed.
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)

        # Check if Maven failed due to compilation errors (not just test failures)
        if result.returncode != 0:
            # Maven compilation errors contain specific markers in output
            compilation_error_indicators = [
                "[ERROR] COMPILATION ERROR",
                "[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin",
                "compilation failure",
                "cannot find symbol",
                "package .* does not exist",
            ]

            combined_output = (result.stdout or "") + (result.stderr or "")
            has_compilation_error = any(
                indicator.lower() in combined_output.lower() for indicator in compilation_error_indicators
            )

            if has_compilation_error:
                logger.error(
                    "Maven compilation failed for %s tests. "
                    "Check that generated test code is syntactically valid Java. "
                    "Return code: %s",
                    mode,
                    result.returncode,
                )
                # Log first 50 lines of output to help diagnose compilation errors
                output_lines = combined_output.split("\n")
                error_context = "\n".join(output_lines[:50]) if len(output_lines) > 50 else combined_output
                logger.error("Maven compilation error output:\n%s", error_context)

        return result

    except Exception as e:
        logger.exception("Maven test execution failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))


def _build_test_filter(test_paths: Any, mode: str = "behavior") -> str:
    """Build a Maven Surefire test filter from test paths.

    Args:
        test_paths: Test files, classes, or methods to include.
        mode: Testing mode - "behavior" or "performance".

    Returns:
        Surefire test filter string.

    """
    if not test_paths:
        logger.debug("_build_test_filter: test_paths is empty/None")
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
                else:
                    logger.debug("_build_test_filter: Could not convert path to class name: %s", path)
            elif isinstance(path, str):
                filters.append(path)
        result = ",".join(filters) if filters else ""
        logger.debug("_build_test_filter (list/tuple): %s filters -> '%s'", len(filters), result)
        return result

    # Handle TestFiles object (has test_files attribute)
    if hasattr(test_paths, "test_files"):
        filters = []
        skipped = 0
        skipped_reasons = []

        for test_file in test_paths.test_files:
            # For performance mode, use benchmarking_file_path
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
            # For behavior mode, use instrumented_behavior_file_path
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

        # If all tests were skipped, log detailed information to help diagnose
        if not filters and skipped > 0:
            logger.error(
                "All %s test files were skipped in _build_test_filter! "
                "Mode: %s. This will cause an empty test filter. "
                "Reasons: %s",  # Show first 5 reasons
                skipped,
                mode,
                skipped_reasons[:5],
            )

        return result

    logger.debug("_build_test_filter: Unknown test_paths type: %s", type(test_paths))
    return ""


def _path_to_class_name(path: Path, source_dirs: list[str] | None = None) -> str | None:
    """Convert a test file path to a Java class name.

    Args:
        path: Path to the test file.
        source_dirs: Optional list of custom source directory prefixes
            (e.g., ["src/main/custom", "app/java"]).

    Returns:
        Fully qualified class name, or None if unable to determine.

    """
    if path.suffix != ".java":
        return None

    path_str = path.as_posix()
    parts = list(path.parts)

    # Try custom source directories first
    if source_dirs:
        for src_dir in source_dirs:
            normalized = src_dir.rstrip("/")
            # Check if the path contains this source directory
            if normalized in path_str:
                idx = path_str.index(normalized) + len(normalized)
                remainder = path_str[idx:].lstrip("/")
                if remainder:
                    return remainder.replace("/", ".").removesuffix(".java")

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
        class_parts = parts[java_idx + 1 :]
        # Remove .java extension from last part
        class_parts[-1] = class_parts[-1].replace(".java", "")
        return ".".join(class_parts)

    # For non-standard source directories (e.g., test/src/com/...),
    # read the package declaration from the Java file itself
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

    # Fallback: just use the file name
    return path.stem


def run_tests(test_files: list[Path], cwd: Path, env: dict[str, str], timeout: int) -> tuple[list[TestResult], Path]:
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

    Args:
        test_paths: TestFiles object or list of test file paths.
        test_env: Environment variables for the test run.
        cwd: Working directory for running tests.
        timeout: Optional timeout in seconds.
        project_root: Project root directory.
        line_profile_output_file: Path where profiling results will be written.
        javaagent_arg: Optional -javaagent:... JVM argument for the profiler agent.

    Returns:
        Tuple of (result_file_path, subprocess_result).

    """
    project_root = project_root or cwd
    build_tool = detect_build_tool(project_root)
    strategy = _get_strategy(build_tool)

    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    strategy.ensure_runtime(build_root, test_module)
    base_env = os.environ.copy()
    base_env.update(test_env)
    strategy.install_multi_module_deps(build_root, test_module, base_env)

    run_env = os.environ.copy()
    run_env.update(test_env)
    run_env["CODEFLASH_MODE"] = "line_profile"
    if line_profile_output_file:
        run_env["CODEFLASH_LINE_PROFILE_OUTPUT"] = str(line_profile_output_file)

    min_timeout = 120
    effective_timeout = max(timeout or min_timeout, min_timeout)
    logger.debug("Running line profiling tests (single run) with timeout=%ds", effective_timeout)

    result, result_xml_path = _run_direct_or_fallback(
        strategy,
        build_root,
        test_module,
        test_paths,
        run_env,
        effective_timeout,
        mode="line_profile",
        candidate_index=-1,
        javaagent_arg=javaagent_arg,
    )

    return result_xml_path, result


## ---- Gradle-specific functions ----


def _ensure_codeflash_runtime_gradle(build_root: Path) -> bool:
    """Ensure codeflash-runtime JAR is installed for Gradle (via mavenLocal)."""
    runtime_jar = _find_runtime_jar()
    if runtime_jar is None:
        logger.error("codeflash-runtime JAR not found. Generated tests will fail to compile.")
        return False

    return install_codeflash_runtime_to_m2(runtime_jar)


def _delete_broken_generated_test_files(
    compile_result: subprocess.CompletedProcess, project_root: Path, test_module: str | None, *, sweep_all: bool = True
) -> int:
    """Delete codeflash-generated test files that caused compilation errors.

    When compileTestJava fails, generated test files with bad imports/syntax poison the entire
    module build. This extracts failing file paths from the error output and deletes only
    codeflash-generated ones (matching __perfinstrumented/__perfonlyinstrumented patterns).

    Args:
        compile_result: The failed compilation result.
        project_root: Root directory of the project.
        test_module: For multi-module projects, the module containing tests.
        sweep_all: If True, also scan the test source dir for any leftover generated files
            not mentioned in the error output. Set to False when the caller will retry the
            same command with a --tests filter that references still-needed generated files.

    Returns the number of files deleted.

    """
    import re

    output = (compile_result.stdout or "") + (compile_result.stderr or "")
    # Match file paths from javac error output like:
    # /path/to/ofTest__perfinstrumented.java:10: error: cannot find symbol
    error_file_pattern = re.compile(r"(/\S+__perf(?:only)?instrumented\S*\.java):\d+:")
    files_to_delete = set()
    for match in error_file_pattern.finditer(output):
        file_path = Path(match.group(1))
        if file_path.exists():
            files_to_delete.add(file_path)

    # Optionally scan test source dir for any leftover generated files not in error output.
    # Only safe when the caller does NOT need these files for a subsequent --tests filter.
    if sweep_all:
        if test_module:
            test_src_dir = project_root / test_module / "src" / "test" / "java"
        else:
            test_src_dir = project_root / "src" / "test" / "java"

        if test_src_dir.exists():
            generated_pattern = re.compile(r".*__perf(?:only)?instrumented(?:_\d+)?\.java$")
            for f in test_src_dir.rglob("*.java"):
                if generated_pattern.match(f.name):
                    files_to_delete.add(f)

    for f in files_to_delete:
        logger.debug("Deleting broken generated test file: %s", f)
        f.unlink(missing_ok=True)

    return len(files_to_delete)


def _compile_tests_gradle(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Compile test code using Gradle."""
    gradle = find_gradle_executable(project_root)
    if not gradle:
        logger.error("Gradle not found")
        return subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found")

    init_script = create_codeflash_gradle_init_script(target_module=test_module)
    if not init_script:
        return subprocess.CompletedProcess(
            args=["gradle"], returncode=-1, stdout="", stderr="Failed to create init script"
        )

    task = f":{test_module}:testClasses" if test_module else "testClasses"
    cmd = [gradle, task, "--no-daemon", "-q", "--init-script", str(init_script)]

    logger.debug("Compiling tests (Gradle): %s in %s", " ".join(cmd), project_root)

    try:
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)
        if result.returncode != 0:
            deleted = _delete_broken_generated_test_files(result, project_root, test_module)
            if deleted:
                logger.info("Deleted %d broken generated test file(s), retrying compilation", deleted)
                result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)
        return result
    except Exception as e:
        logger.exception("Gradle compilation failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))
    finally:
        init_script.unlink(missing_ok=True)


def _get_test_classpath_gradle(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 60
) -> str | None:
    """Get the test classpath from Gradle using init script."""
    from codeflash.languages.java.build_tools import _get_gradle_classpath

    classpath = _get_gradle_classpath(project_root, test_module)
    if not classpath:
        return None

    # Append compiled class dirs and JUnit console standalone
    cp_parts = [classpath]

    if test_module:
        module_path = project_root / test_module
    else:
        module_path = project_root

    test_classes_dir = module_path / "build" / "classes" / "java" / "test"
    main_classes_dir = module_path / "build" / "classes" / "java" / "main"

    if test_classes_dir.exists():
        cp_parts.append(str(test_classes_dir))
    if main_classes_dir.exists():
        cp_parts.append(str(main_classes_dir))

    # For multi-module, include sibling module classes
    if test_module:
        for module_dir in project_root.iterdir():
            if module_dir.is_dir() and module_dir.name != test_module:
                module_classes = module_dir / "build" / "classes" / "java" / "main"
                if module_classes.exists():
                    cp_parts.append(str(module_classes))

    if "console-standalone" not in classpath and "ConsoleLauncher" not in classpath:
        console_jar = _find_junit_console_standalone()
        if console_jar:
            cp_parts.append(str(console_jar))

    return os.pathsep.join(cp_parts)


def _get_test_classpath_gradle_cached(
    project_root: Path, env: dict[str, str], test_module: str | None = None, timeout: int = 60
) -> str | None:
    key = (project_root, test_module)
    cached = _classpath_cache.get(key)
    if cached is not None:
        logger.debug("Using cached Gradle classpath for (%s, %s)", project_root, test_module)
        return cached
    result = _get_test_classpath_gradle(project_root, env, test_module, timeout)
    if result is not None:
        _classpath_cache[key] = result
    return result


def _run_gradle_tests(
    project_root: Path,
    test_paths: Any,
    env: dict[str, str],
    timeout: int = 300,
    mode: str = "behavior",
    test_module: str | None = None,
) -> subprocess.CompletedProcess:
    """Run Gradle tests."""
    gradle = find_gradle_executable(project_root)
    if not gradle:
        logger.error("Gradle not found")
        return subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found")

    init_script = create_codeflash_gradle_init_script(target_module=test_module)
    if not init_script:
        return subprocess.CompletedProcess(
            args=["gradle"], returncode=-1, stdout="", stderr="Failed to create init script"
        )

    test_filter = _build_test_filter(test_paths, mode=mode)
    task = f":{test_module}:test" if test_module else "test"
    cmd = [gradle, task, "--no-daemon", "--init-script", str(init_script)]

    if test_filter:
        # Gradle uses --tests filter; each comma-separated class becomes a separate --tests
        for cls in test_filter.split(","):
            cls = cls.strip()
            if cls:
                cmd.extend(["--tests", cls])

    logger.debug("Running Gradle command: %s in %s", " ".join(cmd), project_root)

    try:
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)
        if result.returncode != 0 and "compileTestJava" in (result.stdout or "") + (result.stderr or ""):
            deleted = _delete_broken_generated_test_files(result, project_root, test_module, sweep_all=False)
            if deleted:
                logger.info("Deleted %d broken generated test file(s), retrying Gradle test", deleted)
                result = _run_cmd_kill_pg_on_timeout(cmd, cwd=project_root, env=env, timeout=timeout)
        return result
    except Exception as e:
        logger.exception("Gradle test execution failed: %s", e)
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))
    finally:
        init_script.unlink(missing_ok=True)


def _run_direct_or_fallback_gradle(
    build_root: Path,
    test_module: str | None,
    test_paths: Any,
    run_env: dict[str, str],
    timeout: int,
    mode: str,
    candidate_index: int = -1,
    javaagent_arg: str | None = None,
) -> tuple[subprocess.CompletedProcess, Path]:
    """Compile once, then run tests directly via JVM. Falls back to Gradle on failure."""
    test_classes = _get_test_class_names(test_paths, mode=mode)
    if not test_classes:
        logger.warning("No test classes found for mode=%s, returning empty result", mode)
        result_xml_path, empty_result = _get_empty_result(build_root, test_module)
        return empty_result, result_xml_path

    # Step 1: Compile tests
    logger.debug("Step 1: Compiling tests for %s mode (Gradle)", mode)
    compile_result = _compile_tests_gradle(build_root, run_env, test_module, timeout=120)
    if compile_result.returncode != 0:
        logger.warning("Compilation failed (rc=%d), falling back to Gradle-based execution", compile_result.returncode)
        result = _run_gradle_tests(build_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module)
        reports_dir = _get_test_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    # Step 2: Get classpath
    logger.debug("Step 2: Getting classpath (Gradle)")
    classpath = _get_test_classpath_gradle_cached(build_root, run_env, test_module, timeout=60)
    if not classpath:
        logger.warning("Failed to get classpath, falling back to Gradle-based execution")
        result = _run_gradle_tests(build_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module)
        reports_dir = _get_test_reports_dir(build_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
        return result, result_xml_path

    # Step 3: Run tests directly via JVM
    working_dir = build_root / test_module if test_module else build_root
    reports_dir = _get_test_reports_dir(build_root, test_module)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Step 3: Running %s tests directly (bypassing Gradle)", mode)
    result = _run_tests_direct(
        classpath,
        test_classes,
        run_env,
        working_dir,
        timeout=timeout,
        reports_dir=reports_dir,
        javaagent_arg=javaagent_arg,
    )

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
            logger.debug("Direct JVM execution failed, falling back to Gradle-based execution")
            result = _run_gradle_tests(
                build_root, test_paths, run_env, timeout=timeout, mode=mode, test_module=test_module
            )

    result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
    return result, result_xml_path


def _run_gradle_tests_coverage(
    build_root: Path,
    test_module: str | None,
    test_paths: Any,
    run_env: dict[str, str],
    timeout: int,
    candidate_index: int,
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run Gradle tests with JaCoCo coverage enabled."""
    gradle = find_gradle_executable(build_root)
    if not gradle:
        return (
            subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="Gradle not found"),
            _get_combined_junit_xml(_get_test_reports_dir(build_root, test_module), candidate_index),
        )

    init_script = create_codeflash_gradle_init_script(enable_jacoco=True, target_module=test_module)
    if not init_script:
        return (
            subprocess.CompletedProcess(args=["gradle"], returncode=-1, stdout="", stderr="No init script"),
            _get_combined_junit_xml(_get_test_reports_dir(build_root, test_module), candidate_index),
        )

    test_filter = _build_test_filter(test_paths, mode="behavior")
    test_task = f":{test_module}:test" if test_module else "test"
    jacoco_task = f":{test_module}:jacocoTestReportCf" if test_module else "jacocoTestReportCf"
    cmd = [gradle, test_task]

    if test_filter:
        for cls in test_filter.split(","):
            cls = cls.strip()
            if cls:
                cmd.extend(["--tests", cls])

    cmd.extend([jacoco_task, "--no-daemon", "--init-script", str(init_script)])

    try:
        result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=run_env, timeout=timeout)
        if result.returncode != 0 and "compileTestJava" in (result.stdout or "") + (result.stderr or ""):
            deleted = _delete_broken_generated_test_files(result, build_root, test_module, sweep_all=False)
            if deleted:
                logger.info("Deleted %d broken generated test file(s), retrying Gradle coverage test", deleted)
                result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=run_env, timeout=timeout)
    except Exception as e:
        result = subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))
    finally:
        init_script.unlink(missing_ok=True)

    reports_dir = _get_test_reports_dir(build_root, test_module)
    result_xml_path = _get_combined_junit_xml(reports_dir, candidate_index)
    return result, result_xml_path


def _run_benchmarking_tests_gradle(
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
    """Fallback: Run benchmarking tests using Gradle (slower but more reliable)."""
    import time

    project_root = project_root or cwd
    build_root, test_module = _find_multi_module_root(project_root, test_paths)

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_start_time = time.time()
    loop_count = 0
    last_result = None

    per_loop_timeout = max(timeout or 0, 120, 60 + inner_iterations)

    logger.debug("Using Gradle-based benchmarking (fallback mode)")

    for loop_idx in range(1, max_loops + 1):
        run_env = os.environ.copy()
        run_env.update(test_env)
        run_env["CODEFLASH_LOOP_INDEX"] = str(loop_idx)
        run_env["CODEFLASH_MODE"] = "performance"
        run_env["CODEFLASH_TEST_ITERATION"] = "0"
        if "CODEFLASH_INNER_ITERATIONS" not in run_env:
            run_env["CODEFLASH_INNER_ITERATIONS"] = str(inner_iterations)

        result = _run_gradle_tests(
            build_root, test_paths, run_env, timeout=per_loop_timeout, mode="performance", test_module=test_module
        )

        last_result = result
        loop_count = loop_idx

        if result.stdout:
            all_stdout.append(result.stdout)
        if result.stderr:
            all_stderr.append(result.stderr)

        elapsed = time.time() - total_start_time
        if loop_idx >= min_loops and elapsed >= target_duration_seconds:
            logger.debug("Stopping Gradle benchmark after %d loops (%.2fs elapsed)", loop_idx, elapsed)
            break

        if result.returncode != 0:
            timing_pattern = re.compile(r"!######[^:]*:[^:]*:[^:]*:[^:]*:[^:]+:[^:]+######!")
            has_timing_markers = bool(timing_pattern.search(result.stdout or ""))
            if not has_timing_markers:
                logger.warning("Tests failed in Gradle loop %d with no timing markers, stopping", loop_idx)
                break

    combined_result = subprocess.CompletedProcess(
        args=last_result.args if last_result else ["gradle", "test"],
        returncode=last_result.returncode if last_result else -1,
        stdout="\n".join(all_stdout),
        stderr="\n".join(all_stderr),
    )

    reports_dir = _get_test_reports_dir(build_root, test_module)
    result_xml_path = _get_combined_junit_xml(reports_dir, -1)

    return result_xml_path, combined_result


## ---- End of Gradle-specific functions ----


def get_test_run_command(project_root: Path, test_classes: list[str] | None = None) -> list[str]:
    """Get the command to run Java tests.

    Args:
        project_root: Root directory of the Maven project.
        test_classes: Optional list of test class names to run.

    Returns:
        Command as list of strings.

    """
    mvn = find_maven_executable() or "mvn"

    cmd = [mvn, "test", "-B"]

    if test_classes:
        # Validate each test class name to prevent command injection
        validated_classes = []
        for test_class in test_classes:
            if not _validate_java_class_name(test_class):
                msg = f"Invalid test class name: '{test_class}'. Test names must follow Java identifier rules."
                raise ValueError(msg)
            validated_classes.append(test_class)

        cmd.append(f"-Dtest={','.join(validated_classes)}")

    return cmd


_MAVEN_STRATEGY = MavenStrategy()

_GRADLE_STRATEGY = GradleStrategy()
