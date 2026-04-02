"""Abstract build tool strategy for Java projects.

Defines the interface for build-tool-specific operations (compilation,
classpath extraction, test execution, coverage). Concrete implementations
live in maven_strategy.py and gradle_strategy.py.
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess

    from codeflash.languages.java.build_tools import JavaProjectInfo

logger = logging.getLogger(__name__)

_RUNTIME_JAR_NAME = "codeflash-runtime-1.0.1.jar"
_JAVA_RUNTIME_DIR = Path(__file__).parent.parent.parent.parent / "codeflash-java-runtime"


def module_to_dir(test_module: str) -> str:
    """Convert a build-tool module name to a filesystem-relative path.

    Gradle uses ``:`` as the module separator (``connect:runtime``), while Maven
    uses the directory name directly.  On the filesystem the separator is always
    ``/`` (or ``os.sep``).
    """
    return test_module.replace(":", os.sep)


class BuildToolStrategy(ABC):
    """Strategy interface for Java build tool operations.

    Only methods that genuinely differ between Maven and Gradle belong here.
    Shared logic (direct JVM execution, JUnit XML parsing) stays in test_runner.py.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for log messages (e.g. 'Maven', 'Gradle')."""
        ...

    @abstractmethod
    def get_project_info(self, project_root: Path) -> JavaProjectInfo | None:
        """Extract project metadata (source roots, versions, etc.) from the build configuration."""
        ...

    def find_runtime_jar(self) -> Path | None:
        """Find the codeflash-runtime JAR file.

        Checks package resources and development build directories.
        Subclasses should override to prepend tool-specific cache paths
        and fall back to super().find_runtime_jar().
        """
        resources_jar = Path(__file__).parent / "resources" / _RUNTIME_JAR_NAME
        if resources_jar.exists():
            return resources_jar

        dev_jar_maven = _JAVA_RUNTIME_DIR / "target" / _RUNTIME_JAR_NAME
        if dev_jar_maven.exists():
            return dev_jar_maven

        dev_jar_gradle = _JAVA_RUNTIME_DIR / "build" / "libs" / _RUNTIME_JAR_NAME
        if dev_jar_gradle.exists():
            return dev_jar_gradle

        return None

    def find_wrapper_executable(
        self, build_root: Path, wrapper_names: tuple[str, ...], system_command: str
    ) -> str | None:
        search = build_root.resolve()
        while search != search.parent:
            for name in wrapper_names:
                candidate = search / name
                if candidate.exists():
                    return str(candidate)
            search = search.parent
        return shutil.which(system_command)

    @abstractmethod
    def find_executable(self, build_root: Path) -> str | None:
        """Find the build tool executable, searching up parent directories if needed."""
        ...

    @abstractmethod
    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        """Install codeflash-runtime JAR and register it as a project dependency."""
        ...

    @abstractmethod
    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> bool:
        """Pre-install multi-module dependencies so later invocations skip recompilation."""
        ...

    @abstractmethod
    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        """Compile test code without running tests."""
        ...

    @abstractmethod
    def compile_source_only(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        """Compile only main source code (not tests). Used when test classes are already compiled."""
        ...

    @abstractmethod
    def get_classpath(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        """Return the full test classpath string. Caching is an implementation detail."""
        ...

    @abstractmethod
    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        """Return the directory containing JUnit XML test reports."""
        ...

    @abstractmethod
    def get_build_output_dir(self, build_root: Path, test_module: str | None) -> Path:
        """Return the build output directory (e.g. target/ for Maven, build/ for Gradle)."""
        ...

    @abstractmethod
    def run_tests_via_build_tool(
        self,
        build_root: Path,
        test_paths: Any,
        env: dict[str, str],
        timeout: int,
        mode: str,
        test_module: str | None,
        javaagent_arg: str | None = None,
        enable_coverage: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run tests via the build tool (e.g. Maven Surefire). Used as fallback when direct JVM fails."""
        ...

    @abstractmethod
    def run_benchmarking_via_build_tool(
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
        """Run benchmarking loop via build tool (fallback when direct JVM fails)."""
        ...

    @abstractmethod
    def run_tests_with_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path | None]:
        """Run tests with coverage enabled. Returns (result, junit_xml_path, coverage_xml_path)."""
        ...

    @abstractmethod
    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        """Configure coverage tool (e.g. JaCoCo) and return expected XML report path."""
        ...

    @abstractmethod
    def get_test_run_command(self, project_root: Path, test_classes: list[str] | None = None) -> list[str]:
        """Return the shell command to run tests, including any test class filters."""
        ...


def _build_strategy_registry() -> dict[str, type[BuildToolStrategy]]:
    """Lazily import and return the {BuildTool.value -> class} mapping."""
    from codeflash.languages.java.gradle_strategy import GradleStrategy
    from codeflash.languages.java.maven_strategy import MavenStrategy

    return {"maven": MavenStrategy, "gradle": GradleStrategy}


def get_strategy(project_root: Path) -> BuildToolStrategy:
    """Detect build tool and return the appropriate strategy."""
    from codeflash.languages.java.build_tools import detect_build_tool

    build_tool = detect_build_tool(project_root)
    registry = _build_strategy_registry()

    strategy_cls = registry.get(build_tool.value)
    if strategy_cls is not None:
        return strategy_cls()

    supported = ", ".join(registry)
    msg = f"No supported build tool found in {project_root}. Expected one of: {supported}."
    raise ValueError(msg)
