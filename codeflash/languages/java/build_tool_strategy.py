"""Abstract build tool strategy for Java projects.

Defines the interface for build-tool-specific operations (compilation,
classpath extraction, test execution, coverage). Concrete implementations
live in maven_strategy.py (and future gradle_strategy.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path


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
    ) -> subprocess.CompletedProcess:
        """Compile test code without running tests."""
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
    ) -> subprocess.CompletedProcess:
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
    ) -> tuple[subprocess.CompletedProcess, Path, Path | None]:
        """Run tests with coverage enabled. Returns (result, junit_xml_path, coverage_xml_path)."""
        ...

    @abstractmethod
    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        """Configure coverage tool (e.g. JaCoCo) and return expected XML report path."""
        ...


def get_strategy(project_root: Path) -> BuildToolStrategy:
    """Detect build tool and return the appropriate strategy.

    Raises NotImplementedError for unsupported build tools.
    """
    from codeflash.languages.java.build_tools import BuildTool, detect_build_tool

    build_tool = detect_build_tool(project_root)

    if build_tool == BuildTool.MAVEN:
        from codeflash.languages.java.maven_strategy import MavenStrategy

        return MavenStrategy()

    if build_tool == BuildTool.GRADLE:
        msg = "Gradle support is not yet implemented. Only Maven projects are supported."
        raise NotImplementedError(msg)

    msg = f"No supported build tool found in {project_root}. Expected pom.xml (Maven) or build.gradle (Gradle)."
    raise ValueError(msg)
