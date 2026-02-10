"""Java project configuration detection.

This module provides functionality to detect and read Java project
configuration, including build tool settings, test framework configuration,
and project structure.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeflash.languages.java.build_tools import (
    BuildTool,
    detect_build_tool,
    find_source_root,
    find_test_root,
    get_project_info,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JavaProjectConfig:
    """Configuration for a Java project."""

    project_root: Path
    build_tool: BuildTool
    source_root: Path | None
    test_root: Path | None
    java_version: str | None
    encoding: str
    test_framework: str  # "junit5", "junit4", "testng"
    group_id: str | None
    artifact_id: str | None
    version: str | None

    # Dependencies
    has_junit5: bool = False
    has_junit4: bool = False
    has_testng: bool = False
    has_mockito: bool = False
    has_assertj: bool = False

    # Build configuration
    compiler_source: str | None = None
    compiler_target: str | None = None

    # Plugin configurations
    surefire_includes: list[str] = field(default_factory=list)
    surefire_excludes: list[str] = field(default_factory=list)


def detect_java_project(project_root: Path) -> JavaProjectConfig | None:
    """Detect and return Java project configuration.

    Args:
        project_root: Root directory of the project.

    Returns:
        JavaProjectConfig if a Java project is detected, None otherwise.

    """
    # Check if this is a Java project
    build_tool = detect_build_tool(project_root)
    if build_tool == BuildTool.UNKNOWN:
        # Check if there are any Java files
        java_files = list(project_root.rglob("*.java"))
        if not java_files:
            return None

    # Get basic project info
    project_info = get_project_info(project_root)

    # Detect test framework
    test_framework, has_junit5, has_junit4, has_testng = _detect_test_framework(project_root, build_tool)

    # Detect other dependencies
    has_mockito, has_assertj = _detect_test_dependencies(project_root, build_tool)

    # Get source/test roots
    source_root = find_source_root(project_root)
    test_root = find_test_root(project_root)

    # Get compiler settings
    compiler_source, compiler_target = _get_compiler_settings(project_root, build_tool)

    # Get surefire configuration
    surefire_includes, surefire_excludes = _get_surefire_config(project_root)

    return JavaProjectConfig(
        project_root=project_root,
        build_tool=build_tool,
        source_root=source_root,
        test_root=test_root,
        java_version=project_info.java_version if project_info else None,
        encoding="UTF-8",  # Default, could be detected from pom.xml
        test_framework=test_framework,
        group_id=project_info.group_id if project_info else None,
        artifact_id=project_info.artifact_id if project_info else None,
        version=project_info.version if project_info else None,
        has_junit5=has_junit5,
        has_junit4=has_junit4,
        has_testng=has_testng,
        has_mockito=has_mockito,
        has_assertj=has_assertj,
        compiler_source=compiler_source,
        compiler_target=compiler_target,
        surefire_includes=surefire_includes,
        surefire_excludes=surefire_excludes,
    )


def _detect_test_framework(project_root: Path, build_tool: BuildTool) -> tuple[str, bool, bool, bool]:
    """Detect which test framework the project uses.

    Args:
        project_root: Root directory of the project.
        build_tool: The detected build tool.

    Returns:
        Tuple of (framework_name, has_junit5, has_junit4, has_testng).

    """
    has_junit5 = False
    has_junit4 = False
    has_testng = False

    if build_tool == BuildTool.MAVEN:
        has_junit5, has_junit4, has_testng = _detect_test_deps_from_pom(project_root)
    elif build_tool == BuildTool.GRADLE:
        has_junit5, has_junit4, has_testng = _detect_test_deps_from_gradle(project_root)

    # Also check test source files for import statements
    test_root = find_test_root(project_root)
    if test_root and test_root.exists():
        for test_file in test_root.rglob("*.java"):
            try:
                content = test_file.read_text(encoding="utf-8")
                if "org.junit.jupiter" in content:
                    has_junit5 = True
                if "org.junit.Test" in content or "org.junit.Assert" in content:
                    has_junit4 = True
                if "org.testng" in content:
                    has_testng = True
            except Exception:
                pass

    # Determine primary framework (prefer JUnit 5)
    if has_junit5:
        return "junit5", has_junit5, has_junit4, has_testng
    if has_junit4:
        return "junit4", has_junit5, has_junit4, has_testng
    if has_testng:
        return "testng", has_junit5, has_junit4, has_testng

    # Default to JUnit 5 if nothing detected
    return "junit5", has_junit5, has_junit4, has_testng


def _detect_test_deps_from_pom(project_root: Path) -> tuple[bool, bool, bool]:
    """Detect test framework dependencies from pom.xml.

    Returns:
        Tuple of (has_junit5, has_junit4, has_testng).

    """
    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return False, False, False

    has_junit5 = False
    has_junit4 = False
    has_testng = False

    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Handle namespace
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        # Search for dependencies
        for deps_path in ["dependencies", "m:dependencies"]:
            deps = root.find(deps_path, ns) if "m:" in deps_path else root.find(deps_path)
            if deps is None:
                continue

            for dep_path in ["dependency", "m:dependency"]:
                deps_list = deps.findall(dep_path, ns) if "m:" in dep_path else deps.findall(dep_path)
                for dep in deps_list:
                    artifact_id = None
                    group_id = None

                    for child in dep:
                        tag = child.tag.replace("{http://maven.apache.org/POM/4.0.0}", "")
                        if tag == "artifactId":
                            artifact_id = child.text
                        elif tag == "groupId":
                            group_id = child.text

                    if group_id == "org.junit.jupiter" or (artifact_id and "junit-jupiter" in artifact_id):
                        has_junit5 = True
                    elif group_id == "junit" and artifact_id == "junit":
                        has_junit4 = True
                    elif group_id == "org.testng":
                        has_testng = True

    except ET.ParseError:
        pass

    return has_junit5, has_junit4, has_testng


def _detect_test_deps_from_gradle(project_root: Path) -> tuple[bool, bool, bool]:
    """Detect test framework dependencies from Gradle build files.

    Returns:
        Tuple of (has_junit5, has_junit4, has_testng).

    """
    has_junit5 = False
    has_junit4 = False
    has_testng = False

    for gradle_file in ["build.gradle", "build.gradle.kts"]:
        gradle_path = project_root / gradle_file
        if gradle_path.exists():
            try:
                content = gradle_path.read_text(encoding="utf-8")
                if "junit-jupiter" in content or "useJUnitPlatform" in content:
                    has_junit5 = True
                if "junit:junit" in content:
                    has_junit4 = True
                if "testng" in content.lower():
                    has_testng = True
            except Exception:
                pass

    return has_junit5, has_junit4, has_testng


def _detect_test_dependencies(project_root: Path, build_tool: BuildTool) -> tuple[bool, bool]:
    """Detect additional test dependencies (Mockito, AssertJ).

    Returns:
        Tuple of (has_mockito, has_assertj).

    """
    has_mockito = False
    has_assertj = False

    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            content = pom_path.read_text(encoding="utf-8")
            has_mockito = "mockito" in content.lower()
            has_assertj = "assertj" in content.lower()
        except Exception:
            pass

    for gradle_file in ["build.gradle", "build.gradle.kts"]:
        gradle_path = project_root / gradle_file
        if gradle_path.exists():
            try:
                content = gradle_path.read_text(encoding="utf-8")
                if "mockito" in content.lower():
                    has_mockito = True
                if "assertj" in content.lower():
                    has_assertj = True
            except Exception:
                pass

    return has_mockito, has_assertj


def _get_compiler_settings(project_root: Path, build_tool: BuildTool) -> tuple[str | None, str | None]:
    """Get compiler source and target settings.

    Returns:
        Tuple of (source_version, target_version).

    """
    if build_tool != BuildTool.MAVEN:
        return None, None

    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return None, None

    source = None
    target = None

    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        # Check properties
        for props_path in ["properties", "m:properties"]:
            props = root.find(props_path, ns) if "m:" in props_path else root.find(props_path)
            if props is not None:
                for child in props:
                    tag = child.tag.replace("{http://maven.apache.org/POM/4.0.0}", "")
                    if tag == "maven.compiler.source":
                        source = child.text
                    elif tag == "maven.compiler.target":
                        target = child.text

    except ET.ParseError:
        pass

    return source, target


def _get_surefire_config(project_root: Path) -> tuple[list[str], list[str]]:
    """Get Maven Surefire plugin includes/excludes configuration.

    Returns:
        Tuple of (includes, excludes) patterns.

    """
    includes: list[str] = []
    excludes: list[str] = []

    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return includes, excludes

    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        # Find surefire plugin configuration
        # This is a simplified search - a full implementation would
        # handle nested build/plugins/plugin structure

        content = pom_path.read_text(encoding="utf-8")
        if "maven-surefire-plugin" in content:
            # Parse includes/excludes if present
            # This is a basic implementation
            pass

    except (ET.ParseError, Exception):
        pass

    # Return default patterns if none configured
    if not includes:
        includes = ["**/Test*.java", "**/*Test.java", "**/*Tests.java", "**/*TestCase.java"]
    if not excludes:
        excludes = ["**/*IT.java", "**/*IntegrationTest.java"]

    return includes, excludes


def is_java_project(project_root: Path) -> bool:
    """Check if a directory is a Java project.

    Args:
        project_root: Directory to check.

    Returns:
        True if this appears to be a Java project.

    """
    # Check for build tool config files
    if (project_root / "pom.xml").exists():
        return True
    if (project_root / "build.gradle").exists():
        return True
    if (project_root / "build.gradle.kts").exists():
        return True

    # Check for Java source files
    return any(list(project_root.glob(pattern)) for pattern in ["src/**/*.java", "*.java"])


def get_test_file_pattern(config: JavaProjectConfig) -> str:
    """Get the test file naming pattern for a project.

    Args:
        config: The project configuration.

    Returns:
        Glob pattern for test files.

    """
    # Default JUnit pattern
    return "*Test.java"


def get_test_class_pattern(config: JavaProjectConfig) -> str:
    """Get the regex pattern for test class names.

    Args:
        config: The project configuration.

    Returns:
        Regex pattern for test class names.

    """
    return r".*Test(s)?$|^Test.*"
