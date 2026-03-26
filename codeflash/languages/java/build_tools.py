"""Java build tool detection and integration.

This module provides functionality to detect and work with Java build tools
(Maven and Gradle), including running tests and managing dependencies.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path  # noqa: TC003 — used at runtime

logger = logging.getLogger(__name__)

CODEFLASH_RUNTIME_VERSION = "1.0.0"
CODEFLASH_RUNTIME_JAR_NAME = f"codeflash-runtime-{CODEFLASH_RUNTIME_VERSION}.jar"

JACOCO_PLUGIN_VERSION = "0.8.13"


def _safe_parse_xml(file_path: Path) -> ET.ElementTree:
    """Safely parse an XML file with protections against XXE attacks.

    Args:
        file_path: Path to the XML file.

    Returns:
        Parsed ElementTree.

    Raises:
        ET.ParseError: If XML parsing fails.

    """
    # Read file content and parse as string to avoid file-based attacks
    # This prevents XXE attacks by not allowing external entity resolution
    content = file_path.read_text(encoding="utf-8")

    # Parse string content (no external entities possible)
    root = ET.fromstring(content)

    # Create ElementTree from root
    return ET.ElementTree(root)


class BuildTool(Enum):
    """Supported Java build tools."""

    MAVEN = "maven"
    GRADLE = "gradle"
    UNKNOWN = "unknown"


@dataclass
class JavaProjectInfo:
    """Information about a Java project."""

    project_root: Path
    build_tool: BuildTool
    source_roots: list[Path]
    test_roots: list[Path]
    target_dir: Path  # build output directory
    group_id: str | None
    artifact_id: str | None
    version: str | None
    java_version: str | None


@dataclass
class MavenTestResult:
    """Result of running Maven tests."""

    success: bool
    tests_run: int
    failures: int
    errors: int
    skipped: int
    surefire_reports_dir: Path | None
    stdout: str
    stderr: str
    returncode: int


def detect_build_tool(project_root: Path) -> BuildTool:
    """Detect which build tool a Java project uses.

    Args:
        project_root: Root directory of the Java project.

    Returns:
        The detected BuildTool enum value.

    """
    # Check for Maven (pom.xml)
    if (project_root / "pom.xml").exists():
        return BuildTool.MAVEN

    # Check for Gradle (build.gradle or build.gradle.kts)
    if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
        return BuildTool.GRADLE

    # Check in parent directories for multi-module projects
    current = project_root
    for _ in range(3):  # Check up to 3 levels
        parent = current.parent
        if parent == current:
            break
        if (parent / "pom.xml").exists():
            return BuildTool.MAVEN
        if (parent / "build.gradle").exists() or (parent / "build.gradle.kts").exists():
            return BuildTool.GRADLE
        current = parent

    return BuildTool.UNKNOWN


def get_project_info(project_root: Path) -> JavaProjectInfo | None:
    """Get information about a Java project.

    Args:
        project_root: Root directory of the Java project.

    Returns:
        JavaProjectInfo if a supported project is found, None otherwise.

    """
    build_tool = detect_build_tool(project_root)
    if build_tool == BuildTool.UNKNOWN:
        return None

    from codeflash.languages.java.build_tool_strategy import get_strategy

    try:
        strategy = get_strategy(project_root)
    except ValueError:
        return None
    return strategy.get_project_info(project_root)


def _parse_surefire_reports(surefire_dir: Path) -> tuple[int, int, int, int]:
    """Parse Surefire XML reports to get test counts.

    Returns:
        Tuple of (tests_run, failures, errors, skipped).

    """
    tests_run = 0
    failures = 0
    errors = 0
    skipped = 0

    if not surefire_dir.exists():
        return tests_run, failures, errors, skipped

    for xml_file in surefire_dir.glob("TEST-*.xml"):
        try:
            tree = _safe_parse_xml(xml_file)
            root = tree.getroot()
            if root is None:
                continue

            try:
                tests_run += int(root.get("tests", "0"))
            except (ValueError, TypeError):
                logger.warning("Invalid 'tests' value in %s, defaulting to 0", xml_file)

            try:
                failures += int(root.get("failures", "0"))
            except (ValueError, TypeError):
                logger.warning("Invalid 'failures' value in %s, defaulting to 0", xml_file)

            try:
                errors += int(root.get("errors", "0"))
            except (ValueError, TypeError):
                logger.warning("Invalid 'errors' value in %s, defaulting to 0", xml_file)

            try:
                skipped += int(root.get("skipped", "0"))
            except (ValueError, TypeError):
                logger.warning("Invalid 'skipped' value in %s, defaulting to 0", xml_file)

        except ET.ParseError as e:
            logger.warning("Failed to parse Surefire report %s: %s", xml_file, e)
        except Exception as e:
            logger.warning("Unexpected error parsing Surefire report %s: %s", xml_file, e)

    return tests_run, failures, errors, skipped


def find_test_root(project_root: Path) -> Path | None:
    """Find the test root directory for a Java project.

    Args:
        project_root: Root directory of the Java project.

    Returns:
        Path to test root, or None if not found.

    """
    build_tool = detect_build_tool(project_root)

    if build_tool in (BuildTool.MAVEN, BuildTool.GRADLE):
        test_root = project_root / "src" / "test" / "java"
        if test_root.exists():
            return test_root

    # Check common alternative locations
    for test_dir in ["test", "tests", "src/test"]:
        test_path = project_root / test_dir
        if test_path.exists():
            return test_path

    return None


def find_source_root(project_root: Path) -> Path | None:
    """Find the main source root directory for a Java project.

    Args:
        project_root: Root directory of the Java project.

    Returns:
        Path to source root, or None if not found.

    """
    build_tool = detect_build_tool(project_root)

    if build_tool in (BuildTool.MAVEN, BuildTool.GRADLE):
        src_root = project_root / "src" / "main" / "java"
        if src_root.exists():
            return src_root

    # Check common alternative locations
    for src_dir in ["src", "source", "java"]:
        src_path = project_root / src_dir
        if src_path.exists() and any(src_path.rglob("*.java")):
            return src_path

    return None
