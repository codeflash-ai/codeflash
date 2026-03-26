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

    if build_tool == BuildTool.MAVEN:
        return _get_maven_project_info(project_root)
    if build_tool == BuildTool.GRADLE:
        return _get_gradle_project_info(project_root)

    return None


def _get_maven_project_info(project_root: Path) -> JavaProjectInfo | None:
    """Get project info from Maven pom.xml.

    Args:
        project_root: Root directory of the Maven project.

    Returns:
        JavaProjectInfo extracted from pom.xml.

    """
    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return None

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()

        # Handle Maven namespace
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        def get_text(xpath: str, default: str | None = None) -> str | None:
            # Try with namespace first
            elem = root.find(f"m:{xpath}", ns)
            if elem is None:
                # Try without namespace
                elem = root.find(xpath)
            return elem.text if elem is not None else default

        group_id = get_text("groupId")
        artifact_id = get_text("artifactId")
        version = get_text("version")

        # Get Java version from properties or compiler plugin
        java_version = _extract_java_version_from_pom(root, ns)

        # Standard Maven directory structure
        source_roots = []
        test_roots = []

        main_src = project_root / "src" / "main" / "java"
        if main_src.exists():
            source_roots.append(main_src)

        test_src = project_root / "src" / "test" / "java"
        if test_src.exists():
            test_roots.append(test_src)

        # Check for custom source directories in pom.xml <build> section
        for build in [root.find("m:build", ns), root.find("build")]:
            if build is not None:
                for tag, roots_list in [("sourceDirectory", source_roots), ("testSourceDirectory", test_roots)]:
                    for elem in [build.find(f"m:{tag}", ns), build.find(tag)]:
                        if elem is not None and elem.text:
                            custom_dir = project_root / elem.text.strip()
                            if custom_dir.exists() and custom_dir not in roots_list:
                                roots_list.append(custom_dir)

        target_dir = project_root / "target"

        return JavaProjectInfo(
            project_root=project_root,
            build_tool=BuildTool.MAVEN,
            source_roots=source_roots,
            test_roots=test_roots,
            target_dir=target_dir,
            group_id=group_id,
            artifact_id=artifact_id,
            version=version,
            java_version=java_version,
        )

    except ET.ParseError as e:
        logger.warning("Failed to parse pom.xml: %s", e)
        return None


def _extract_java_version_from_pom(root: ET.Element, ns: dict[str, str]) -> str | None:
    """Extract Java version from Maven pom.xml.

    Checks multiple locations:
    1. properties/maven.compiler.source
    2. properties/java.version
    3. build/plugins/plugin[compiler]/configuration/source

    Args:
        root: Root element of the pom.xml.
        ns: XML namespace mapping.

    Returns:
        Java version string or None.

    """
    # Check properties
    for prop_name in ("maven.compiler.source", "java.version", "maven.compiler.release"):
        for props in [root.find("m:properties", ns), root.find("properties")]:
            if props is not None:
                for prop in [props.find(f"m:{prop_name}", ns), props.find(prop_name)]:
                    if prop is not None and prop.text:
                        return prop.text

    # Check compiler plugin configuration
    for build in [root.find("m:build", ns), root.find("build")]:
        if build is not None:
            for plugins in [build.find("m:plugins", ns), build.find("plugins")]:
                if plugins is not None:
                    for plugin in plugins.findall("m:plugin", ns) + plugins.findall("plugin"):
                        artifact_id = plugin.find("m:artifactId", ns) or plugin.find("artifactId")
                        if artifact_id is not None and artifact_id.text == "maven-compiler-plugin":
                            config = plugin.find("m:configuration", ns) or plugin.find("configuration")
                            if config is not None:
                                source = config.find("m:source", ns) or config.find("source")
                                if source is not None and source.text:
                                    return source.text

    return None


def _get_gradle_project_info(project_root: Path) -> JavaProjectInfo | None:
    """Get project info from Gradle build file.

    Note: This is a basic implementation. Full Gradle parsing would require
    running Gradle with a custom task or using the Gradle tooling API.

    Args:
        project_root: Root directory of the Gradle project.

    Returns:
        JavaProjectInfo with basic Gradle project structure.

    """
    # Standard Gradle directory structure
    source_roots = []
    test_roots = []

    main_src = project_root / "src" / "main" / "java"
    if main_src.exists():
        source_roots.append(main_src)

    test_src = project_root / "src" / "test" / "java"
    if test_src.exists():
        test_roots.append(test_src)

    build_dir = project_root / "build"

    return JavaProjectInfo(
        project_root=project_root,
        build_tool=BuildTool.GRADLE,
        source_roots=source_roots,
        test_roots=test_roots,
        target_dir=build_dir,
        group_id=None,  # Would need to parse build.gradle
        artifact_id=None,
        version=None,
        java_version=None,
    )


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
