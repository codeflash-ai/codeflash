"""Java build tool detection and integration.

This module provides functionality to detect and work with Java build tools
(Maven and Gradle), including running tests and managing dependencies.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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


def parse_java_project_config(project_root: Path) -> dict[str, Any] | None:
    """Parse codeflash config from Maven/Gradle build files.

    Reads codeflash.* properties from pom.xml <properties> or gradle.properties,
    then fills in defaults from auto-detected build tool conventions.

    Returns None if no Java build tool is detected.
    """
    build_tool = detect_build_tool(project_root)
    if build_tool == BuildTool.UNKNOWN:
        return None

    # Read explicit codeflash properties from build files
    user_config: dict[str, str] = {}
    if build_tool == BuildTool.MAVEN:
        user_config = _read_maven_codeflash_properties(project_root)
    elif build_tool == BuildTool.GRADLE:
        user_config = _read_gradle_codeflash_properties(project_root)

    # Auto-detect defaults — for multi-module Maven projects, scan module pom.xml files
    source_root = find_source_root(project_root)
    test_root = find_test_root(project_root)

    if build_tool == BuildTool.MAVEN:
        source_from_modules, test_from_modules = _detect_roots_from_maven_modules(project_root)
        # Module-level pom.xml declarations are more precise than directory-name heuristics
        if source_from_modules is not None:
            source_root = source_from_modules
        if test_from_modules is not None:
            test_root = test_from_modules

    # Build the config dict matching the format expected by the rest of codeflash
    config: dict[str, Any] = {
        "language": "java",
        "module_root": str(
            (project_root / user_config["moduleRoot"]).resolve()
            if "moduleRoot" in user_config
            else (source_root or project_root / "src" / "main" / "java")
        ),
        "tests_root": str(
            (project_root / user_config["testsRoot"]).resolve()
            if "testsRoot" in user_config
            else (test_root or project_root / "src" / "test" / "java")
        ),
        "pytest_cmd": "pytest",
        "git_remote": user_config.get("gitRemote", "origin"),
        "disable_telemetry": user_config.get("disableTelemetry", "false").lower() == "true",
        "disable_imports_sorting": False,
        "override_fixtures": False,
        "benchmark": False,
        "formatter_cmds": [],
        "ignore_paths": [],
    }

    if "ignorePaths" in user_config:
        config["ignore_paths"] = [
            str((project_root / p.strip()).resolve()) for p in user_config["ignorePaths"].split(",") if p.strip()
        ]

    if "formatterCmds" in user_config:
        config["formatter_cmds"] = [cmd.strip() for cmd in user_config["formatterCmds"].split(",") if cmd.strip()]

    return config


def _read_maven_codeflash_properties(project_root: Path) -> dict[str, str]:
    """Read codeflash.* properties from pom.xml <properties> section."""
    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return {}

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        result: dict[str, str] = {}
        for props in [root.find("m:properties", ns), root.find("properties")]:
            if props is None:
                continue
            for child in props:
                tag = child.tag
                # Strip Maven namespace prefix
                if "}" in tag:
                    tag = tag.split("}", 1)[1]
                if tag.startswith("codeflash.") and child.text:
                    key = tag[len("codeflash.") :]
                    result[key] = child.text.strip()
        return result
    except Exception:
        logger.debug("Failed to read codeflash properties from pom.xml", exc_info=True)
        return {}


def _read_gradle_codeflash_properties(project_root: Path) -> dict[str, str]:
    """Read codeflash.* properties from gradle.properties."""
    props_path = project_root / "gradle.properties"
    if not props_path.exists():
        return {}

    result: dict[str, str] = {}
    try:
        with props_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                if key.startswith("codeflash."):
                    result[key[len("codeflash.") :]] = value.strip()
        return result
    except Exception:
        logger.debug("Failed to read codeflash properties from gradle.properties", exc_info=True)
        return {}


def _detect_roots_from_maven_modules(project_root: Path) -> tuple[Path | None, Path | None]:
    """Scan Maven module pom.xml files for custom sourceDirectory/testSourceDirectory.

    For multi-module projects like aerospike (client/, test/, benchmarks/),
    finds the main source module and test module by parsing each module's build config.
    """
    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return None, None

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        # Find <modules> to get module names
        modules: list[str] = []
        for modules_elem in [root.find("m:modules", ns), root.find("modules")]:
            if modules_elem is not None:
                for mod in modules_elem:
                    if mod.text:
                        modules.append(mod.text.strip())

        if not modules:
            return None, None

        # Collect candidate source and test roots with Java file counts
        source_candidates: list[tuple[Path, int]] = []
        test_root: Path | None = None

        skip_modules = {"example", "examples", "benchmark", "benchmarks", "demo", "sample", "samples"}

        for module_name in modules:
            module_pom = project_root / module_name / "pom.xml"
            if not module_pom.exists():
                continue

            # Modules named "test" are test modules, not source modules
            is_test_module = "test" in module_name.lower()

            try:
                mod_tree = _safe_parse_xml(module_pom)
                mod_root = mod_tree.getroot()

                for build in [mod_root.find("m:build", ns), mod_root.find("build")]:
                    if build is None:
                        continue

                    for src_elem in [build.find("m:sourceDirectory", ns), build.find("sourceDirectory")]:
                        if src_elem is not None and src_elem.text:
                            src_text = src_elem.text.replace("${project.basedir}", str(project_root / module_name))
                            src_path = Path(src_text)
                            if not src_path.is_absolute():
                                src_path = project_root / module_name / src_path
                            if src_path.exists():
                                if is_test_module and test_root is None:
                                    test_root = src_path
                                elif module_name.lower() not in skip_modules:
                                    java_count = sum(1 for _ in src_path.rglob("*.java"))
                                    if java_count > 0:
                                        source_candidates.append((src_path, java_count))

                    for test_elem in [build.find("m:testSourceDirectory", ns), build.find("testSourceDirectory")]:
                        if test_elem is not None and test_elem.text:
                            test_text = test_elem.text.replace("${project.basedir}", str(project_root / module_name))
                            test_path = Path(test_text)
                            if not test_path.is_absolute():
                                test_path = project_root / module_name / test_path
                            if test_path.exists() and test_root is None:
                                test_root = test_path

                # Also check standard module layouts
                if module_name.lower() not in skip_modules and not is_test_module:
                    std_src = project_root / module_name / "src" / "main" / "java"
                    if std_src.exists():
                        java_count = sum(1 for _ in std_src.rglob("*.java"))
                        if java_count > 0:
                            source_candidates.append((std_src, java_count))

                if test_root is None:
                    std_test = project_root / module_name / "src" / "test" / "java"
                    if std_test.exists() and any(std_test.rglob("*.java")):
                        test_root = std_test

            except Exception:
                continue

        # Pick the source root with the most Java files (likely the main library)
        source_root = max(source_candidates, key=lambda x: x[1])[0] if source_candidates else None
        return source_root, test_root

    except Exception:
        return None, None


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
