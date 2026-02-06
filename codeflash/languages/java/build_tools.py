"""Java build tool detection and integration.

This module provides functionality to detect and work with Java build tools
(Maven and Gradle), including running tests and managing dependencies.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


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


def find_maven_executable() -> str | None:
    """Find the Maven executable.

    Returns:
        Path to mvn executable, or None if not found.

    """
    # Check for Maven wrapper first
    if Path("mvnw").exists():
        return "./mvnw"
    if Path("mvnw.cmd").exists():
        return "mvnw.cmd"

    # Check system Maven
    mvn_path = shutil.which("mvn")
    if mvn_path:
        return mvn_path

    return None


def find_gradle_executable() -> str | None:
    """Find the Gradle executable.

    Returns:
        Path to gradle executable, or None if not found.

    """
    # Check for Gradle wrapper first
    if Path("gradlew").exists():
        return "./gradlew"
    if Path("gradlew.bat").exists():
        return "gradlew.bat"

    # Check system Gradle
    gradle_path = shutil.which("gradle")
    if gradle_path:
        return gradle_path

    return None


def run_maven_tests(
    project_root: Path,
    test_classes: list[str] | None = None,
    test_methods: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    skip_compilation: bool = False,
) -> MavenTestResult:
    """Run Maven tests using Surefire.

    Args:
        project_root: Root directory of the Maven project.
        test_classes: Optional list of test class names to run.
        test_methods: Optional list of specific test methods (format: ClassName#methodName).
        env: Optional environment variables.
        timeout: Maximum time in seconds for test execution.
        skip_compilation: Whether to skip compilation (useful when only running tests).

    Returns:
        MavenTestResult with test execution results.

    """
    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found. Please install Maven or use Maven wrapper.")
        return MavenTestResult(
            success=False,
            tests_run=0,
            failures=0,
            errors=0,
            skipped=0,
            surefire_reports_dir=None,
            stdout="",
            stderr="Maven not found",
            returncode=-1,
        )

    # Build Maven command
    cmd = [mvn]

    if skip_compilation:
        cmd.append("-Dmaven.test.skip=false")
        cmd.append("-DskipTests=false")
        cmd.append("surefire:test")
    else:
        cmd.append("test")

    # Add test filtering
    if test_classes or test_methods:
        if test_methods:
            # Format: -Dtest=ClassName#method1+method2,OtherClass#method3
            tests = ",".join(test_methods)
        elif test_classes:
            tests = ",".join(test_classes)
        cmd.extend(["-Dtest=" + tests])

    # Fail at end to run all tests
    cmd.append("-fae")

    # Use full environment with optional overrides
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd, check=False, cwd=project_root, env=run_env, capture_output=True, text=True, timeout=timeout
        )

        # Parse test results from Surefire reports
        surefire_dir = project_root / "target" / "surefire-reports"
        tests_run, failures, errors, skipped = _parse_surefire_reports(surefire_dir)

        return MavenTestResult(
            success=result.returncode == 0,
            tests_run=tests_run,
            failures=failures,
            errors=errors,
            skipped=skipped,
            surefire_reports_dir=surefire_dir if surefire_dir.exists() else None,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

    except subprocess.TimeoutExpired:
        logger.exception("Maven test execution timed out after %d seconds", timeout)
        return MavenTestResult(
            success=False,
            tests_run=0,
            failures=0,
            errors=0,
            skipped=0,
            surefire_reports_dir=None,
            stdout="",
            stderr=f"Test execution timed out after {timeout} seconds",
            returncode=-2,
        )
    except Exception as e:
        logger.exception("Maven test execution failed: %s", e)
        return MavenTestResult(
            success=False,
            tests_run=0,
            failures=0,
            errors=0,
            skipped=0,
            surefire_reports_dir=None,
            stdout="",
            stderr=str(e),
            returncode=-1,
        )


def _parse_surefire_reports(surefire_dir: Path) -> tuple[int, int, int, int]:
    """Parse Surefire XML reports to get test counts.

    Args:
        surefire_dir: Directory containing Surefire XML reports.

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

            # Safely parse numeric attributes with validation
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


def compile_maven_project(
    project_root: Path, include_tests: bool = True, env: dict[str, str] | None = None, timeout: int = 300
) -> tuple[bool, str, str]:
    """Compile a Maven project.

    Args:
        project_root: Root directory of the Maven project.
        include_tests: Whether to compile test classes as well.
        env: Optional environment variables.
        timeout: Maximum time in seconds for compilation.

    Returns:
        Tuple of (success, stdout, stderr).

    """
    mvn = find_maven_executable()
    if not mvn:
        return False, "", "Maven not found"

    cmd = [mvn]

    if include_tests:
        cmd.append("test-compile")
    else:
        cmd.append("compile")

    # Skip test execution
    cmd.append("-DskipTests")

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            cmd, check=False, cwd=project_root, env=run_env, capture_output=True, text=True, timeout=timeout
        )

        return result.returncode == 0, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return False, "", f"Compilation timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)


def install_codeflash_runtime(project_root: Path, runtime_jar_path: Path) -> bool:
    """Install the codeflash runtime JAR to the local Maven repository.

    Args:
        project_root: Root directory of the Maven project.
        runtime_jar_path: Path to the codeflash-runtime.jar file.

    Returns:
        True if installation succeeded, False otherwise.

    """
    mvn = find_maven_executable()
    if not mvn:
        logger.error("Maven not found")
        return False

    if not runtime_jar_path.exists():
        logger.error("Runtime JAR not found: %s", runtime_jar_path)
        return False

    cmd = [
        mvn,
        "install:install-file",
        f"-Dfile={runtime_jar_path}",
        "-DgroupId=com.codeflash",
        "-DartifactId=codeflash-runtime",
        "-Dversion=1.0.0",
        "-Dpackaging=jar",
    ]

    try:
        result = subprocess.run(cmd, check=False, cwd=project_root, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            logger.info("Successfully installed codeflash-runtime to local Maven repository")
            return True
        logger.error("Failed to install codeflash-runtime: %s", result.stderr)
        return False

    except Exception as e:
        logger.exception("Failed to install codeflash-runtime: %s", e)
        return False


def add_codeflash_dependency_to_pom(pom_path: Path) -> bool:
    """Add codeflash-runtime dependency to pom.xml if not present.

    Args:
        pom_path: Path to the pom.xml file.

    Returns:
        True if dependency was added or already present, False on error.

    """
    if not pom_path.exists():
        return False

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()

        # Handle Maven namespace
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}
        ns_prefix = "{http://maven.apache.org/POM/4.0.0}"

        # Check if namespace is used
        if root.tag.startswith("{"):
            use_ns = True
        else:
            use_ns = False
            ns_prefix = ""

        # Find or create dependencies section
        deps = root.find(f"{ns_prefix}dependencies" if use_ns else "dependencies")
        if deps is None:
            deps = ET.SubElement(root, f"{ns_prefix}dependencies" if use_ns else "dependencies")

        # Check if codeflash dependency already exists
        for dep in deps.findall(f"{ns_prefix}dependency" if use_ns else "dependency"):
            group = dep.find(f"{ns_prefix}groupId" if use_ns else "groupId")
            artifact = dep.find(f"{ns_prefix}artifactId" if use_ns else "artifactId")
            if group is not None and artifact is not None:
                if group.text == "com.codeflash" and artifact.text == "codeflash-runtime":
                    logger.info("codeflash-runtime dependency already present in pom.xml")
                    return True

        # Add codeflash dependency
        dep_elem = ET.SubElement(deps, f"{ns_prefix}dependency" if use_ns else "dependency")

        group_elem = ET.SubElement(dep_elem, f"{ns_prefix}groupId" if use_ns else "groupId")
        group_elem.text = "com.codeflash"

        artifact_elem = ET.SubElement(dep_elem, f"{ns_prefix}artifactId" if use_ns else "artifactId")
        artifact_elem.text = "codeflash-runtime"

        version_elem = ET.SubElement(dep_elem, f"{ns_prefix}version" if use_ns else "version")
        version_elem.text = "1.0.0"

        scope_elem = ET.SubElement(dep_elem, f"{ns_prefix}scope" if use_ns else "scope")
        scope_elem.text = "test"

        # Write back to file
        tree.write(pom_path, xml_declaration=True, encoding="utf-8")
        logger.info("Added codeflash-runtime dependency to pom.xml")
        return True

    except ET.ParseError as e:
        logger.exception("Failed to parse pom.xml: %s", e)
        return False
    except Exception as e:
        logger.exception("Failed to add dependency to pom.xml: %s", e)
        return False


JACOCO_PLUGIN_VERSION = "0.8.11"


def is_jacoco_configured(pom_path: Path) -> bool:
    """Check if JaCoCo plugin is already configured in pom.xml.

    Checks both the main build section and any profile build sections.

    Args:
        pom_path: Path to the pom.xml file.

    Returns:
        True if JaCoCo plugin is configured anywhere in the pom.xml, False otherwise.

    """
    if not pom_path.exists():
        return False

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()

        # Handle Maven namespace
        ns_prefix = "{http://maven.apache.org/POM/4.0.0}"

        # Check if namespace is used
        use_ns = root.tag.startswith("{")
        if not use_ns:
            ns_prefix = ""

        # Search all build/plugins sections (including those in profiles)
        # Using .// to search recursively for all plugin elements
        for plugin in root.findall(f".//{ns_prefix}plugin" if use_ns else ".//plugin"):
            artifact_id = plugin.find(f"{ns_prefix}artifactId" if use_ns else "artifactId")
            if artifact_id is not None and artifact_id.text == "jacoco-maven-plugin":
                group_id = plugin.find(f"{ns_prefix}groupId" if use_ns else "groupId")
                # Verify groupId if present (it's optional for org.jacoco)
                if group_id is None or group_id.text == "org.jacoco":
                    return True

        return False

    except ET.ParseError as e:
        logger.warning("Failed to parse pom.xml for JaCoCo check: %s", e)
        return False


def add_jacoco_plugin_to_pom(pom_path: Path) -> bool:
    """Add JaCoCo Maven plugin to pom.xml for coverage collection.

    Uses string manipulation to preserve the original XML format and avoid
    namespace prefix issues that ElementTree causes.

    Args:
        pom_path: Path to the pom.xml file.

    Returns:
        True if plugin was added or already present, False on error.

    """
    if not pom_path.exists():
        logger.error("pom.xml not found: %s", pom_path)
        return False

    # Check if already configured
    if is_jacoco_configured(pom_path):
        logger.info("JaCoCo plugin already configured in pom.xml")
        return True

    try:
        content = pom_path.read_text(encoding="utf-8")

        # Basic validation that it's a Maven pom.xml
        if "</project>" not in content:
            logger.error("Invalid pom.xml: no closing </project> tag found")
            return False

        # JaCoCo plugin XML to insert (indented for typical pom.xml format)
        # Note: For multi-module projects where tests are in a separate module,
        # we configure the report to look in multiple directories for classes
        jacoco_plugin = f"""
      <plugin>
        <groupId>org.jacoco</groupId>
        <artifactId>jacoco-maven-plugin</artifactId>
        <version>{JACOCO_PLUGIN_VERSION}</version>
        <executions>
          <execution>
            <id>prepare-agent</id>
            <goals>
              <goal>prepare-agent</goal>
            </goals>
          </execution>
          <execution>
            <id>report</id>
            <phase>verify</phase>
            <goals>
              <goal>report</goal>
            </goals>
            <configuration>
              <!-- For multi-module projects, include dependency classes -->
              <includes>
                <include>**/*.class</include>
              </includes>
            </configuration>
          </execution>
        </executions>
      </plugin>"""

        # Find the main <build> section (not inside <profiles>)
        # We need to find a <build> that appears after </profiles> or before <profiles>
        # or if there's no profiles section at all
        profiles_start = content.find("<profiles>")
        profiles_end = content.find("</profiles>")

        # Find all <build> tags

        # Find the main build section - it's the one NOT inside profiles
        # Strategy: Look for <build> that comes after </profiles> or before <profiles> (or no profiles)
        if profiles_start == -1:
            # No profiles, any <build> is the main one
            build_start = content.find("<build>")
            build_end = content.find("</build>")
        else:
            # Has profiles - find <build> outside of profiles
            # Check for <build> before <profiles>
            build_before_profiles = content[:profiles_start].rfind("<build>")
            # Check for <build> after </profiles>
            build_after_profiles = content[profiles_end:].find("<build>") if profiles_end != -1 else -1
            if build_after_profiles != -1:
                build_after_profiles += profiles_end

            if build_before_profiles != -1:
                build_start = build_before_profiles
                # Find corresponding </build> - need to handle nested builds
                build_end = _find_closing_tag(content, build_start, "build")
            elif build_after_profiles != -1:
                build_start = build_after_profiles
                build_end = _find_closing_tag(content, build_start, "build")
            else:
                build_start = -1
                build_end = -1

        if build_start != -1 and build_end != -1:
            # Found main build section, find plugins within it
            build_section = content[build_start : build_end + len("</build>")]
            plugins_start_in_build = build_section.find("<plugins>")
            plugins_end_in_build = build_section.rfind("</plugins>")

            if plugins_start_in_build != -1 and plugins_end_in_build != -1:
                # Insert before </plugins> within the main build section
                absolute_plugins_end = build_start + plugins_end_in_build
                content = content[:absolute_plugins_end] + jacoco_plugin + "\n    " + content[absolute_plugins_end:]
            else:
                # No plugins section in main build, add one before </build>
                plugins_section = f"<plugins>{jacoco_plugin}\n    </plugins>\n  "
                content = content[:build_end] + plugins_section + content[build_end:]
        else:
            # No main build section found, add one before </project>
            project_end = content.rfind("</project>")
            build_section = f"""
  <build>
    <plugins>{jacoco_plugin}
    </plugins>
  </build>
"""
            content = content[:project_end] + build_section + content[project_end:]

        pom_path.write_text(content, encoding="utf-8")
        logger.info("Added JaCoCo plugin to pom.xml")
        return True

    except Exception as e:
        logger.exception("Failed to add JaCoCo plugin to pom.xml: %s", e)
        return False


def _find_closing_tag(content: str, start_pos: int, tag_name: str) -> int:
    """Find the position of the closing tag that matches the opening tag at start_pos.

    Handles nested tags of the same name.

    Args:
        content: The XML content.
        start_pos: Position of the opening tag.
        tag_name: Name of the tag.

    Returns:
        Position of the closing tag, or -1 if not found.

    """
    open_tag = f"<{tag_name}>"
    open_tag_short = f"<{tag_name} "  # For tags with attributes
    close_tag = f"</{tag_name}>"

    # Start searching after the opening tag we're matching
    depth = 1  # We've already found the opening tag at start_pos
    pos = start_pos + len(f"<{tag_name}")  # Move past the opening tag

    while pos < len(content):
        next_open = content.find(open_tag, pos)
        next_open_short = content.find(open_tag_short, pos)
        next_close = content.find(close_tag, pos)

        if next_close == -1:
            return -1

        # Find the earliest opening tag (if any)
        candidates = [x for x in [next_open, next_open_short] if x != -1 and x < next_close]
        next_open_any = min(candidates) if candidates else len(content) + 1

        if next_open_any < next_close:
            # Found opening tag first - nested tag
            depth += 1
            pos = next_open_any + 1
        else:
            # Found closing tag first
            depth -= 1
            if depth == 0:
                return next_close
            pos = next_close + len(close_tag)

    return -1


def get_jacoco_xml_path(project_root: Path) -> Path:
    """Get the expected path to the JaCoCo XML report.

    Args:
        project_root: Root directory of the Maven project.

    Returns:
        Path to the JaCoCo XML report file.

    """
    return project_root / "target" / "site" / "jacoco" / "jacoco.xml"


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


def get_classpath(project_root: Path) -> str | None:
    """Get the classpath for a Java project.

    For Maven projects, this runs 'mvn dependency:build-classpath'.

    Args:
        project_root: Root directory of the Java project.

    Returns:
        Classpath string, or None if unable to determine.

    """
    build_tool = detect_build_tool(project_root)

    if build_tool == BuildTool.MAVEN:
        return _get_maven_classpath(project_root)
    if build_tool == BuildTool.GRADLE:
        return _get_gradle_classpath(project_root)

    return None


def _get_maven_classpath(project_root: Path) -> str | None:
    """Get classpath from Maven."""
    mvn = find_maven_executable()
    if not mvn:
        return None

    try:
        result = subprocess.run(
            [mvn, "dependency:build-classpath", "-q", "-DincludeScope=test"],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # The classpath is in stdout
            return result.stdout.strip()

    except Exception as e:
        logger.warning("Failed to get Maven classpath: %s", e)

    return None


def _get_gradle_classpath(project_root: Path) -> str | None:
    """Get classpath from Gradle.

    Note: This requires a custom task to be added to build.gradle.
    Returns None for now as Gradle support is not fully implemented.
    """
    return None
