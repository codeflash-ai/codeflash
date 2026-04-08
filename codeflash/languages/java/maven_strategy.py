"""Maven build tool strategy for Java projects.

Implements BuildToolStrategy for Maven-based projects, handling compilation,
classpath extraction, test execution via Surefire, and JaCoCo coverage.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from codeflash.languages.java.build_tool_strategy import BuildToolStrategy, module_to_dir
from codeflash.languages.java.build_tools import (
    CODEFLASH_RUNTIME_JAR_NAME,
    CODEFLASH_RUNTIME_VERSION,
    BuildTool,
    JavaProjectInfo,
    _safe_parse_xml,
)

_TARGET = "target"

logger = logging.getLogger(__name__)

# Skip validation/analysis plugins that reject generated instrumented files
_MAVEN_VALIDATION_SKIP_FLAGS = [
    "-Drat.skip=true",
    "-Dcheckstyle.skip=true",
    "-Ddisable.checks=true",
    "-Dcheckstyle.failOnViolation=false",
    "-Dcheckstyle.failsOnError=false",
    "-Dmaven-checkstyle-plugin.failsOnError=false",
    "-Dmaven-checkstyle-plugin.failOnViolation=false",
    "-Dspotbugs.skip=true",
    "-Dpmd.skip=true",
    "-Denforcer.skip=true",
    "-Djapicmp.skip=true",
    "-Derrorprone.skip=true",
    "-Dspotless.check.skip=true",
    "-Dspotless.apply.skip=true",
    "-Dmaven.compiler.failOnWarning=false",
    "-Dmaven.compiler.showWarnings=false",
]

# Cache for classpath strings — keyed on (maven_root, test_module).
_classpath_cache: dict[tuple[Path, str | None], str] = {}

# Cache for multi-module dependency installs — keyed on (maven_root, test_module).
_multimodule_deps_installed: set[tuple[Path, str]] = set()

JACOCO_PLUGIN_VERSION = "0.8.13"

GITHUB_RELEASE_URL = (
    "https://github.com/codeflash-ai/codeflash/releases/download"
    f"/runtime-v{CODEFLASH_RUNTIME_VERSION}/{CODEFLASH_RUNTIME_JAR_NAME}"
)

CODEFLASH_CACHE_DIR = Path.home() / ".cache" / "codeflash"

CODEFLASH_DEPENDENCY_SNIPPET = f"""\
        <dependency>
            <groupId>com.codeflash</groupId>
            <artifactId>codeflash-runtime</artifactId>
            <version>{CODEFLASH_RUNTIME_VERSION}</version>
            <scope>test</scope>
        </dependency>
    </dependencies>"""


def download_from_github_releases() -> Path | None:
    """Download codeflash-runtime JAR from GitHub Releases.

    Downloads to ~/.cache/codeflash/ and returns the path to the downloaded JAR.
    Returns None if the download fails (e.g., no release published yet, network error).
    """
    cache_jar = CODEFLASH_CACHE_DIR / CODEFLASH_RUNTIME_JAR_NAME
    if cache_jar.exists():
        logger.info("Found cached codeflash-runtime JAR: %s", cache_jar)
        return cache_jar

    try:
        CODEFLASH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading codeflash-runtime from GitHub Releases: %s", GITHUB_RELEASE_URL)
        urllib.request.urlretrieve(GITHUB_RELEASE_URL, cache_jar)  # noqa: S310
        logger.info("Downloaded codeflash-runtime to %s", cache_jar)
        return cache_jar
    except Exception as e:
        logger.debug("GitHub Releases download failed: %s", e)
        cache_jar.unlink(missing_ok=True)
        return None


def resolve_from_maven_central(maven_root: Path) -> bool:
    """Ask Maven to resolve codeflash-runtime from Maven Central.

    Downloads the JAR to ~/.m2/repository/ automatically.
    Returns True if Maven successfully resolved the artifact.
    """
    mvn = shutil.which("mvn")
    if not mvn:
        return False
    cmd = [
        mvn,
        "dependency:resolve",
        f"-Dartifact=com.codeflash:codeflash-runtime:{CODEFLASH_RUNTIME_VERSION}",
        "-B",
        "-q",
    ]
    try:
        result = subprocess.run(cmd, check=False, cwd=maven_root, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("Resolved codeflash-runtime %s from Maven Central", CODEFLASH_RUNTIME_VERSION)
            return True
        logger.debug("Maven Central resolution failed: %s", result.stderr)
        return False
    except Exception as e:
        logger.debug("Maven Central resolution error: %s", e)
        return False


def install_codeflash_runtime(project_root: Path, runtime_jar_path: Path, mvn: str | None = None) -> bool:
    if not mvn:
        mvn = shutil.which("mvn")
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
        f"-Dversion={CODEFLASH_RUNTIME_VERSION}",
        "-Dpackaging=jar",
        "-B",
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


# Properties set to "true" to enable skipping
_VALIDATION_SKIP_PROPERTIES_TRUE = [
    "checkstyle.skip",
    "disable.checks",
    "spotbugs.skip",
    "pmd.skip",
    "rat.skip",
    "enforcer.skip",
    "japicmp.skip",
]

# Properties set to "false" to disable failure on violations
_VALIDATION_SKIP_PROPERTIES_FALSE = [
    "checkstyle.failOnViolation",
    "checkstyle.failsOnError",
    "maven-checkstyle-plugin.failsOnError",
    "maven-checkstyle-plugin.failOnViolation",
]

# Plugin overrides that explicitly set <skip>true</skip> in the plugin <configuration>.
# This handles parent POMs with custom execution IDs that ignore skip properties.
_VALIDATION_PLUGIN_OVERRIDES = """\
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-checkstyle-plugin</artifactId>
                <configuration>
                    <skip>true</skip>
                    <failOnViolation>false</failOnViolation>
                    <failsOnError>false</failsOnError>
                </configuration>
            </plugin>
            <plugin>
                <groupId>com.github.spotbugs</groupId>
                <artifactId>spotbugs-maven-plugin</artifactId>
                <configuration><skip>true</skip></configuration>
            </plugin>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-pmd-plugin</artifactId>
                <configuration><skip>true</skip></configuration>
            </plugin>
"""


def inject_validation_skip_properties(pom_path: Path) -> bool:
    """Inject validation skip properties and plugin overrides into the POM.

    Two-layer approach:
    1. Properties — works when the plugin reads from standard property names.
    2. Plugin configuration overrides — handles parent POMs with custom execution
       bindings that ignore the standard skip properties.
    """
    if not pom_path.exists():
        return False

    try:
        content = pom_path.read_text(encoding="utf-8")

        if "<!-- codeflash-validation-skip -->" in content:
            return True

        props_lines = "".join(f"        <{p}>true</{p}>\n" for p in _VALIDATION_SKIP_PROPERTIES_TRUE)
        props_lines += "".join(f"        <{p}>false</{p}>\n" for p in _VALIDATION_SKIP_PROPERTIES_FALSE)

        # 1. Inject properties
        closing_idx = content.find("</properties>")
        if closing_idx != -1:
            content = content[:closing_idx] + props_lines + content[closing_idx:]
        else:
            project_close = content.rfind("</project>")
            if project_close == -1:
                logger.warning("No </project> tag found in %s", pom_path)
                return False
            content = (
                content[:project_close]
                + "    <properties>\n"
                + props_lines
                + "    </properties>\n"
                + content[project_close:]
            )

        # 2. Inject plugin configuration overrides
        plugins_block = (
            "    <!-- codeflash-validation-skip -->\n"
            "    <build>\n"
            "        <plugins>\n" + _VALIDATION_PLUGIN_OVERRIDES + "        </plugins>\n"
            "    </build>\n"
        )

        build_close = content.find("</build>")
        if build_close != -1:
            # Insert plugins before existing </build>
            plugins_close = content.find("</plugins>", 0, build_close)
            if plugins_close != -1:
                content = (
                    content[:plugins_close]
                    + "<!-- codeflash-validation-skip -->\n"
                    + _VALIDATION_PLUGIN_OVERRIDES
                    + content[plugins_close:]
                )
            else:
                content = (
                    content[:build_close]
                    + "    <!-- codeflash-validation-skip -->\n"
                    + "    <plugins>\n"
                    + _VALIDATION_PLUGIN_OVERRIDES
                    + "    </plugins>\n"
                    + content[build_close:]
                )
        else:
            project_close = content.rfind("</project>")
            if project_close != -1:
                content = content[:project_close] + plugins_block + content[project_close:]

        pom_path.write_text(content, encoding="utf-8")
        logger.info("Injected validation skip properties and plugin overrides into %s", pom_path)
        return True

    except Exception:
        logger.debug("Failed to inject validation skip properties into %s", pom_path, exc_info=True)
        return False


def add_codeflash_dependency(pom_path: Path) -> bool:
    if not pom_path.exists():
        return False

    try:
        content = pom_path.read_text(encoding="utf-8")

        if "codeflash-runtime" in content:

            def update_codeflash_dep(match: re.Match[str]) -> str:
                block: str = match.group(0)
                if "codeflash-runtime" not in block:
                    return block
                return (
                    "<dependency>\n"
                    "            <groupId>com.codeflash</groupId>\n"
                    "            <artifactId>codeflash-runtime</artifactId>\n"
                    f"            <version>{CODEFLASH_RUNTIME_VERSION}</version>\n"
                    "            <scope>test</scope>\n"
                    "        </dependency>"
                )

            updated = re.sub(r"<dependency>[\s\S]*?</dependency>", update_codeflash_dep, content)
            if updated != content:
                pom_path.write_text(updated, encoding="utf-8")
                logger.info("Updated codeflash-runtime dependency to version %s in pom.xml", CODEFLASH_RUNTIME_VERSION)
            else:
                logger.info("codeflash-runtime dependency already up to date in pom.xml")
            return True

        closing_tag = "</dependencies>"
        idx = content.find(closing_tag)
        if idx == -1:
            logger.warning("No </dependencies> tag found in pom.xml, cannot add dependency")
            return False

        new_content = content[:idx] + CODEFLASH_DEPENDENCY_SNIPPET
        new_content += content[idx + len(closing_tag) :]

        pom_path.write_text(new_content, encoding="utf-8")
        logger.info("Added codeflash-runtime dependency to pom.xml")
        return True

    except Exception as e:
        logger.exception("Failed to add dependency to pom.xml: %s", e)
        return False


def is_jacoco_configured(pom_path: Path) -> bool:
    if not pom_path.exists():
        return False

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()
        if root is None:
            return False

        ns_prefix = "{http://maven.apache.org/POM/4.0.0}"
        use_ns = root.tag.startswith("{")
        if not use_ns:
            ns_prefix = ""

        for plugin in root.findall(f".//{ns_prefix}plugin" if use_ns else ".//plugin"):
            artifact_id = plugin.find(f"{ns_prefix}artifactId" if use_ns else "artifactId")
            if artifact_id is not None and artifact_id.text == "jacoco-maven-plugin":
                group_id = plugin.find(f"{ns_prefix}groupId" if use_ns else "groupId")
                if group_id is None or group_id.text == "org.jacoco":
                    return True

        return False

    except ET.ParseError as e:
        logger.warning("Failed to parse pom.xml for JaCoCo check: %s", e)
        return False


def _find_closing_tag(content: str, start_pos: int, tag_name: str) -> int:
    open_tag = f"<{tag_name}>"
    open_tag_short = f"<{tag_name} "
    close_tag = f"</{tag_name}>"

    depth = 1
    pos = start_pos + len(f"<{tag_name}")

    while pos < len(content):
        next_open = content.find(open_tag, pos)
        next_open_short = content.find(open_tag_short, pos)
        next_close = content.find(close_tag, pos)

        if next_close == -1:
            return -1

        candidates = [x for x in [next_open, next_open_short] if x != -1 and x < next_close]
        next_open_any = min(candidates) if candidates else len(content) + 1

        if next_open_any < next_close:
            depth += 1
            pos = next_open_any + 1
        else:
            depth -= 1
            if depth == 0:
                return next_close
            pos = next_close + len(close_tag)

    return -1


def add_jacoco_plugin(pom_path: Path) -> bool:
    if not pom_path.exists():
        logger.error("pom.xml not found: %s", pom_path)
        return False

    if is_jacoco_configured(pom_path):
        logger.info("JaCoCo plugin already configured in pom.xml")
        return True

    try:
        content = pom_path.read_text(encoding="utf-8")

        if "</project>" not in content:
            logger.error("Invalid pom.xml: no closing </project> tag found")
            return False

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

        profiles_start = content.find("<profiles>")
        profiles_end = content.find("</profiles>")

        if profiles_start == -1:
            build_start = content.find("<build>")
            build_end = content.find("</build>")
        else:
            build_before_profiles = content[:profiles_start].rfind("<build>")
            build_after_profiles = content[profiles_end:].find("<build>") if profiles_end != -1 else -1
            if build_after_profiles != -1:
                build_after_profiles += profiles_end

            if build_before_profiles != -1:
                build_start = build_before_profiles
                build_end = _find_closing_tag(content, build_start, "build")
            elif build_after_profiles != -1:
                build_start = build_after_profiles
                build_end = _find_closing_tag(content, build_start, "build")
            else:
                build_start = -1
                build_end = -1

        if build_start != -1 and build_end != -1:
            build_section = content[build_start : build_end + len("</build>")]
            plugins_start_in_build = build_section.find("<plugins>")
            plugins_end_in_build = build_section.rfind("</plugins>")

            if plugins_start_in_build != -1 and plugins_end_in_build != -1:
                absolute_plugins_end = build_start + plugins_end_in_build
                content = content[:absolute_plugins_end] + jacoco_plugin + "\n    " + content[absolute_plugins_end:]
            else:
                plugins_section = f"<plugins>{jacoco_plugin}\n    </plugins>\n  "
                content = content[:build_end] + plugins_section + content[build_end:]
        else:
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


def get_jacoco_report_path(project_root: Path) -> Path:
    return project_root / "target" / "site" / "jacoco" / "jacoco.xml"


def _extract_java_version_from_pom(root: ET.Element, ns: dict[str, str]) -> str | None:
    """Extract Java version from Maven pom.xml properties or compiler plugin."""
    for prop_name in ("maven.compiler.source", "java.version", "maven.compiler.release"):
        for props in [root.find("m:properties", ns), root.find("properties")]:
            if props is not None:
                for prop in [props.find(f"m:{prop_name}", ns), props.find(prop_name)]:
                    if prop is not None and prop.text:
                        return prop.text

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


def _discover_maven_submodule_roots(
    project_root: Path, root: ET.Element, ns: dict[str, str]
) -> tuple[list[Path], list[Path]]:
    """Discover source and test roots from Maven submodules."""
    source_roots: list[Path] = []
    test_roots: list[Path] = []

    modules: list[str] = []
    for modules_elem in [root.find("m:modules", ns), root.find("modules")]:
        if modules_elem is not None:
            for mod in modules_elem:
                if mod.text:
                    modules.append(mod.text.strip())

    for module_name in modules:
        module_dir = project_root / module_name
        if not module_dir.is_dir():
            continue

        std_src = module_dir / "src" / "main" / "java"
        if std_src.exists():
            source_roots.append(std_src)

        std_test = module_dir / "src" / "test" / "java"
        if std_test.exists():
            test_roots.append(std_test)

        module_pom = module_dir / "pom.xml"
        if module_pom.exists():
            try:
                mod_tree = _safe_parse_xml(module_pom)
                mod_root = mod_tree.getroot()
                if mod_root is None:
                    continue
                for build in [mod_root.find("m:build", ns), mod_root.find("build")]:
                    if build is not None:
                        for tag, roots_list in [("sourceDirectory", source_roots), ("testSourceDirectory", test_roots)]:
                            for elem in [build.find(f"m:{tag}", ns), build.find(tag)]:
                                if elem is not None and elem.text:
                                    custom_dir = module_dir / elem.text.strip()
                                    if custom_dir.exists() and custom_dir not in roots_list:
                                        roots_list.append(custom_dir)
            except Exception:
                continue

    return source_roots, test_roots


class MavenStrategy(BuildToolStrategy):
    """Maven-specific build tool operations."""

    _M2_JAR = (
        Path.home()
        / ".m2"
        / "repository"
        / "com"
        / "codeflash"
        / "codeflash-runtime"
        / "1.0.1"
        / "codeflash-runtime-1.0.1.jar"
    )

    @property
    def name(self) -> str:
        return "Maven"

    def get_project_info(self, project_root: Path) -> JavaProjectInfo | None:
        pom_path = project_root / "pom.xml"
        if not pom_path.exists():
            return None

        try:
            tree = _safe_parse_xml(pom_path)
            root = tree.getroot()
            if root is None:
                return None
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}

            def get_text(xpath: str, default: str | None = None) -> str | None:
                elem = root.find(f"m:{xpath}", ns)
                if elem is None:
                    elem = root.find(xpath)
                return elem.text if elem is not None else default

            group_id = get_text("groupId")
            artifact_id = get_text("artifactId")
            version = get_text("version")
            java_version = _extract_java_version_from_pom(root, ns)

            source_roots: list[Path] = []
            test_roots: list[Path] = []

            main_src = project_root / "src" / "main" / "java"
            if main_src.exists():
                source_roots.append(main_src)

            test_src = project_root / "src" / "test" / "java"
            if test_src.exists():
                test_roots.append(test_src)

            for build in [root.find("m:build", ns), root.find("build")]:
                if build is not None:
                    for tag, roots_list in [("sourceDirectory", source_roots), ("testSourceDirectory", test_roots)]:
                        for elem in [build.find(f"m:{tag}", ns), build.find(tag)]:
                            if elem is not None and elem.text:
                                custom_dir = project_root / elem.text.strip()
                                if custom_dir.exists() and custom_dir not in roots_list:
                                    roots_list.append(custom_dir)

            sub_sources, sub_tests = _discover_maven_submodule_roots(project_root, root, ns)
            for root_path in sub_sources:
                if root_path not in source_roots:
                    source_roots.append(root_path)
            for root_path in sub_tests:
                if root_path not in test_roots:
                    test_roots.append(root_path)

            return JavaProjectInfo(
                project_root=project_root,
                build_tool=BuildTool.MAVEN,
                source_roots=source_roots,
                test_roots=test_roots,
                target_dir=project_root / "target",
                group_id=group_id,
                artifact_id=artifact_id,
                version=version,
                java_version=java_version,
            )

        except ET.ParseError as e:
            logger.warning("Failed to parse pom.xml: %s", e)
            return None

    def find_executable(self, build_root: Path) -> str | None:
        return self.find_wrapper_executable(build_root, ("mvnw", "mvnw.cmd"), "mvn")

    def find_runtime_jar(self) -> Path | None:
        if self._M2_JAR.exists():
            return self._M2_JAR
        return super().find_runtime_jar()

    def ensure_runtime(self, build_root: Path, test_module: str | None) -> bool:
        if not self._M2_JAR.exists():
            if resolve_from_maven_central(build_root):
                logger.info("Resolved codeflash-runtime from Maven Central")
            else:
                runtime_jar = self.find_runtime_jar()
                if runtime_jar is None:
                    runtime_jar = download_from_github_releases()
                if runtime_jar is None:
                    logger.error(
                        "codeflash-runtime JAR not found. Maven Central resolution failed and "
                        "GitHub Releases download failed. Generated tests will fail to compile."
                    )
                    return False
                logger.info("Installing codeflash-runtime JAR to local Maven repository from %s", runtime_jar)
                if not install_codeflash_runtime(build_root, runtime_jar, mvn=self.find_executable(build_root)):
                    logger.error("Failed to install codeflash-runtime to local Maven repository")
                    return False

        if test_module:
            pom_path = build_root / module_to_dir(test_module) / "pom.xml"
        else:
            pom_path = build_root / "pom.xml"

        if pom_path.exists():
            if not add_codeflash_dependency(pom_path):
                logger.error("Failed to add codeflash-runtime dependency to %s", pom_path)
                return False
            inject_validation_skip_properties(pom_path)
        else:
            logger.warning("pom.xml not found at %s, cannot add codeflash-runtime dependency", pom_path)
            return False

        return True

    def install_multi_module_deps(self, build_root: Path, test_module: str | None, env: dict[str, str]) -> bool:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        if not test_module:
            return True

        cache_key = (build_root, test_module)
        if cache_key in _multimodule_deps_installed:
            logger.debug("Multi-module deps already installed for %s:%s, skipping", build_root, test_module)
            return True

        mvn = self.find_executable(build_root)
        if not mvn:
            logger.error("Maven not found — cannot pre-install multi-module dependencies")
            return False

        cmd = [mvn, "install", "-DskipTests", "-B", "-pl", module_to_dir(test_module), "-am"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        logger.info("Pre-installing multi-module dependencies: %s (module: %s)", build_root, test_module)
        logger.debug("Running: %s", " ".join(cmd))

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=300)
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
        logger.info("Multi-module dependencies installed successfully for %s:%s", build_root, test_module)
        return True

    def compile_tests(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        mvn = self.find_executable(build_root)
        if not mvn:
            logger.error("Maven not found")
            return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

        cmd = [mvn, "test-compile", "-e", "-B"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        if test_module:
            cmd.extend(["-pl", module_to_dir(test_module)])

        logger.debug("Compiling tests: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Maven compilation failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def compile_source_only(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 120
    ) -> subprocess.CompletedProcess[str]:
        from codeflash.languages.java.test_runner import _run_cmd_kill_pg_on_timeout

        mvn = self.find_executable(build_root)
        if not mvn:
            logger.error("Maven not found")
            return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

        cmd = [mvn, "compile", "-e", "-B"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        if test_module:
            cmd.extend(["-pl", module_to_dir(test_module)])

        logger.debug("Compiling source only: %s in %s", " ".join(cmd), build_root)

        try:
            return _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)
        except Exception as e:
            logger.exception("Maven source compilation failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

    def get_classpath(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        key = (build_root, test_module)
        cached = _classpath_cache.get(key)
        if cached is not None:
            logger.debug("Using cached classpath for (%s, %s)", build_root, test_module)
            return cached
        result = self._get_classpath_uncached(build_root, env, test_module, timeout)
        if result is not None:
            _classpath_cache[key] = result
        return result

    def _get_classpath_uncached(
        self, build_root: Path, env: dict[str, str], test_module: str | None, timeout: int = 60
    ) -> str | None:
        from codeflash.languages.java.test_runner import _find_junit_console_standalone, _run_cmd_kill_pg_on_timeout

        mvn = self.find_executable(build_root)
        if not mvn:
            return None

        cp_file = build_root / ".codeflash_classpath.txt"

        cmd = [mvn, "dependency:build-classpath", "-DincludeScope=test", f"-Dmdep.outputFile={cp_file}", "-q", "-B"]

        if test_module:
            cmd.extend(["-pl", module_to_dir(test_module)])

        logger.debug("Getting classpath: %s", " ".join(cmd))

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
                logger.error("Failed to get classpath: %s", result.stderr)
                return None

            if not cp_file.exists():
                logger.error("Classpath file not created")
                return None

            classpath = cp_file.read_text(encoding="utf-8").strip()

            if test_module:
                module_path = build_root / module_to_dir(test_module)
            else:
                module_path = build_root

            test_classes = module_path / "target" / "test-classes"
            main_classes = module_path / "target" / "classes"

            cp_parts = [classpath]
            if test_classes.exists():
                cp_parts.append(str(test_classes))
            if main_classes.exists():
                cp_parts.append(str(main_classes))

            if test_module:
                module_dir_name = module_to_dir(test_module)
                for module_dir in build_root.iterdir():
                    if module_dir.is_dir() and module_dir.name != module_dir_name:
                        module_classes = module_dir / "target" / "classes"
                        if module_classes.exists():
                            logger.debug("Adding multi-module classpath: %s", module_classes)
                            cp_parts.append(str(module_classes))

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
            if cp_file.exists():
                cp_file.unlink()

    def get_reports_dir(self, build_root: Path, test_module: str | None) -> Path:
        target_dir = self.get_build_output_dir(build_root, test_module)
        return target_dir / "surefire-reports"

    def get_build_output_dir(self, build_root: Path, test_module: str | None) -> Path:
        if test_module:
            return build_root.joinpath(module_to_dir(test_module), _TARGET)
        return build_root.joinpath(_TARGET)

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
        from codeflash.languages.java.test_runner import (
            _build_test_filter,
            _run_cmd_kill_pg_on_timeout,
            _validate_test_filter,
        )

        mvn = self.find_executable(build_root)
        if not mvn:
            logger.error("Maven not found")
            return subprocess.CompletedProcess(args=["mvn"], returncode=-1, stdout="", stderr="Maven not found")

        test_filter = _build_test_filter(test_paths, mode=mode)
        logger.debug("Built test filter for mode=%s: '%s' (empty=%s)", mode, test_filter, not test_filter)

        maven_goal = "verify" if enable_coverage else "test"
        cmd = [mvn, maven_goal, "-fae", "-B"]
        cmd.extend(_MAVEN_VALIDATION_SKIP_FLAGS)

        add_opens_flags = (
            "--add-opens java.base/java.util=ALL-UNNAMED"
            " --add-opens java.base/java.lang=ALL-UNNAMED"
            " --add-opens java.base/java.lang.reflect=ALL-UNNAMED"
            " --add-opens java.base/java.io=ALL-UNNAMED"
            " --add-opens java.base/java.math=ALL-UNNAMED"
            " --add-opens java.base/java.net=ALL-UNNAMED"
            " --add-opens java.base/java.util.zip=ALL-UNNAMED"
        )
        if enable_coverage:
            # When coverage is enabled, JaCoCo's prepare-agent goal sets argLine via
            # @{argLine}. Overriding -DargLine would clobber the JaCoCo agent flag.
            # Pass add-opens and javaagent via JDK_JAVA_OPTIONS instead.
            jdk_opts_parts = [add_opens_flags]
            if javaagent_arg:
                jdk_opts_parts.insert(0, javaagent_arg)
            env["JDK_JAVA_OPTIONS"] = " ".join(jdk_opts_parts)
        elif javaagent_arg:
            cmd.append(f"-DargLine={javaagent_arg} {add_opens_flags}")
        else:
            cmd.append(f"-DargLine={add_opens_flags}")

        if mode == "performance":
            cmd.append("-Dsurefire.useFile=false")

        if enable_coverage:
            cmd.append("-Dmaven.test.failure.ignore=true")

        if test_module:
            cmd.extend(
                [
                    "-pl",
                    module_to_dir(test_module),
                    "-DfailIfNoTests=false",
                    "-Dsurefire.failIfNoSpecifiedTests=false",
                    "-DskipTests=false",
                ]
            )

        if test_filter:
            validated_filter = _validate_test_filter(test_filter)
            cmd.append(f"-Dtest={validated_filter}")
            logger.debug("Added -Dtest=%s to Maven command", validated_filter)
        else:
            error_msg = (
                f"Test filter is EMPTY for mode={mode}! "
                f"Maven will run ALL tests instead of the specified tests. "
                f"This indicates a problem with test file instrumentation or path resolution."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug("Running Maven command: %s in %s", " ".join(cmd), build_root)

        try:
            result = _run_cmd_kill_pg_on_timeout(cmd, cwd=build_root, env=env, timeout=timeout)

            if result.returncode != 0:
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
                    output_lines = combined_output.split("\n")
                    error_context = "\n".join(output_lines[:50]) if len(output_lines) > 50 else combined_output
                    logger.error("Maven compilation error output:\n%s", error_context)

            return result

        except Exception as e:
            logger.exception("Maven test execution failed: %s", e)
            return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr=str(e))

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
        import time

        from codeflash.languages.java.test_runner import _find_multi_module_root, _get_combined_junit_xml

        project_root = project_root or cwd
        maven_root, test_module = _find_multi_module_root(project_root, test_paths)

        all_stdout: list[str] = []
        all_stderr: list[str] = []
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

            result = self.run_tests_via_build_tool(
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

        reports_dir = self.get_reports_dir(maven_root, test_module)
        result_xml_path = _get_combined_junit_xml(reports_dir, -1)

        return result_xml_path, combined_result

    def run_tests_with_coverage(
        self,
        build_root: Path,
        test_module: str | None,
        test_paths: Any,
        run_env: dict[str, str],
        timeout: int,
        candidate_index: int,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path | None]:
        from codeflash.languages.java.test_runner import _get_combined_junit_xml

        coverage_xml_path = self.setup_coverage(build_root, test_module, build_root)

        result = self.run_tests_via_build_tool(
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

        return result, result_xml_path, coverage_xml_path

    def setup_coverage(self, build_root: Path, test_module: str | None, project_root: Path) -> Path | None:
        if test_module:
            test_module_pom = build_root / module_to_dir(test_module) / "pom.xml"
            if test_module_pom.exists():
                if not is_jacoco_configured(test_module_pom):
                    logger.info("Adding JaCoCo plugin to test module pom.xml: %s", test_module_pom)
                    add_jacoco_plugin(test_module_pom)
                return get_jacoco_report_path(build_root / module_to_dir(test_module))
        else:
            pom_path = project_root / "pom.xml"
            if pom_path.exists():
                if not is_jacoco_configured(pom_path):
                    logger.info("Adding JaCoCo plugin to pom.xml for coverage collection")
                    add_jacoco_plugin(pom_path)
                return get_jacoco_report_path(project_root)
        return None

    def get_test_run_command(self, project_root: Path, test_classes: list[str] | None = None) -> list[str]:
        from codeflash.languages.java.test_runner import _validate_java_class_name

        if test_classes:
            for test_class in test_classes:
                if not _validate_java_class_name(test_class):
                    msg = f"Invalid test class name: '{test_class}'. Test names must follow Java identifier rules."
                    raise ValueError(msg)

        mvn = self.find_executable(project_root) or "mvn"
        cmd = [mvn, "test", "-B"]
        if test_classes:
            cmd.append(f"-Dtest={','.join(test_classes)}")
        return cmd
