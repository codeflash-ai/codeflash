"""Strategy pattern for Java build-config read/write/remove operations.

Defines BuildConfigStrategy ABC with MavenConfigStrategy (lxml-based pom.xml)
and GradleConfigStrategy (line-based gradle.properties) implementations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lxml import etree

logger = logging.getLogger(__name__)

MAVEN_NS = "http://maven.apache.org/POM/4.0.0"

# Maps kebab-case config keys to camelCase Maven/Gradle property names
_KEY_MAP: dict[str, str] = {
    "module-root": "moduleRoot",
    "tests-root": "testsRoot",
    "git-remote": "gitRemote",
    "disable-telemetry": "disableTelemetry",
    "ignore-paths": "ignorePaths",
    "formatter-cmds": "formatterCmds",
}


class BuildConfigStrategy(ABC):
    """Strategy interface for Java build-config read/write/remove operations."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def read_codeflash_properties(self, project_root: Path) -> dict[str, str]:
        """Read codeflash.* properties from the build file.

        Returns a dict mapping property suffix to value, e.g. {"moduleRoot": "src/main/java"}.
        """
        ...

    @abstractmethod
    def write_codeflash_properties(self, project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
        """Write codeflash.* properties to the build file. Only writes non-default overrides."""
        ...

    @abstractmethod
    def remove_codeflash_properties(self, project_root: Path) -> tuple[bool, str]:
        """Remove all codeflash.* properties from the build file."""
        ...


def _local_tag(element: etree._Element) -> str:
    """Strip namespace prefix from an lxml element tag to get the local name."""
    tag = element.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return str(tag)


def _make_tag(root: etree._Element, local_name: str) -> str:
    """Create a tag name respecting the document's default namespace."""
    ns = root.nsmap.get(None)
    if ns:
        return f"{{{ns}}}{local_name}"
    return local_name


def _find_element(parent: etree._Element, local_name: str) -> etree._Element | None:
    """Find a direct child element by local name, handling namespaces."""
    ns = parent.nsmap.get(None)
    if ns:
        return parent.find(f"{{{ns}}}{local_name}")
    return parent.find(local_name)


def _detect_child_indent(properties_elem: etree._Element) -> str:
    """Detect indentation used for children of a <properties> element."""
    for child in properties_elem:
        if isinstance(child.tag, str) and child.tail and "\n" in child.tail:
            # Indent is whitespace after the last newline in tail
            lines = child.tail.split("\n")
            if len(lines) > 1 and lines[-1].strip() == "":
                return lines[-1]
    # Try the element's own text (whitespace before first child)
    if properties_elem.text and "\n" in properties_elem.text:
        lines = properties_elem.text.split("\n")
        if len(lines) > 1:
            return lines[-1]
    return "        "  # 8-space default (typical Maven indent)


def _format_value(value: Any) -> str:
    """Convert a config value to a string suitable for build file properties."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


class MavenConfigStrategy(BuildConfigStrategy):
    """Read/write/remove codeflash.* properties in pom.xml using lxml."""

    @property
    def name(self) -> str:
        return "Maven"

    def read_codeflash_properties(self, project_root: Path) -> dict[str, str]:
        pom_path = project_root / "pom.xml"
        if not pom_path.exists():
            return {}
        try:
            _tree, root = self._parse_pom(pom_path)
            props = _find_element(root, "properties")
            if props is None:
                return {}
            result: dict[str, str] = {}
            for child in props:
                if not isinstance(child.tag, str):
                    continue
                local = _local_tag(child)
                if local.startswith("codeflash.") and child.text:
                    key = local[len("codeflash.") :]
                    result[key] = child.text.strip()
            return result
        except Exception:
            logger.debug("Failed to read codeflash properties from pom.xml", exc_info=True)
            return {}

    def write_codeflash_properties(self, project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
        pom_path = project_root / "pom.xml"
        if not pom_path.exists():
            return False, f"No pom.xml found at {project_root}"
        try:
            tree, root = self._parse_pom(pom_path)
            props = _find_element(root, "properties")

            if props is None:
                # Create <properties> section
                props = etree.SubElement(root, _make_tag(root, "properties"))
                props.text = "\n    "
                props.tail = "\n"
            else:
                # Remove existing codeflash.* elements
                for child in list(props):
                    if isinstance(child.tag, str) and _local_tag(child).startswith("codeflash."):
                        self._remove_preserving_whitespace(props, child)

            indent = _detect_child_indent(props)

            # Add new codeflash.* elements
            for key, value in config.items():
                prop_name = f"codeflash.{_KEY_MAP.get(key, key)}"
                tag = _make_tag(root, prop_name)
                elem = etree.SubElement(props, tag)
                elem.text = _format_value(value)
                elem.tail = "\n" + indent

            # Fix the last element's tail to align with the closing </properties>
            last = props[-1] if len(props) > 0 else None
            if last is not None:
                # Closing tag indent = one level less than child indent
                parent_indent = indent[:-4] if len(indent) >= 4 else indent[:-2] if len(indent) >= 2 else ""
                last.tail = "\n" + parent_indent

            # Ensure props.text has proper indent for first child
            if props.text is None or props.text.strip() == "":
                props.text = "\n" + indent

            tree.write(str(pom_path), xml_declaration=True, encoding="UTF-8")
            return True, f"Config saved to {pom_path} <properties>"
        except Exception as e:
            return False, f"Failed to write Maven properties: {e}"

    def remove_codeflash_properties(self, project_root: Path) -> tuple[bool, str]:
        pom_path = project_root / "pom.xml"
        if not pom_path.exists():
            return True, "No pom.xml found"
        try:
            tree, root = self._parse_pom(pom_path)
            props = _find_element(root, "properties")
            if props is None:
                return True, "No codeflash properties found in pom.xml"

            removed = False
            for child in list(props):
                if isinstance(child.tag, str) and _local_tag(child).startswith("codeflash."):
                    self._remove_preserving_whitespace(props, child)
                    removed = True

            if removed:
                tree.write(str(pom_path), xml_declaration=True, encoding="UTF-8")
            return True, "Removed codeflash properties from pom.xml"
        except Exception as e:
            return False, f"Failed to remove config from pom.xml: {e}"

    @staticmethod
    def _parse_pom(pom_path: Path) -> tuple[etree._ElementTree, etree._Element]:
        parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False)
        tree = etree.parse(str(pom_path), parser)
        return tree, tree.getroot()

    @staticmethod
    def _remove_preserving_whitespace(parent: etree._Element, child: etree._Element) -> None:
        """Remove a child element, merging its tail whitespace into the previous sibling or parent text."""
        prev = child.getprevious()
        if prev is not None:
            # Merge child's tail into previous sibling's tail
            prev.tail = (
                (prev.tail or "") if child.tail is None else (prev.tail or "").rstrip(" \t") + (child.tail or "")
            )
        # First child — merge tail into parent's text
        elif child.tail is not None:
            parent.text = (parent.text or "").rstrip(" \t") + child.tail
        parent.remove(child)


class GradleConfigStrategy(BuildConfigStrategy):
    """Read/write/remove codeflash.* properties in gradle.properties."""

    @property
    def name(self) -> str:
        return "Gradle"

    def read_codeflash_properties(self, project_root: Path) -> dict[str, str]:
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

    def write_codeflash_properties(self, project_root: Path, config: dict[str, Any]) -> tuple[bool, str]:
        props_path = project_root / "gradle.properties"
        try:
            lines: list[str] = []
            if props_path.exists():
                lines = props_path.read_text(encoding="utf-8").splitlines()

            # Remove existing codeflash.* lines and our comment header
            lines = [
                line
                for line in lines
                if not line.strip().startswith("codeflash.")
                and line.strip() != "# Codeflash configuration \u2014 https://docs.codeflash.ai"
            ]

            # Add blank line separator if needed
            if lines and lines[-1].strip():
                lines.append("")
            lines.append("# Codeflash configuration \u2014 https://docs.codeflash.ai")
            for key, value in config.items():
                gradle_key = f"codeflash.{_KEY_MAP.get(key, key)}"
                lines.append(f"{gradle_key}={_format_value(value)}")

            props_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True, f"Config saved to {props_path}"
        except Exception as e:
            return False, f"Failed to write gradle.properties: {e}"

    def remove_codeflash_properties(self, project_root: Path) -> tuple[bool, str]:
        props_path = project_root / "gradle.properties"
        if not props_path.exists():
            return True, "No gradle.properties found"
        try:
            lines = props_path.read_text(encoding="utf-8").splitlines()
            filtered = [
                line
                for line in lines
                if not line.strip().startswith("codeflash.")
                and line.strip() != "# Codeflash configuration \u2014 https://docs.codeflash.ai"
            ]
            props_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
            return True, "Removed codeflash properties from gradle.properties"
        except Exception as e:
            return False, f"Failed to remove config from gradle.properties: {e}"


def get_config_strategy(project_root: Path) -> BuildConfigStrategy:
    """Detect build tool and return the appropriate config strategy."""
    from codeflash.languages.java.build_tools import BuildTool, detect_build_tool

    build_tool = detect_build_tool(project_root)
    if build_tool == BuildTool.MAVEN:
        return MavenConfigStrategy()
    if build_tool == BuildTool.GRADLE:
        return GradleConfigStrategy()
    msg = f"No supported Java build tool found in {project_root}"
    raise ValueError(msg)


def parse_java_project_config(project_root: Path) -> dict[str, Any] | None:
    """Parse codeflash config from Maven/Gradle build files.

    Reads codeflash.* properties from pom.xml or gradle.properties,
    then fills in defaults from auto-detected build tool conventions.

    Returns None if no Java build tool is detected.
    """
    from codeflash.languages.java.build_tools import BuildTool, detect_build_tool, find_source_root, find_test_root

    build_tool = detect_build_tool(project_root)
    if build_tool == BuildTool.UNKNOWN:
        return None

    try:
        strategy = get_config_strategy(project_root)
        user_config = strategy.read_codeflash_properties(project_root)
    except ValueError:
        user_config = {}

    source_root = find_source_root(project_root)
    test_root = find_test_root(project_root)

    if build_tool == BuildTool.MAVEN:
        source_from_modules, test_from_modules = _detect_roots_from_maven_modules(project_root)
        if source_from_modules is not None:
            source_root = source_from_modules
        if test_from_modules is not None:
            test_root = test_from_modules

    default_source = project_root / "src" / "main" / "java"
    default_test = project_root / "src" / "test" / "java"
    config: dict[str, Any] = {
        "language": "java",
        "module_root": str(
            (project_root / user_config["moduleRoot"]).resolve()
            if "moduleRoot" in user_config
            else (source_root or (default_source if default_source.is_dir() else project_root))
        ),
        "tests_root": str(
            (project_root / user_config["testsRoot"]).resolve()
            if "testsRoot" in user_config
            else (test_root or (default_test if default_test.is_dir() else project_root))
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


def _detect_roots_from_maven_modules(project_root: Path) -> tuple[Path | None, Path | None]:
    """Scan Maven module pom.xml files for custom sourceDirectory/testSourceDirectory."""
    from codeflash.languages.java.build_tools import _safe_parse_xml

    pom_path = project_root / "pom.xml"
    if not pom_path.exists():
        return None, None

    try:
        tree = _safe_parse_xml(pom_path)
        root = tree.getroot()
        ns = {"m": MAVEN_NS}

        modules: list[str] = []
        for modules_elem in [root.find("m:modules", ns), root.find("modules")]:
            if modules_elem is not None:
                for mod in modules_elem:
                    if mod.text:
                        modules.append(mod.text.strip())

        if not modules:
            return None, None

        source_candidates: list[tuple[Path, int]] = []
        test_root: Path | None = None
        skip_modules = {"example", "examples", "benchmark", "benchmarks", "demo", "sample", "samples"}

        for module_name in modules:
            module_pom = project_root / module_name / "pom.xml"
            if not module_pom.exists():
                continue

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

        source_root = max(source_candidates, key=lambda x: x[1])[0] if source_candidates else None
        return source_root, test_root

    except Exception:
        return None, None
