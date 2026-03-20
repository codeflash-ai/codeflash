"""Config writer for native config files.

This module writes Codeflash configuration to native config files:
- Python: pyproject.toml [tool.codeflash]
- JavaScript/TypeScript: package.json { "codeflash": {} }
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import tomlkit

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.setup.config_schema import CodeflashConfig
    from codeflash.setup.detector import DetectedProject


def write_config(detected: DetectedProject, config: CodeflashConfig | None = None) -> tuple[bool, str]:
    """Write Codeflash config to the appropriate native config file.

    Args:
        detected: DetectedProject with project information.
        config: Optional CodeflashConfig to write. If None, creates from detected.

    Returns:
        Tuple of (success, message).

    """
    from codeflash.setup.config_schema import CodeflashConfig

    if config is None:
        config = CodeflashConfig.from_detected_project(detected)

    if detected.language == "python":
        return _write_pyproject_toml(detected.project_root, config)
    if detected.language == "java":
        return _write_java_build_config(detected.project_root, config)
    return _write_package_json(detected.project_root, config)


def _write_pyproject_toml(project_root: Path, config: CodeflashConfig) -> tuple[bool, str]:
    """Write config to pyproject.toml [tool.codeflash] section.

    Creates pyproject.toml if it doesn't exist.
    Preserves existing content and formatting.

    Args:
        project_root: Project root directory.
        config: CodeflashConfig to write.

    Returns:
        Tuple of (success, message).

    """
    pyproject_path = project_root / "pyproject.toml"

    try:
        # Load existing or create new
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        # Ensure [tool] section exists
        if "tool" not in doc:
            doc["tool"] = tomlkit.table()

        # Create codeflash section
        codeflash_table = tomlkit.table()
        codeflash_table.add(tomlkit.comment("Codeflash configuration - https://docs.codeflash.ai"))

        # Add config values
        config_dict = config.to_pyproject_dict()
        for key, value in config_dict.items():
            codeflash_table[key] = value

        # Update the document
        doc["tool"]["codeflash"] = codeflash_table

        # Write back
        with pyproject_path.open("w", encoding="utf8") as f:
            f.write(tomlkit.dumps(doc))

        return True, f"Config saved to {pyproject_path}"

    except Exception as e:
        return False, f"Failed to write pyproject.toml: {e}"


def _write_java_build_config(project_root: Path, config: CodeflashConfig) -> tuple[bool, str]:
    """Write codeflash config to pom.xml properties or gradle.properties.

    Only writes non-default values. Standard Maven/Gradle layouts need no config.

    Args:
        project_root: Project root directory.
        config: CodeflashConfig to write.

    Returns:
        Tuple of (success, message).

    """
    config_dict = config.to_pyproject_dict()

    # Filter out default values — only write overrides
    defaults = {"module-root": "src/main/java", "tests-root": "src/test/java", "language": "java"}
    non_default = {k: v for k, v in config_dict.items() if k not in defaults or str(v) != defaults.get(k)}
    # Remove empty lists and False booleans
    non_default = {k: v for k, v in non_default.items() if v not in ([], False, "", None)}

    if not non_default:
        return True, "Standard Maven/Gradle layout detected — no config needed"

    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        return _write_maven_properties(pom_path, non_default)

    gradle_props_path = project_root / "gradle.properties"
    return _write_gradle_properties(gradle_props_path, non_default)


def _write_maven_properties(pom_path: Path, config: dict[str, Any]) -> tuple[bool, str]:
    """Add codeflash.* properties to pom.xml <properties> section."""
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(str(pom_path))
        root = tree.getroot()
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}

        # Find or create <properties>
        properties = root.find("m:properties", ns) or root.find("properties")
        if properties is None:
            properties = ET.SubElement(root, "properties")

        # Convert kebab-case keys to camelCase for Maven convention
        key_map = {
            "module-root": "moduleRoot",
            "tests-root": "testsRoot",
            "git-remote": "gitRemote",
            "disable-telemetry": "disableTelemetry",
            "ignore-paths": "ignorePaths",
            "formatter-cmds": "formatterCmds",
        }

        for key, value in config.items():
            maven_key = f"codeflash.{key_map.get(key, key)}"
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)

            existing = properties.find(maven_key)
            if existing is None:
                elem = ET.SubElement(properties, maven_key)
                elem.text = value
            else:
                existing.text = value

        tree.write(str(pom_path), xml_declaration=True, encoding="UTF-8")
        return True, f"Config saved to {pom_path} <properties>"

    except Exception as e:
        return False, f"Failed to write Maven properties: {e}"


def _write_gradle_properties(props_path: Path, config: dict[str, Any]) -> tuple[bool, str]:
    """Add codeflash.* entries to gradle.properties."""
    key_map = {
        "module-root": "moduleRoot",
        "tests-root": "testsRoot",
        "git-remote": "gitRemote",
        "disable-telemetry": "disableTelemetry",
        "ignore-paths": "ignorePaths",
        "formatter-cmds": "formatterCmds",
    }

    try:
        lines = []
        if props_path.exists():
            lines = props_path.read_text(encoding="utf-8").splitlines()

        # Remove existing codeflash.* lines
        lines = [line for line in lines if not line.strip().startswith("codeflash.")]

        # Add new config
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# Codeflash configuration — https://docs.codeflash.ai")
        for key, value in config.items():
            gradle_key = f"codeflash.{key_map.get(key, key)}"
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = str(value).lower()
            else:
                value = str(value)
            lines.append(f"{gradle_key}={value}")

        props_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True, f"Config saved to {props_path}"

    except Exception as e:
        return False, f"Failed to write gradle.properties: {e}"


def _write_package_json(project_root: Path, config: CodeflashConfig) -> tuple[bool, str]:
    """Write config to package.json codeflash section.

    Preserves existing content and formatting.
    Creates minimal config (only non-default values).

    Args:
        project_root: Project root directory.
        config: CodeflashConfig to write.

    Returns:
        Tuple of (success, message).

    """
    package_json_path = project_root / "package.json"

    if not package_json_path.exists():
        return False, f"No package.json found at {project_root}"

    try:
        # Load existing
        with package_json_path.open(encoding="utf8") as f:
            doc = json.load(f)

        # Get config dict (only non-default values)
        config_dict = config.to_package_json_dict()

        # Update or remove codeflash section
        if config_dict:
            doc["codeflash"] = config_dict
            action = "Updated"
        else:
            # Remove codeflash section if empty (all defaults)
            doc.pop("codeflash", None)
            action = "Using auto-detected defaults (no config needed)"

        # Write back with nice formatting
        with package_json_path.open("w", encoding="utf8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")  # Trailing newline

        if config_dict:
            return True, f"{action} config in {package_json_path}"
        return True, action

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON in package.json: {e}"
    except Exception as e:
        return False, f"Failed to write package.json: {e}"


def remove_config(project_root: Path, language: str) -> tuple[bool, str]:
    """Remove Codeflash config from native config file.

    Args:
        project_root: Project root directory.
        language: Project language ("python", "javascript", "typescript").

    Returns:
        Tuple of (success, message).

    """
    if language == "python":
        return _remove_from_pyproject(project_root)
    if language == "java":
        return _remove_java_build_config(project_root)
    return _remove_from_package_json(project_root)


def _remove_from_pyproject(project_root: Path) -> tuple[bool, str]:
    """Remove [tool.codeflash] section from pyproject.toml."""
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return True, "No pyproject.toml found"

    try:
        with pyproject_path.open("rb") as f:
            doc = tomlkit.parse(f.read())

        if "tool" in doc and "codeflash" in doc["tool"]:
            del doc["tool"]["codeflash"]

            with pyproject_path.open("w", encoding="utf8") as f:
                f.write(tomlkit.dumps(doc))

            return True, "Removed [tool.codeflash] section from pyproject.toml"

        return True, "No codeflash config found in pyproject.toml"

    except Exception as e:
        return False, f"Failed to remove config: {e}"


def _remove_java_build_config(project_root: Path) -> tuple[bool, str]:
    """Remove codeflash.* properties from pom.xml or gradle.properties."""
    # Try gradle.properties first (simpler)
    gradle_props = project_root / "gradle.properties"
    if gradle_props.exists():
        try:
            lines = gradle_props.read_text(encoding="utf-8").splitlines()
            filtered = [
                line
                for line in lines
                if not line.strip().startswith("codeflash.")
                and line.strip() != "# Codeflash configuration — https://docs.codeflash.ai"
            ]
            gradle_props.write_text("\n".join(filtered) + "\n", encoding="utf-8")
            return True, "Removed codeflash properties from gradle.properties"
        except Exception as e:
            return False, f"Failed to remove config from gradle.properties: {e}"

    # Try pom.xml
    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(str(pom_path))
            root = tree.getroot()
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}
            for properties in [root.find("m:properties", ns), root.find("properties")]:
                if properties is None:
                    continue
                to_remove = [child for child in properties if child.tag.split("}")[-1].startswith("codeflash.")]
                for elem in to_remove:
                    properties.remove(elem)
            tree.write(str(pom_path), xml_declaration=True, encoding="UTF-8")
            return True, "Removed codeflash properties from pom.xml"
        except Exception as e:
            return False, f"Failed to remove config from pom.xml: {e}"

    return True, "No Java build config found"


def _remove_from_package_json(project_root: Path) -> tuple[bool, str]:
    """Remove codeflash section from package.json."""
    package_json_path = project_root / "package.json"

    if not package_json_path.exists():
        return True, "No package.json found"

    try:
        with package_json_path.open(encoding="utf8") as f:
            doc = json.load(f)

        if "codeflash" in doc:
            del doc["codeflash"]

            with package_json_path.open("w", encoding="utf8") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")

            return True, "Removed codeflash section from package.json"

        return True, "No codeflash config found in package.json"

    except Exception as e:
        return False, f"Failed to remove config: {e}"
