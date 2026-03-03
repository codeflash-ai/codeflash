"""Config writer for native config files.

This module writes Codeflash configuration to native config files:
- Python: pyproject.toml [tool.codeflash]
- JavaScript/TypeScript: package.json { "codeflash": {} }
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
        return _write_codeflash_toml(detected.project_root, config)
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


def _write_codeflash_toml(project_root: Path, config: CodeflashConfig) -> tuple[bool, str]:
    """Write config to codeflash.toml [tool.codeflash] section for Java projects.

    Creates codeflash.toml if it doesn't exist.

    Args:
        project_root: Project root directory.
        config: CodeflashConfig to write.

    Returns:
        Tuple of (success, message).

    """
    codeflash_toml_path = project_root / "codeflash.toml"

    try:
        # Load existing or create new
        if codeflash_toml_path.exists():
            with codeflash_toml_path.open("rb") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        # Ensure [tool] section exists
        if "tool" not in doc:
            doc["tool"] = tomlkit.table()

        # Create codeflash section
        codeflash_table = tomlkit.table()
        codeflash_table.add(tomlkit.comment("Codeflash configuration for Java - https://docs.codeflash.ai"))

        # Add config values
        config_dict = config.to_pyproject_dict()
        for key, value in config_dict.items():
            codeflash_table[key] = value

        # Update the document
        doc["tool"]["codeflash"] = codeflash_table

        # Write back
        with codeflash_toml_path.open("w", encoding="utf8") as f:
            f.write(tomlkit.dumps(doc))

        return True, f"Config saved to {codeflash_toml_path}"

    except Exception as e:
        return False, f"Failed to write codeflash.toml: {e}"


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


def create_pyproject_toml(project_root: Path) -> tuple[bool, str]:
    """Create a minimal pyproject.toml file.

    Used when no pyproject.toml exists for a Python project.

    Args:
        project_root: Project root directory.

    Returns:
        Tuple of (success, message).

    """
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        return False, f"pyproject.toml already exists at {pyproject_path}"

    try:
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Created by Codeflash"))
        doc.add(tomlkit.nl())

        # Add minimal [tool.codeflash] section
        tool_table = tomlkit.table()
        codeflash_table = tomlkit.table()
        codeflash_table.add(tomlkit.comment("Codeflash configuration - https://docs.codeflash.ai"))
        tool_table["codeflash"] = codeflash_table
        doc["tool"] = tool_table

        with pyproject_path.open("w", encoding="utf8") as f:
            f.write(tomlkit.dumps(doc))

        return True, f"Created {pyproject_path}"

    except Exception as e:
        return False, f"Failed to create pyproject.toml: {e}"


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
        return _remove_from_codeflash_toml(project_root)
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


def _remove_from_codeflash_toml(project_root: Path) -> tuple[bool, str]:
    """Remove [tool.codeflash] section from codeflash.toml."""
    codeflash_toml_path = project_root / "codeflash.toml"

    if not codeflash_toml_path.exists():
        return True, "No codeflash.toml found"

    try:
        with codeflash_toml_path.open("rb") as f:
            doc = tomlkit.parse(f.read())

        if "tool" in doc and "codeflash" in doc["tool"]:
            del doc["tool"]["codeflash"]

            with codeflash_toml_path.open("w", encoding="utf8") as f:
                f.write(tomlkit.dumps(doc))

            return True, "Removed [tool.codeflash] section from codeflash.toml"

        return True, "No codeflash config found in codeflash.toml"

    except Exception as e:
        return False, f"Failed to remove config: {e}"


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
