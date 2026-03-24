"""Config writer for pyproject.toml.

This module writes Codeflash configuration to pyproject.toml [tool.codeflash].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tomlkit

from codeflash_core.danom import Err, Ok

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.danom import Result
    from codeflash_python.setup.config_schema import CodeflashConfig
    from codeflash_python.setup.detector import DetectedProject


def write_config(detected: DetectedProject, config: CodeflashConfig | None = None) -> Result[str, str]:
    """Write Codeflash config to pyproject.toml."""
    from codeflash_python.setup.config_schema import CodeflashConfig

    if config is None:
        config = CodeflashConfig.from_detected_project(detected)

    return write_pyproject_toml(detected.project_root, config)


def write_pyproject_toml(project_root: Path, config: CodeflashConfig) -> Result[str, str]:
    """Write config to pyproject.toml [tool.codeflash] section."""
    pyproject_path = project_root / "pyproject.toml"

    try:
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                doc = tomlkit.parse(f.read())
        else:
            doc = tomlkit.document()

        if "tool" not in doc:
            doc["tool"] = tomlkit.table()

        codeflash_table = tomlkit.table()
        codeflash_table.add(tomlkit.comment("Codeflash configuration - https://docs.codeflash.ai"))

        config_dict = config.to_pyproject_dict()
        for key, value in config_dict.items():
            codeflash_table[key] = value

        doc["tool"]["codeflash"] = codeflash_table  # type: ignore[index]

        with pyproject_path.open("w", encoding="utf8") as f:
            f.write(tomlkit.dumps(doc))

        return Ok(f"Config saved to {pyproject_path}")

    except Exception as e:
        return Err(f"Failed to write pyproject.toml: {e}")


def create_pyproject_toml(project_root: Path) -> Result[str, str]:
    """Create a minimal pyproject.toml file."""
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        return Err(f"pyproject.toml already exists at {pyproject_path}")

    try:
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Created by Codeflash"))
        doc.add(tomlkit.nl())

        tool_table = tomlkit.table()
        codeflash_table = tomlkit.table()
        codeflash_table.add(tomlkit.comment("Codeflash configuration - https://docs.codeflash.ai"))
        tool_table["codeflash"] = codeflash_table
        doc["tool"] = tool_table

        with pyproject_path.open("w", encoding="utf8") as f:
            f.write(tomlkit.dumps(doc))

        return Ok(f"Created {pyproject_path}")

    except Exception as e:
        return Err(f"Failed to create pyproject.toml: {e}")


def remove_config(project_root: Path) -> Result[str, str]:
    """Remove Codeflash config from pyproject.toml."""
    return remove_from_pyproject(project_root)


def remove_from_pyproject(project_root: Path) -> Result[str, str]:
    """Remove [tool.codeflash] section from pyproject.toml."""
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        return Ok("No pyproject.toml found")

    try:
        with pyproject_path.open("rb") as f:
            doc = tomlkit.parse(f.read())

        if "tool" in doc and "codeflash" in doc["tool"]:  # type: ignore[operator]
            del doc["tool"]["codeflash"]  # type: ignore[attr-defined]

            with pyproject_path.open("w", encoding="utf8") as f:
                f.write(tomlkit.dumps(doc))

            return Ok("Removed [tool.codeflash] section from pyproject.toml")

        return Ok("No codeflash config found in pyproject.toml")

    except Exception as e:
        return Err(f"Failed to remove config: {e}")
