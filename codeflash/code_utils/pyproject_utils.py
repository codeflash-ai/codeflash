from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tomlkit

DEFAULT_MINIMAL_PROJECT_VERSION = "0.0.0"


def infer_minimal_project_name(project_root: Path) -> str:
    """Generate a PEP 621-friendly project name from the directory name."""
    candidate = project_root.resolve().name.strip().lower()
    if not candidate:
        return "codeflash-project"

    normalized = re.sub(r"[^a-z0-9._-]+", "-", candidate)
    normalized = re.sub(r"[-_.]{2,}", "-", normalized).strip("-_.")
    return normalized or "codeflash-project"


def ensure_minimal_project_metadata(pyproject_data: Any, project_root: Path) -> bool:
    """Add minimal [project] metadata when the file has no standard project section.

    Codeflash can create a pyproject.toml solely for tool configuration. Some tooling,
    including uv in certain flows, expects PEP 621 metadata once a pyproject.toml exists.
    We only inject minimal metadata when neither [project] nor [tool.poetry] exists.
    """

    if "project" in pyproject_data:
        return False

    tool_section = pyproject_data.get("tool", {})
    if isinstance(tool_section, dict) and "poetry" in tool_section:
        return False

    project_section = tomlkit.table()
    project_section["name"] = infer_minimal_project_name(project_root)
    project_section["version"] = DEFAULT_MINIMAL_PROJECT_VERSION
    pyproject_data["project"] = project_section
    return True
