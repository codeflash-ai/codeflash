from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoProjectConfig:
    project_root: Path
    module_path: str
    go_version: str | None = None
    has_vendor: bool = False


def detect_go_project(project_root: Path) -> GoProjectConfig | None:
    go_mod = project_root / "go.mod"
    if not go_mod.exists():
        return None

    module_path = ""
    go_version = None

    try:
        content = go_mod.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("module "):
                module_path = line[len("module ") :].strip()
            elif line.startswith("go "):
                go_version = line[len("go ") :].strip()
    except (OSError, UnicodeDecodeError):
        logger.warning("Failed to read go.mod at %s", go_mod)
        return None

    has_vendor = (project_root / "vendor").is_dir()

    return GoProjectConfig(
        project_root=project_root, module_path=module_path, go_version=go_version, has_vendor=has_vendor
    )


def detect_go_version() -> str | None:
    try:
        result = subprocess.run(["go", "version"], capture_output=True, text=True, timeout=10, check=False)
        if result.returncode != 0:
            return None
        match = re.search(r"go(\d+\.\d+(?:\.\d+)?)", result.stdout)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def is_go_project(project_root: Path) -> bool:
    if (project_root / "go.mod").exists():
        return True
    return any(project_root.glob("*.go"))
