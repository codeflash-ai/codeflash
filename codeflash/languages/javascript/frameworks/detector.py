"""Framework detection for JavaScript/TypeScript projects.

Detects React (and potentially other frameworks) by inspecting package.json
dependencies. Results are cached per project root.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameworkInfo:
    """Information about the frontend framework used in a project."""

    name: str  # "react", "vue", "angular", "none"
    version: str | None = None  # e.g., "18.2.0"
    react_version_major: int | None = None  # e.g., 18
    has_testing_library: bool = False  # @testing-library/react installed
    has_react_compiler: bool = False  # React 19+ compiler detected
    dev_dependencies: frozenset[str] = field(default_factory=frozenset)


_EMPTY_FRAMEWORK = FrameworkInfo(name="none")


@lru_cache(maxsize=32)
def detect_framework(project_root: Path) -> FrameworkInfo:
    """Detect the frontend framework from package.json.

    Reads dependencies and devDependencies to identify React and its ecosystem.
    Results are cached per project root path.
    """
    package_json_path = project_root / "package.json"
    if not package_json_path.exists():
        return _EMPTY_FRAMEWORK

    try:
        package_data = json.loads(package_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to read package.json at %s: %s", package_json_path, e)
        return _EMPTY_FRAMEWORK

    deps = package_data.get("dependencies", {})
    dev_deps = package_data.get("devDependencies", {})
    all_deps = {**deps, **dev_deps}

    # Detect React
    react_version_str = deps.get("react") or dev_deps.get("react")
    if not react_version_str:
        return _EMPTY_FRAMEWORK

    version = _parse_version_string(react_version_str)
    major = _parse_major_version(version)

    has_testing_library = "@testing-library/react" in all_deps
    has_react_compiler = (
        "babel-plugin-react-compiler" in all_deps
        or "react-compiler-runtime" in all_deps
        or (major is not None and major >= 19)
    )

    return FrameworkInfo(
        name="react",
        version=version,
        react_version_major=major,
        has_testing_library=has_testing_library,
        has_react_compiler=has_react_compiler,
        dev_dependencies=frozenset(all_deps.keys()),
    )


def _parse_version_string(version_spec: str) -> str | None:
    """Extract a clean version from a semver range like ^18.2.0 or ~17.0.0."""
    stripped = version_spec.lstrip("^~>=<! ")
    if stripped and stripped[0].isdigit():
        return stripped
    return None


def _parse_major_version(version: str | None) -> int | None:
    """Extract major version number from a version string."""
    if not version:
        return None
    try:
        return int(version.split(".")[0])
    except (ValueError, IndexError):
        return None
