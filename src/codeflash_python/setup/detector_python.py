"""Python-specific project detection functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tomlkit

if TYPE_CHECKING:
    from pathlib import Path


def detect_python_module_root(project_root: Path) -> tuple[Path, str]:
    """Detect Python module root.

    Priority:
    1. pyproject.toml [tool.poetry.name] or [project.name]
    2. src/ directory with __init__.py
    3. Directory with __init__.py matching project name
    4. src/ directory (even without __init__.py)
    5. Project root

    """
    # Try to get project name from pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    project_name = None

    if pyproject_path.exists():
        try:
            with pyproject_path.open("rb") as f:
                data = tomlkit.parse(f.read())

            # Try poetry name
            project_name = data.get("tool", {}).get("poetry", {}).get("name")
            # Try standard project name
            if not project_name:
                project_name = data.get("project", {}).get("name")
        except Exception:
            pass

    # Check for src layout
    src_dir = project_root / "src"
    if src_dir.is_dir():
        # Check for package inside src
        if project_name:
            pkg_dir = src_dir / project_name
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                return pkg_dir, f"src/{project_name}/ (from pyproject.toml name)"

        # Check for any package in src
        for child in src_dir.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                return child, f"src/{child.name}/ (first package in src)"

        # Use src/ even without __init__.py
        return src_dir, "src/ directory"

    # Check for package at project root
    if project_name:
        pkg_dir = project_root / project_name
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
            return pkg_dir, f"{project_name}/ (from pyproject.toml name)"

    # Look for any directory with __init__.py at project root
    for child in project_root.iterdir():
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and child.name not in ("tests", "test", "docs", "venv", ".venv", "env")
        ):
            if (child / "__init__.py").exists():
                return child, f"{child.name}/ (has __init__.py)"

    # Default to project root
    return project_root, "project root (no package structure detected)"


def detect_python_test_runner(project_root: Path) -> tuple[str, str]:
    """Detect Python test runner."""
    # Check for pytest markers
    pytest_markers = ["pytest.ini", "pyproject.toml", "conftest.py", "setup.cfg"]
    for marker in pytest_markers:
        marker_path = project_root / marker
        if marker_path.exists():
            if marker == "pyproject.toml":
                # Check for [tool.pytest] section
                try:
                    with marker_path.open("rb") as f:
                        data = tomlkit.parse(f.read())
                    if "tool" in data and "pytest" in data["tool"]:  # type: ignore[unsupported-operator]
                        return "pytest", "pyproject.toml [tool.pytest]"
                except Exception:
                    pass
            elif marker == "conftest.py":
                return "pytest", "conftest.py found"
            elif marker in ("pytest.ini", "setup.cfg"):
                # Check for pytest section in setup.cfg
                if marker == "setup.cfg":
                    try:
                        content = marker_path.read_text(encoding="utf8")
                        if "[tool:pytest]" in content or "[pytest]" in content:
                            return "pytest", "setup.cfg [pytest]"
                    except Exception:
                        pass
                else:
                    return "pytest", "pytest.ini found"

    # Default to pytest (most common)
    return "pytest", "default"


def detect_python_formatter(project_root: Path) -> tuple[list[str], str]:
    """Detect Python formatter."""
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        try:
            with pyproject_path.open("rb") as f:
                data = tomlkit.parse(f.read())

            tool = data.get("tool", {})

            # Check for ruff
            if "ruff" in tool:
                return ["ruff check --exit-zero --fix $file", "ruff format $file"], "from pyproject.toml [tool.ruff]"

            # Check for black
            if "black" in tool:
                return ["black $file"], "from pyproject.toml [tool.black]"
        except Exception:
            pass

    # Check for config files
    if (project_root / "ruff.toml").exists() or (project_root / ".ruff.toml").exists():
        return ["ruff check --exit-zero --fix $file", "ruff format $file"], "ruff.toml found"

    if (project_root / ".black").exists() or (project_root / "pyproject.toml").exists():
        # Default to black if pyproject.toml exists (common setup)
        return ["black $file"], "default (black)"

    return [], "none detected"
