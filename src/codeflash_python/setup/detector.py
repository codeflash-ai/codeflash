"""Python project detection engine for Codeflash.

Usage:
    from codeflash_python.setup import detect_project

    detected = detect_project()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomlkit

from codeflash_python.setup.detector_python import (
    detect_python_formatter,
    detect_python_module_root,
    detect_python_test_runner,
)


@dataclass
class DetectedProject:
    """Result of project auto-detection.

    All paths are absolute. The confidence score indicates how certain
    we are about the detection (0.0 = guessing, 1.0 = certain).
    """

    # Core detection results
    language: str
    project_root: Path
    module_root: Path
    tests_root: Path | None

    # Tooling detection
    test_runner: str
    formatter_cmds: list[str]

    # Ignore paths (absolute paths to ignore)
    ignore_paths: list[Path] = field(default_factory=list)

    # Confidence score for the detection (0.0 - 1.0)
    confidence: float = 0.8

    # Detection details (for debugging/display)
    detection_details: dict[str, str] = field(default_factory=dict)

    def to_display_dict(self) -> dict[str, str]:
        """Convert to dictionary for display purposes."""
        formatter_display = self.formatter_cmds[0] if self.formatter_cmds else "none detected"
        if len(self.formatter_cmds) > 1:
            formatter_display += f" (+{len(self.formatter_cmds) - 1} more)"

        ignore_display = ", ".join(p.name for p in self.ignore_paths[:3])
        if len(self.ignore_paths) > 3:
            ignore_display += f" (+{len(self.ignore_paths) - 3} more)"

        return {
            "Language": self.language.capitalize(),
            "Module root": str(self.module_root.relative_to(self.project_root))
            if self.module_root != self.project_root
            else ".",
            "Tests root": str(self.tests_root.relative_to(self.project_root)) if self.tests_root else "not detected",
            "Test runner": self.test_runner,
            "Formatter": formatter_display or "none",
            "Ignoring": ignore_display or "defaults only",
        }


def detect_project(path: Path | None = None) -> DetectedProject:
    """Auto-detect all project settings.

    This is the main entry point for project detection. It finds the project root,
    detects the language, and auto-detects all configuration values.

    Args:
        path: Starting path for detection. Defaults to current working directory.

    Returns:
        DetectedProject with all detected settings.

    Raises:
        ValueError: If no valid project can be detected.

    """
    start_path = path or Path.cwd()
    detection_details: dict[str, str] = {}

    # Step 1: Find project root
    project_root = find_project_root(start_path)
    if project_root is None:
        project_root = start_path
        detection_details["project_root"] = "using current directory (no markers found)"
    else:
        detection_details["project_root"] = f"found at {project_root}"

    detection_details["language"] = "python"

    # Step 2: Detect module root
    module_root, module_detail = detect_python_module_root(project_root)
    detection_details["module_root"] = module_detail

    # Step 3: Detect tests root
    tests_root, tests_detail = detect_tests_root(project_root)
    detection_details["tests_root"] = tests_detail

    # Step 4: Detect test runner
    test_runner, runner_detail = detect_python_test_runner(project_root)
    detection_details["test_runner"] = runner_detail

    # Step 5: Detect formatter
    formatter_cmds, formatter_detail = detect_python_formatter(project_root)
    detection_details["formatter"] = formatter_detail

    # Step 6: Detect ignore paths
    ignore_paths, ignore_detail = detect_ignore_paths(project_root)
    detection_details["ignore_paths"] = ignore_detail

    return DetectedProject(
        language="python",
        project_root=project_root,
        module_root=module_root,
        tests_root=tests_root,
        test_runner=test_runner,
        formatter_cmds=formatter_cmds,
        ignore_paths=ignore_paths,
        confidence=1.0,
        detection_details=detection_details,
    )


def find_project_root(start_path: Path) -> Path | None:
    """Find the project root by walking up the directory tree.

    Looks for:
    - .git directory (git repository root)
    - pyproject.toml (Python project)

    """
    current = start_path.resolve()

    while current != current.parent:
        markers = [".git", "pyproject.toml", "setup.py", "setup.cfg"]
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    return None


def detect_tests_root(project_root: Path) -> tuple[Path | None, str]:
    """Detect the tests directory."""
    for test_dir in ("tests", "test"):
        test_path = project_root / test_dir
        if test_path.is_dir():
            return test_path, f"{test_dir}/ directory"

    # Check if tests are alongside source
    test_files = list(project_root.glob("test_*.py"))
    if test_files:
        return project_root, "test files in project root"

    return None, "not detected"


def detect_ignore_paths(project_root: Path) -> tuple[list[Path], str]:
    """Detect paths to ignore during optimization."""
    ignore_paths: list[Path] = []
    sources: list[str] = []

    default_ignores = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "env",
        ".env",
        "dist",
        "build",
        ".egg-info",
        ".tox",
        ".nox",
        "htmlcov",
        ".coverage",
    ]

    for pattern in default_ignores:
        path = project_root / pattern
        if path.exists():
            ignore_paths.append(path)

    if ignore_paths:
        sources.append("defaults")

    # Parse .gitignore
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        try:
            content = gitignore_path.read_text(encoding="utf8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!"):
                    continue
                pattern = line.rstrip("/").lstrip("/")
                if "*" in pattern or "?" in pattern:
                    continue
                path = project_root / pattern
                if path.exists() and path not in ignore_paths:
                    ignore_paths.append(path)

            if ".gitignore" not in sources:
                sources.append(".gitignore")
        except Exception:
            pass

    detail = " + ".join(sources) if sources else "none"
    return ignore_paths, detail


def has_existing_config(project_root: Path) -> tuple[bool, str | None]:
    """Check if project has existing Codeflash configuration.

    Returns:
        Tuple of (has_config, config_file_type).
        config_file_type is "pyproject.toml", "codeflash.toml", or None.

    """
    for toml_filename in ("pyproject.toml", "codeflash.toml"):
        toml_path = project_root / toml_filename
        if toml_path.exists():
            try:
                with toml_path.open("rb") as f:
                    data = tomlkit.parse(f.read())
                if "tool" in data and "codeflash" in data["tool"]:  # type: ignore[unsupported-operator]
                    return True, toml_filename
            except Exception:
                pass

    return False, None
