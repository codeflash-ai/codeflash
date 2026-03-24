"""Codeflash configuration schema using Pydantic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from pathlib import Path


class CodeflashConfig(BaseModel):
    """Internal representation of Codeflash configuration.

    This is the canonical config format used internally. It can be converted
    to/from pyproject.toml [tool.codeflash] format.

    Note: All paths are stored as strings (relative to project root).
    """

    # Core settings (always present after detection)
    language: str = Field(default="python", description="Project language")
    module_root: str = Field(default=".", description="Root directory containing source code")
    tests_root: str | None = Field(default=None, description="Root directory containing tests")

    # Tooling settings (auto-detected, can be overridden)
    test_framework: str | None = Field(default=None, description="Test framework: pytest")
    formatter_cmds: list[str] = Field(default_factory=list, description="Formatter commands")

    # Optional settings
    ignore_paths: list[str] = Field(default_factory=list, description="Paths to ignore")
    benchmarks_root: str | None = Field(default=None, description="Benchmarks directory")

    # Git settings
    git_remote: str = Field(default="origin", description="Git remote for PRs")

    # Privacy settings
    disable_telemetry: bool = Field(default=False, description="Disable telemetry")

    # Python-specific settings
    pytest_cmd: str = Field(default="pytest", description="Pytest command")
    disable_imports_sorting: bool = Field(default=False, description="Disable import sorting")
    override_fixtures: bool = Field(default=False, description="Override test fixtures")

    model_config = ConfigDict(extra="allow")  # Allow extra fields for forward compatibility

    def to_pyproject_dict(self) -> dict[str, Any]:
        """Convert to pyproject.toml [tool.codeflash] format.

        Uses kebab-case keys as per TOML conventions.
        Only includes non-default values to keep config minimal.
        """
        config: dict[str, Any] = {}

        # Always include required fields
        config["module-root"] = self.module_root
        if self.tests_root:
            config["tests-root"] = self.tests_root

        # Include non-default optional fields
        if self.ignore_paths:
            config["ignore-paths"] = self.ignore_paths

        if self.formatter_cmds and self.formatter_cmds != ["black $file"]:
            config["formatter-cmds"] = self.formatter_cmds
        elif not self.formatter_cmds:
            config["formatter-cmds"] = ["disabled"]

        if self.benchmarks_root:
            config["benchmarks-root"] = self.benchmarks_root

        if self.git_remote and self.git_remote != "origin":
            config["git-remote"] = self.git_remote

        if self.disable_telemetry:
            config["disable-telemetry"] = True

        if self.pytest_cmd and self.pytest_cmd != "pytest":
            config["pytest-cmd"] = self.pytest_cmd

        if self.disable_imports_sorting:
            config["disable-imports-sorting"] = True

        if self.override_fixtures:
            config["override-fixtures"] = True

        return config

    @classmethod
    def from_detected_project(cls, detected: Any) -> CodeflashConfig:
        """Create config from DetectedProject."""
        return cls(
            language="python",
            module_root=str(detected.module_root.relative_to(detected.project_root))
            if detected.module_root != detected.project_root
            else ".",
            tests_root=str(detected.tests_root.relative_to(detected.project_root)) if detected.tests_root else None,
            test_framework=detected.test_runner,
            formatter_cmds=detected.formatter_cmds,
            ignore_paths=[
                str(p.relative_to(detected.project_root)) for p in detected.ignore_paths if p != detected.project_root
            ],
            pytest_cmd=detected.test_runner,
        )

    @classmethod
    def from_pyproject_dict(cls, data: dict[str, Any], project_root: Path | None = None) -> CodeflashConfig:
        """Create config from pyproject.toml [tool.codeflash] section."""
        _ = project_root  # Reserved for future path resolution

        def convert_key(key: str) -> str:
            """Convert kebab-case to snake_case."""
            return key.replace("-", "_")

        converted = {convert_key(k): v for k, v in data.items()}
        converted.setdefault("language", "python")
        return cls(**converted)
