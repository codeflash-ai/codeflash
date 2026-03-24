"""Setup module for Codeflash auto-detection and first-run experience.

This module provides:
- Python project detection
- First-run experience with auto-detection and quick confirm
- Config writing to pyproject.toml
"""

from __future__ import annotations

from typing import Any

from codeflash_python.setup.config_schema import CodeflashConfig

try:
    from codeflash_python.setup.detector import DetectedProject, detect_project, has_existing_config
except ImportError:
    # Stub imports if detector not available
    class DetectedProject:
        pass

    def detect_project() -> Any:
        msg = "detector not available"
        raise NotImplementedError(msg)

    def has_existing_config(project_root: Any) -> tuple[bool, None]:
        return False, None


try:
    from codeflash_python.setup.config_writer import write_config
except ImportError:
    from codeflash_core.danom import Err, Result  # noqa: TC001

    def write_config(detected: Any, config: Any = None) -> Result[str, str]:
        return Err("config_writer not available")


try:
    from codeflash_python.setup.first_run import handle_first_run, is_first_run
except ImportError:

    def is_first_run(project_root: Any = None) -> bool:
        return False

    def handle_first_run(args: Any = None, skip_confirm: bool = False, skip_api_key: bool = False) -> Any:
        return args


__all__ = [
    "CodeflashConfig",
    "DetectedProject",
    "detect_project",
    "handle_first_run",
    "has_existing_config",
    "is_first_run",
    "write_config",
]
