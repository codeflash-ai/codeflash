"""Setup module for Codeflash auto-detection and first-run experience.

This module provides:
- Universal project detection across all supported languages
- First-run experience with auto-detection and quick confirm
- Config writing to native config files (pyproject.toml, package.json)
"""

from codeflash.setup.config_schema import CodeflashConfig
from codeflash.setup.config_writer import write_config
from codeflash.setup.detector import DetectedProject, detect_project, has_existing_config
from codeflash.setup.first_run import handle_first_run, is_first_run

__all__ = [
    "CodeflashConfig",
    "DetectedProject",
    "detect_project",
    "handle_first_run",
    "has_existing_config",
    "is_first_run",
    "write_config",
]
