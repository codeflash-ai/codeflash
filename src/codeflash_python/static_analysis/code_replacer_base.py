"""Code replacement utilities for swapping function definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from codeflash_python.models.models import CodeStringsMarkdown


def get_optimized_code_for_module(relative_path: Path, optimized_code: CodeStringsMarkdown) -> str:
    file_to_code_context = optimized_code.file_to_path()
    module_optimized_code = file_to_code_context.get(str(relative_path))
    if module_optimized_code is not None:
        return module_optimized_code

    # Fallback 1: single code block with no file path
    if "None" in file_to_code_context and len(file_to_code_context) == 1:
        logger.debug("Using code block with None file_path for %s", relative_path)
        return file_to_code_context["None"]

    # Fallback 2: match by filename (basename) -- the LLM sometimes returns a different
    # directory prefix but the correct filename
    target_name = relative_path.name
    basename_matches = [
        code for path, code in file_to_code_context.items() if path != "None" and Path(path).name == target_name
    ]
    if len(basename_matches) == 1:
        logger.debug("Using basename-matched code block for %s", relative_path)
        return basename_matches[0]

    logger.warning(
        "Optimized code not found for %s, existing files are %s", relative_path, list(file_to_code_context.keys())
    )
    return ""
