from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def unified_diff(original: str, optimized: str, file_path: Path, context_lines: int = 3) -> str:
    """Generate a unified diff between original and optimized code."""
    original_lines = original.splitlines(keepends=True)
    optimized_lines = optimized.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines, optimized_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}", n=context_lines
    )
    return "".join(diff)
