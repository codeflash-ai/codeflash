"""Python-specific helpers for FunctionToOptimize.

The canonical types (FunctionToOptimize, FunctionParent) live in
codeflash_core.models. Import them from there directly.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from codeflash_core.models import FunctionToOptimize  # noqa: TC001


def qualified_name_with_modules_from_root(fto: FunctionToOptimize, project_root_path: Path) -> str:
    from codeflash_python.code_utils.code_utils import module_name_from_file_path

    return f"{module_name_from_file_path(fto.file_path, project_root_path)}.{fto.qualified_name}"
