from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def file_path_from_module_name(module_name: str, project_root_path: Path) -> Path:
    """Get file path from module path."""
    return project_root_path / (module_name.replace(".", os.sep) + ".py")


@lru_cache(maxsize=100)
def file_name_from_test_module_name(test_module_name: str, base_dir: Path) -> Path | None:
    partial_test_class = test_module_name
    while partial_test_class:
        test_path = file_path_from_module_name(partial_test_class, base_dir)
        if (base_dir / test_path).exists():
            return base_dir / test_path
        partial_test_class = ".".join(partial_test_class.split(".")[:-1])
    return None
