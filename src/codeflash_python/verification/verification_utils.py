from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from codeflash_core.config import TestConfig

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["TestConfig"]


def get_test_file_path(
    test_dir: Path,
    function_name: str,
    iteration: int = 0,
    test_type: str = "unit",
    source_file_path: Path | None = None,
) -> Path:
    assert test_type in {"unit", "inspired", "replay", "perf"}
    function_name_safe = function_name.replace(".", "_")
    extension = ".py"

    path = test_dir / f"test_{function_name_safe}__{test_type}_test_{iteration}{extension}"

    if path.exists():
        return get_test_file_path(test_dir, function_name, iteration + 1, test_type, source_file_path=source_file_path)
    return path


def delete_multiple_if_name_main(test_ast: ast.Module) -> ast.Module:
    if_indexes = []
    for index, node in enumerate(test_ast.body):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) > 0
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) > 0
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        ):
            if_indexes.append(index)
    for index in list(reversed(if_indexes))[1:]:
        del test_ast.body[index]
    return test_ast


class ModifyInspiredTests(ast.NodeTransformer):
    """Transformer for modifying inspired test classes.

    Class is currently not in active use.
    """

    def __init__(self, import_list: list[ast.stmt], test_framework: str) -> None:
        self.import_list = import_list
        self.test_framework = test_framework

    def visit_Import(self, node: ast.Import) -> None:
        self.import_list.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_list.append(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if self.test_framework != "unittest":
            return node
        found = False
        if node.bases:
            for base in node.bases:
                if (
                    isinstance(base, ast.Attribute)
                    and base.attr == "TestCase"
                    and isinstance(base.value, ast.Name)
                    and base.value.id == "unittest"
                ):
                    found = True
                    break
                # TODO: Check if this is actually a unittest.TestCase
                if isinstance(base, ast.Name) and base.id == "TestCase":
                    found = True
                    break
        if not found:
            return node
        node.name = node.name + "Inspired"
        return node
