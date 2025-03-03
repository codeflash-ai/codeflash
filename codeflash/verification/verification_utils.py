from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass


def get_test_file_path(test_dir: Path, function_name: str, iteration: int = 0, test_type: str = "unit") -> Path:
    assert test_type in ["unit", "inspired", "replay", "perf"]
    function_name = function_name.replace(".", "_")
    path = test_dir / f"test_{function_name}__{test_type}_test_{iteration}.py"
    if path.exists():
        return get_test_file_path(test_dir, function_name, iteration + 1, test_type)
    return path


def delete_multiple_if_name_main(test_ast: ast.Module) -> ast.Module:
    if_indexes = []
    for index, node in enumerate(test_ast.body):
        if isinstance(node, ast.If) and (
            node.test.comparators[0].value == "__main__"
            and node.test.left.id == "__name__"
            and isinstance(node.test.ops[0], ast.Eq)
        ):
            if_indexes.append(index)
    for index in list(reversed(if_indexes))[1:]:
        del test_ast.body[index]
    return test_ast


class ModifyInspiredTests(ast.NodeTransformer):
    """Transformer for modifying inspired test classes.

    Class is currently not in active use.
    """

    def __init__(self, import_list: list[ast.AST], test_framework: str) -> None:
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
                if isinstance(base, ast.Attribute) and base.attr == "TestCase" and base.value.id == "unittest":
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


@dataclass
class TestConfig:
    tests_root: Path
    project_root_path: Path
    test_framework: str
    tests_project_rootdir: Path
    # tests_project_rootdir corresponds to pytest rootdir,
    # or for unittest - project_root_from_module_root(args.tests_root, pyproject_file_path)
    concolic_test_root_dir: Optional[Path] = None
    pytest_cmd: str = "pytest"
