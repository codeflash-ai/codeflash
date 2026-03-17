from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass

from codeflash.languages import current_language_support


def get_test_file_path(
    test_dir: Path,
    function_name: str,
    iteration: int = 0,
    test_type: str = "unit",
    package_name: str | None = None,
    class_name: str | None = None,
    source_file_path: Path | None = None,
) -> Path:
    assert test_type in {"unit", "inspired", "replay", "perf"}
    function_name_safe = function_name.replace(".", "_")
    # Use appropriate file extension based on language
    lang_support = current_language_support()
    extension = lang_support.get_test_file_suffix()

    if package_name:
        # For Java, create package directory structure
        # e.g., com.example -> com/example/
        package_path = package_name.replace(".", "/")
        java_class_name = class_name or f"{function_name_safe.title()}Test"
        # Add suffix to avoid conflicts
        if test_type == "perf":
            java_class_name = f"{java_class_name}__perfonlyinstrumented"
        elif test_type == "unit":
            java_class_name = f"{java_class_name}__perfinstrumented"
        path = test_dir / package_path / f"{java_class_name}{extension}"
        # Create package directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Let language support find the appropriate test subdirectory
        # (e.g., for JS monorepos: packages/workflow/test/codeflash-generated/)
        package_test_dir = lang_support.get_test_dir_for_source(test_dir, source_file_path)
        if package_test_dir:
            test_dir = package_test_dir

        path = test_dir / f"test_{function_name_safe}__{test_type}_test_{iteration}{extension}"

    if path.exists():
        return get_test_file_path(
            test_dir, function_name, iteration + 1, test_type, package_name, class_name, source_file_path
        )
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
    tests_project_rootdir: Path
    # tests_project_rootdir corresponds to pytest rootdir
    concolic_test_root_dir: Optional[Path] = None
    pytest_cmd: str = "pytest"
    benchmark_tests_root: Optional[Path] = None
    use_cache: bool = True
    _language: Optional[str] = None  # Language identifier for multi-language support
    js_project_root: Optional[Path] = None  # JavaScript project root (directory containing package.json)

    def __post_init__(self) -> None:
        self.project_root_path = self.project_root_path.resolve()
        self.tests_project_rootdir = self.tests_project_rootdir.resolve()

    @property
    def test_framework(self) -> str:
        """Returns the appropriate test framework based on language."""
        return current_language_support().test_framework

    def set_language(self, language: str) -> None:
        """Set the language for this test config.

        Args:
            language: Language identifier (e.g., "python", "javascript").

        """
        self._language = language

    @property
    def language(self) -> Optional[str]:
        """Get the current language setting."""
        return self._language
