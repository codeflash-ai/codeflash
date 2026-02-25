from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass

from codeflash.languages import current_language_support, is_javascript


def get_test_file_path(
    test_dir: Path,
    function_name: str,
    iteration: int = 0,
    test_type: str = "unit",
    source_file_path: Path | None = None,
) -> Path:
    assert test_type in {"unit", "inspired", "replay", "perf"}
    function_name = function_name.replace(".", "_")
    # Use appropriate file extension based on language
    extension = current_language_support().get_test_file_suffix() if is_javascript() else ".py"

    # For JavaScript/TypeScript, place generated tests in a subdirectory that matches
    # Vitest/Jest include patterns (e.g., test/**/*.test.ts)
    # if is_javascript():
    #     # For monorepos, first try to find the package directory from the source file path
    #     # e.g., packages/workflow/src/utils.ts -> packages/workflow/test/codeflash-generated/
    #     package_test_dir = _find_js_package_test_dir(test_dir, source_file_path)
    #     if package_test_dir:
    #         test_dir = package_test_dir

    path = test_dir / f"test_{function_name}__{test_type}_test_{iteration}{extension}"
    if path.exists():
        return get_test_file_path(test_dir, function_name, iteration + 1, test_type, source_file_path)
    return path


def _find_js_package_test_dir(tests_root: Path, source_file_path: Path | None) -> Path | None:
    """Find the appropriate test directory for a JavaScript/TypeScript package.

    For monorepos, this finds the package's test directory from the source file path.
    For example: packages/workflow/src/utils.ts -> packages/workflow/test/codeflash-generated/

    Args:
        tests_root: The root tests directory (may be monorepo packages root).
        source_file_path: Path to the source file being tested.

    Returns:
        The test directory path, or None if not found.

    """
    if source_file_path is None:
        # No source path provided, check if test_dir itself has a test subdirectory
        for test_subdir_name in ["test", "tests", "__tests__", "src/__tests__"]:
            test_subdir = tests_root / test_subdir_name
            if test_subdir.is_dir():
                codeflash_test_dir = test_subdir / "codeflash-generated"
                codeflash_test_dir.mkdir(parents=True, exist_ok=True)
                return codeflash_test_dir
        return None

    try:
        # Resolve paths for reliable comparison
        tests_root = tests_root.resolve()
        source_path = Path(source_file_path).resolve()

        # Walk up from the source file to find a directory with package.json or test/ folder
        package_dir = None

        for parent in source_path.parents:
            # Stop if we've gone above or reached the tests_root level
            # For monorepos, tests_root might be /packages/ and we want to search within packages
            if parent in (tests_root, tests_root.parent):
                break

            # Check if this looks like a package root
            has_package_json = (parent / "package.json").exists()
            has_test_dir = any((parent / d).is_dir() for d in ["test", "tests", "__tests__"])

            if has_package_json or has_test_dir:
                package_dir = parent
                break

        if package_dir:
            # Find the test directory in this package
            for test_subdir_name in ["test", "tests", "__tests__", "src/__tests__"]:
                test_subdir = package_dir / test_subdir_name
                if test_subdir.is_dir():
                    codeflash_test_dir = test_subdir / "codeflash-generated"
                    codeflash_test_dir.mkdir(parents=True, exist_ok=True)
                    return codeflash_test_dir

        return None
    except Exception:
        return None


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
        """Returns the appropriate test framework based on language.

        For JavaScript/TypeScript: uses the configured framework (vitest, jest, or mocha).
        For Python: uses pytest as default.
        """
        if is_javascript():
            from codeflash.languages.test_framework import get_js_test_framework_or_default

            return get_js_test_framework_or_default()
        return "pytest"

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
