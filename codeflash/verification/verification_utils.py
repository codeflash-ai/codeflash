from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass

from codeflash.languages import current_language_support, is_java, is_javascript


def get_test_file_path(
    test_dir: Path,
    function_name: str,
    iteration: int = 0,
    test_type: str = "unit",
    package_name: str | None = None,
    class_name: str | None = None,
) -> Path:
    assert test_type in {"unit", "inspired", "replay", "perf"}
    function_name_safe = function_name.replace(".", "_")
    # Use appropriate file extension based on language
    if is_javascript():
        extension = current_language_support().get_test_file_suffix()
    elif is_java():
        extension = ".java"
    else:
        extension = ".py"

    if is_java() and package_name:
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
        path = test_dir / f"test_{function_name_safe}__{test_type}_test_{iteration}{extension}"

    if path.exists():
        return get_test_file_path(test_dir, function_name, iteration + 1, test_type, package_name, class_name)
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

    @property
    def test_framework(self) -> str:
        """Returns the appropriate test framework based on language.

        For JavaScript/TypeScript: uses the configured framework (vitest, jest, or mocha).
        For Python: uses pytest as default.
        """
        if is_javascript():
            from codeflash.languages.test_framework import get_js_test_framework_or_default

            return get_js_test_framework_or_default()
        if is_java():
            return self._detect_java_test_framework()
        return "pytest"

    def _detect_java_test_framework(self) -> str:
        """Detect the Java test framework from the project configuration.

        Returns 'junit4', 'junit5', or 'testng' based on project dependencies.
        Checks both the project root and parent directories for multi-module projects.
        Defaults to 'junit5' if detection fails.
        """
        try:
            from codeflash.languages.java.config import detect_java_project

            # First try the project root
            config = detect_java_project(self.project_root_path)
            if config and config.test_framework and (config.has_junit4 or config.has_junit5 or config.has_testng):
                return config.test_framework

            # For multi-module projects, check parent directories
            current = self.project_root_path.parent
            while current != current.parent:
                pom_path = current / "pom.xml"
                if pom_path.exists():
                    parent_config = detect_java_project(current)
                    if parent_config and (parent_config.has_junit4 or parent_config.has_junit5 or parent_config.has_testng):
                        return parent_config.test_framework
                current = current.parent

            # Return whatever the initial detection found, or default
            if config and config.test_framework:
                return config.test_framework
        except Exception:
            pass
        return "junit5"  # Default fallback

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
