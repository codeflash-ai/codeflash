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
    source_file_path: Path | None = None,
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
        # For JavaScript/TypeScript, place generated tests in a subdirectory that matches
        # Vitest/Jest include patterns (e.g., test/**/*.test.ts)
        if is_javascript():
            package_test_dir = _find_js_package_test_dir(test_dir, source_file_path)
            if package_test_dir:
                test_dir = package_test_dir

        path = test_dir / f"test_{function_name_safe}__{test_type}_test_{iteration}{extension}"

    if path.exists():
        return get_test_file_path(
            test_dir, function_name, iteration + 1, test_type, package_name, class_name, source_file_path
        )
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
    _test_framework: Optional[str] = None  # Cached test framework detection result

    def __post_init__(self) -> None:
        self.tests_root = self.tests_root.resolve()
        self.project_root_path = self.project_root_path.resolve()
        self.tests_project_rootdir = self.tests_project_rootdir.resolve()

    @property
    def test_framework(self) -> str:
        """Returns the appropriate test framework based on language.

        For JavaScript/TypeScript: uses the configured framework (vitest, jest, or mocha).
        For Python: uses pytest as default.
        Result is cached after first detection to avoid repeated pom.xml parsing.
        """
        if self._test_framework is not None:
            return self._test_framework
        if is_javascript():
            from codeflash.languages.test_framework import get_js_test_framework_or_default

            self._test_framework = get_js_test_framework_or_default()
        elif is_java():
            self._test_framework = self._detect_java_test_framework()
        else:
            self._test_framework = "pytest"
        return self._test_framework

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
                    if parent_config and (
                        parent_config.has_junit4 or parent_config.has_junit5 or parent_config.has_testng
                    ):
                        return parent_config.test_framework
                current = current.parent

            # Return whatever the initial detection found, or default
            if config and config.test_framework:
                return config.test_framework
        except Exception:
            pass
        return "junit4"  # Default fallback (JUnit 4 is more common in legacy projects)

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
