from __future__ import annotations

import ast
import os
import site
from tempfile import TemporaryDirectory

from codeflash.cli_cmds.console import logger
from pathlib import Path


def module_name_from_file_path(file_path: Path, project_root_path: Path) -> str:
    relative_path = file_path.relative_to(project_root_path)
    return relative_path.with_suffix("").as_posix().replace("/", ".")


def file_path_from_module_name(module_name: str, project_root_path: Path) -> Path:
    """Get file path from module path."""
    return project_root_path / (module_name.replace(".", os.sep) + ".py")


def get_imports_from_file(
    file_path: Path | None = None,
    file_string: str | None = None,
    file_ast: ast.AST | None = None,
) -> list[ast.Import | ast.ImportFrom]:
    assert (
        sum([file_path is not None, file_string is not None, file_ast is not None]) == 1
    ), "Must provide exactly one of file_path, file_string, or file_ast"
    if file_path:
        with file_path.open(encoding="utf8") as file:
            file_string = file.read()
    if file_ast is None:
        if file_string is None:
            logger.error("file_string cannot be None when file_ast is not provided")
            return []
        try:
            file_ast = ast.parse(file_string)
        except SyntaxError as e:
            logger.exception(f"Syntax error in code: {e}")
            return []
    return [node for node in ast.walk(file_ast) if isinstance(node, (ast.Import, ast.ImportFrom))]


def get_all_function_names(code: str) -> tuple[bool, list[str]]:
    try:
        module = ast.parse(code)
    except SyntaxError as e:
        logger.exception(f"Syntax error in code: {e}")
        return False, []

    function_names = [
        node.name for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return True, function_names


def get_run_tmp_file(file_path: Path) -> Path:
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return Path(get_run_tmp_file.tmpdir.name) / file_path


def path_belongs_to_site_packages(file_path: Path) -> bool:
    site_packages = [Path(p) for p in site.getsitepackages()]
    return any(file_path.resolve().is_relative_to(site_package_path) for site_package_path in site_packages)


def is_class_defined_in_file(class_name: str, file_path: Path) -> bool:
    if not file_path.exists():
        return False
    with file_path.open(encoding="utf8") as file:
        source = file.read()
    tree = ast.parse(source)
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))
