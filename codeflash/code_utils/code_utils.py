from __future__ import annotations

import ast
import logging
import os
import site
from tempfile import TemporaryDirectory


def module_name_from_file_path(file_path: str, project_root_path: str) -> str:
    relative_path = os.path.relpath(file_path, project_root_path)
    module_path = relative_path.replace(os.sep, ".")
    if module_path.lower().endswith(".py"):
        module_path = module_path[:-3]
    return module_path


def file_path_from_module_name(module_name: str, project_root_path: str) -> str:
    """Get file path from module path"""
    return os.path.join(project_root_path, module_name.replace(".", os.sep) + ".py")


def get_imports_from_file(
    file_path: str | None = None,
    file_string: str | None = None,
    file_ast: ast.AST | None = None,
) -> list[ast.Import | ast.ImportFrom]:
    assert (
        sum([file_path is not None, file_string is not None, file_ast is not None]) == 1
    ), "Must provide exactly one of file_path, file_string, or file_ast"
    if file_path:
        with open(file_path, encoding="utf8") as file:
            file_string = file.read()
    if file_ast is None:
        try:
            file_ast = ast.parse(file_string)
        except SyntaxError as e:
            logging.exception(f"Syntax error in code: {e}")
            return []
    imports = []
    for node in ast.walk(file_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    return imports


def get_all_function_names(code: str) -> tuple[bool, list[str]]:
    try:
        module = ast.parse(code)
    except SyntaxError as e:
        logging.exception(f"Syntax error in code: {e}")
        return False, []

    function_names = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
    return True, function_names


def get_run_tmp_file(file_path: str) -> str:
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return os.path.join(get_run_tmp_file.tmpdir.name, file_path)


def path_belongs_to_site_packages(file_path: str) -> bool:
    site_packages = site.getsitepackages()
    for site_package_path in site_packages:
        if file_path.startswith(site_package_path + os.sep):
            return True
    return False


def is_class_defined_in_file(class_name: str, file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path) as file:
        source = file.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
    return False
