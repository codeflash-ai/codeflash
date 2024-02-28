import ast
import os
import site
from tempfile import TemporaryDirectory
from typing import Optional, List, Union


def module_name_from_file_path(file_path: str, project_root: str) -> str:
    relative_path = os.path.relpath(file_path, project_root)
    module_path = relative_path.replace(os.sep, ".")
    if module_path.lower().endswith(".py"):
        module_path = module_path[:-3]
    return module_path


def file_path_from_module_name(module_name: str, project_root: str) -> str:
    """Get file path from module path"""
    return os.path.join(project_root, module_name.replace(".", os.sep) + ".py")


def ellipsis_in_ast(module: ast.AST) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Constant):
            if node.value == ...:
                return True
    return False


def get_imports_from_file(
    file_path: Optional[str] = None,
    file_string: Optional[str] = None,
    file_ast: Optional[ast.AST] = None,
) -> List[Union[ast.Import, ast.ImportFrom]]:
    assert (
        sum([file_path is not None, file_string is not None, file_ast is not None]) == 1
    ), "Must provide exactly one of file_path, file_string, or file_ast"
    if file_path:
        with open(file_path, "r") as file:
            file_string = file.read()
    if file_ast is None:
        file_ast = ast.parse(file_string)
    imports = []
    for node in ast.walk(file_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    return imports


def get_all_function_names(code: str) -> List[str]:
    module = ast.parse(code)
    function_names = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)
    return function_names


def get_run_tmp_file(file_path: str) -> str:
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return os.path.join(get_run_tmp_file.tmpdir.name, file_path)


def path_belongs_to_site_packages(file_path: str) -> bool:
    return any(  # The definition is not part of a site-package Python library
        [
            file_path.startswith(site_package_path + os.sep)
            for site_package_path in site.getsitepackages()
        ]
    )
