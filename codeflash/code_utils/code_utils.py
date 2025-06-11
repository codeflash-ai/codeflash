from __future__ import annotations

import ast
import os
import re
import shutil
import site
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import tomlkit

from codeflash.cli_cmds.console import logger, paneled_text
from codeflash.code_utils.config_parser import find_pyproject_toml

ImportErrorPattern = re.compile(r"ModuleNotFoundError.*$", re.MULTILINE)


@contextmanager
def custom_addopts() -> None:
    pyproject_file = find_pyproject_toml()
    original_content = None
    non_blacklist_plugin_args = ""

    try:
        # Read original file
        if pyproject_file.exists():
            with Path.open(pyproject_file, encoding="utf-8") as f:
                original_content = f.read()
                data = tomlkit.parse(original_content)
            # Backup original addopts
            original_addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
            # nothing to do if no addopts present
            if original_addopts != "" and isinstance(original_addopts, list):
                original_addopts = [x.strip() for x in original_addopts]
                non_blacklist_plugin_args = re.sub(r"-n(?: +|=)\S+", "", " ".join(original_addopts)).split(" ")
                non_blacklist_plugin_args = [x for x in non_blacklist_plugin_args if x != ""]
                if non_blacklist_plugin_args != original_addopts:
                    data["tool"]["pytest"]["ini_options"]["addopts"] = non_blacklist_plugin_args
                    # Write modified file
                    with Path.open(pyproject_file, "w", encoding="utf-8") as f:
                        f.write(tomlkit.dumps(data))

        yield

    finally:
        # Restore original file
        if (
            original_content
            and pyproject_file.exists()
            and tuple(original_addopts) not in {(), tuple(non_blacklist_plugin_args)}
        ):
            with Path.open(pyproject_file, "w", encoding="utf-8") as f:
                f.write(original_content)


@contextmanager
def add_addopts_to_pyproject() -> None:
    pyproject_file = find_pyproject_toml()
    original_content = None
    try:
        # Read original file
        if pyproject_file.exists():
            with Path.open(pyproject_file, encoding="utf-8") as f:
                original_content = f.read()
                data = tomlkit.parse(original_content)
            data["tool"]["pytest"] = {}
            data["tool"]["pytest"]["ini_options"] = {}
            data["tool"]["pytest"]["ini_options"]["addopts"] = [
                "-n=auto",
                "-n",
                "1",
                "-n 1",
                "-n      1",
                "-n      auto",
            ]
            with Path.open(pyproject_file, "w", encoding="utf-8") as f:
                f.write(tomlkit.dumps(data))

        yield

    finally:
        # Restore original file
        with Path.open(pyproject_file, "w", encoding="utf-8") as f:
            f.write(original_content)


def encoded_tokens_len(s: str) -> int:
    """Return the approximate length of the encoded tokens.

    It's an approximation of BPE encoding (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
    """
    return int(len(s) * 0.25)


def get_qualified_name(module_name: str, full_qualified_name: str) -> str:
    if not full_qualified_name:
        msg = "full_qualified_name cannot be empty"
        raise ValueError(msg)
    if not full_qualified_name.startswith(module_name):
        msg = f"{full_qualified_name} does not start with {module_name}"
        raise ValueError(msg)
    if module_name == full_qualified_name:
        msg = f"{full_qualified_name} is the same as {module_name}"
        raise ValueError(msg)
    return full_qualified_name[len(module_name) + 1 :]


def module_name_from_file_path(file_path: Path, project_root_path: Path) -> str:
    relative_path = file_path.relative_to(project_root_path)
    return relative_path.with_suffix("").as_posix().replace("/", ".")


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


def get_imports_from_file(
    file_path: Path | None = None, file_string: str | None = None, file_ast: ast.AST | None = None
) -> list[ast.Import | ast.ImportFrom]:
    assert sum([file_path is not None, file_string is not None, file_ast is not None]) == 1, (
        "Must provide exactly one of file_path, file_string, or file_ast"
    )
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


def validate_python_code(code: str) -> str:
    """Validate a string of Python code by attempting to compile it."""
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        msg = f"Invalid Python code: {e.msg} (line {e.lineno}, column {e.offset})"
        raise ValueError(msg) from e
    return code


def has_any_async_functions(code: str) -> bool:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(module))


def cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if path and path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def restore_conftest(path_to_content_map: dict[Path, str]) -> None:
    for path, file_content in path_to_content_map.items():
        path.write_text(file_content, encoding="utf8")


def exit_with_message(message: str, *, error_on_exit: bool = False) -> None:
    paneled_text(message, panel_args={"style": "red"})

    sys.exit(1 if error_on_exit else 0)
