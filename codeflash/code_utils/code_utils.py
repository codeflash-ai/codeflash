from __future__ import annotations

import ast
import configparser
import difflib
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
from codeflash.code_utils.config_parser import find_pyproject_toml, get_all_closest_config_files
from codeflash.lsp.helpers import is_LSP_enabled

ImportErrorPattern = re.compile(r"ModuleNotFoundError.*$", re.MULTILINE)

BLACKLIST_ADDOPTS = ("--benchmark", "--sugar", "--codespeed", "--cov", "--profile", "--junitxml", "-n")


def unified_diff_strings(code1: str, code2: str, fromfile: str = "original", tofile: str = "modified") -> str:
    """Return the unified diff between two code strings as a single string.

    :param code1: First code string (original).
    :param code2: Second code string (modified).
    :param fromfile: Label for the first code string.
    :param tofile: Label for the second code string.
    :return: Unified diff as a string.
    """
    code1_lines = code1.splitlines(keepends=True)
    code2_lines = code2.splitlines(keepends=True)

    diff = difflib.unified_diff(code1_lines, code2_lines, fromfile=fromfile, tofile=tofile, lineterm="")

    return "".join(diff)


def diff_length(a: str, b: str) -> int:
    """Compute the length (in characters) of the unified diff between two strings.

    Args:
        a (str): Original string.
        b (str): Modified string.

    Returns:
        int: Total number of characters in the diff.

    """
    # Split input strings into lines for line-by-line diff
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)

    # Compute unified diff
    diff_lines = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))

    # Join all lines with newline to calculate total diff length
    diff_text = "\n".join(diff_lines)

    return len(diff_text)


def create_rank_dictionary_compact(int_array: list[int]) -> dict[int, int]:
    """Create a dictionary from a list of ints, mapping the original index to its rank.

    This version uses a more compact, "Pythonic" implementation.

    Args:
        int_array: A list of integers.

    Returns:
        A dictionary where keys are original indices and values are the
        rank of the element in ascending order.

    """
    # Sort the indices of the array based on their corresponding values
    sorted_indices = sorted(range(len(int_array)), key=lambda i: int_array[i])

    # Create a dictionary mapping the original index to its rank (its position in the sorted list)
    return {original_index: rank for rank, original_index in enumerate(sorted_indices)}


def filter_args(addopts_args: list[str]) -> list[str]:
    # Convert BLACKLIST_ADDOPTS to a set for faster lookup of simple matches
    # But keep tuple for startswith
    blacklist = BLACKLIST_ADDOPTS
    # Precompute the length for re-use
    n = len(addopts_args)
    filtered_args = []
    i = 0
    while i < n:
        current_arg = addopts_args[i]
        if current_arg.startswith(blacklist):
            i += 1
            if i < n and not addopts_args[i].startswith("-"):
                i += 1
        else:
            filtered_args.append(current_arg)
            i += 1
    return filtered_args


def modify_addopts(config_file: Path) -> tuple[str, bool]:  # noqa : PLR0911
    file_type = config_file.suffix.lower()
    filename = config_file.name
    config = None
    if file_type not in {".toml", ".ini", ".cfg"} or not config_file.exists():
        return "", False
    # Read original file
    with Path.open(config_file, encoding="utf-8") as f:
        content = f.read()
    try:
        if filename == "pyproject.toml":
            # use tomlkit
            data = tomlkit.parse(content)
            original_addopts = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("addopts", "")
            # nothing to do if no addopts present
            if original_addopts == "":
                return content, False
            if isinstance(original_addopts, list):
                original_addopts = " ".join(original_addopts)
            original_addopts = original_addopts.replace("=", " ")
            addopts_args = (
                original_addopts.split()
            )  # any number of space characters as delimiter, doesn't look at = which is fine
        else:
            # use configparser
            config = configparser.ConfigParser()
            config.read_string(content)
            data = {section: dict(config[section]) for section in config.sections()}
            if config_file.name in {"pytest.ini", ".pytest.ini", "tox.ini"}:
                original_addopts = data.get("pytest", {}).get("addopts", "")  # should only be a string
            else:
                original_addopts = data.get("tool:pytest", {}).get("addopts", "")  # should only be a string
            original_addopts = original_addopts.replace("=", " ")
            addopts_args = original_addopts.split()
        new_addopts_args = filter_args(addopts_args)
        if new_addopts_args == addopts_args:
            return content, False
        # change addopts now
        if file_type == ".toml":
            data["tool"]["pytest"]["ini_options"]["addopts"] = " ".join(new_addopts_args)
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                f.write(tomlkit.dumps(data))
                return content, True
        elif config_file.name in {"pytest.ini", ".pytest.ini", "tox.ini"}:
            config.set("pytest", "addopts", " ".join(new_addopts_args))
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                config.write(f)
                return content, True
        else:
            config.set("tool:pytest", "addopts", " ".join(new_addopts_args))
            # Write modified file
            with Path.open(config_file, "w", encoding="utf-8") as f:
                config.write(f)
                return content, True

    except Exception:
        logger.debug("Trouble parsing")
        return content, False  # not modified


@contextmanager
def custom_addopts() -> None:
    closest_config_files = get_all_closest_config_files()

    original_content = {}

    try:
        for config_file in closest_config_files:
            original_content[config_file] = modify_addopts(config_file)
        yield

    finally:
        # Restore original file
        for file, (content, was_modified) in original_content.items():
            if was_modified:
                with Path.open(file, "w", encoding="utf-8") as f:
                    f.write(content)


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


def module_name_from_file_path(file_path: Path, project_root_path: Path, *, traverse_up: bool = False) -> str:
    try:
        relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        return relative_path.with_suffix("").as_posix().replace("/", ".")
    except ValueError:
        if traverse_up:
            parent = file_path.parent
            while parent not in (project_root_path, parent.parent):
                try:
                    relative_path = file_path.resolve().relative_to(parent.resolve())
                    return relative_path.with_suffix("").as_posix().replace("/", ".")
                except ValueError:
                    parent = parent.parent
        msg = f"File {file_path} is not within the project root {project_root_path}."
        raise ValueError(msg)  # noqa: B904


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


def get_run_tmp_file(file_path: Path | str) -> Path:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not hasattr(get_run_tmp_file, "tmpdir"):
        get_run_tmp_file.tmpdir = TemporaryDirectory(prefix="codeflash_")
    return Path(get_run_tmp_file.tmpdir.name) / file_path


def path_belongs_to_site_packages(file_path: Path) -> bool:
    file_path_resolved = file_path.resolve()
    site_packages = [Path(p).resolve() for p in site.getsitepackages()]
    return any(file_path_resolved.is_relative_to(site_package_path) for site_package_path in site_packages)


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
    """Don't Call it inside the lsp process, it will terminate the lsp server."""
    if is_LSP_enabled():
        logger.error(message)
        return
    paneled_text(message, panel_args={"style": "red"})

    sys.exit(1 if error_on_exit else 0)


def extract_unique_errors(pytest_output: str) -> set[str]:
    unique_errors = set()

    # Regex pattern to match error lines:
    # - Start with 'E' followed by optional whitespace
    # - Capture the actual error message
    pattern = r"^E\s+(.*)$"

    for error_message in re.findall(pattern, pytest_output, re.MULTILINE):
        error_message = error_message.strip()  # noqa: PLW2901
        if error_message:
            unique_errors.add(error_message)

    return unique_errors
