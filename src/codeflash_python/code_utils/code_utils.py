from __future__ import annotations

import ast
import difflib
import logging
import re
import shutil
import site
import sys
from pathlib import Path

logger = logging.getLogger("codeflash_python")

ImportErrorPattern = re.compile(r"ModuleNotFoundError.*$", re.MULTILINE)


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


def encoded_tokens_len(s: str) -> int:
    """Return the approximate length of the encoded tokens.

    It's an approximation of BPE encoding (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
    """
    return int(len(s) * 0.25)


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
        raise ValueError(msg) from None


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
            logger.exception("Syntax error in code: %s", e)
            return []
    return [node for node in ast.walk(file_ast) if isinstance(node, (ast.Import, ast.ImportFrom))]


def get_all_function_names(code: str) -> tuple[bool, list[str]]:
    try:
        module = ast.parse(code)
    except SyntaxError as e:
        logger.exception("Syntax error in code: %s", e)
        return False, []

    function_names = [
        node.name for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return True, function_names


def get_run_tmp_file(file_path: Path | str) -> Path:
    import tempfile

    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not hasattr(get_run_tmp_file, "tmpdir_path"):
        # Use mkdtemp instead of TemporaryDirectory to avoid auto-cleanup
        # which can delete the dir before subprocess tests finish using it
        get_run_tmp_file.tmpdir_path = Path(tempfile.mkdtemp(prefix="codeflash_"))  # type: ignore[attr-defined]
    return get_run_tmp_file.tmpdir_path / file_path  # type: ignore[attr-defined]


def path_belongs_to_site_packages(file_path: Path) -> bool:
    file_path_resolved = file_path.resolve()
    site_packages = [Path(p).resolve() for p in site.getsitepackages()]
    return any(file_path_resolved.is_relative_to(site_package_path) for site_package_path in site_packages)


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
    print(message)

    sys.exit(1 if error_on_exit else 0)
