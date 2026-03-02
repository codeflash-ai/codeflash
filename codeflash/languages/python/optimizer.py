from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.models.models import ValidCode

if TYPE_CHECKING:
    from pathlib import Path


def prepare_python_module(
    original_module_code: str, original_module_path: Path, project_root: Path
) -> tuple[dict[Path, ValidCode], ast.Module] | None:
    """Parse a Python module, normalize its code, and validate imported callee modules.

    Returns a mapping of file paths to ValidCode (for the module and its imported callees)
    plus the parsed AST, or None on syntax error.
    """
    from codeflash.languages.python.static_analysis.code_replacer import normalize_code, normalize_node
    from codeflash.languages.python.static_analysis.static_analysis import analyze_imported_modules

    try:
        original_module_ast = ast.parse(original_module_code)
    except SyntaxError as e:
        logger.warning(f"Syntax error parsing code in {original_module_path}: {e}")
        logger.info("Skipping optimization due to file error.")
        return None

    normalized_original_module_code = ast.unparse(normalize_node(original_module_ast))
    validated_original_code: dict[Path, ValidCode] = {
        original_module_path: ValidCode(
            source_code=original_module_code, normalized_code=normalized_original_module_code
        )
    }

    imported_module_analyses = analyze_imported_modules(original_module_code, original_module_path, project_root)

    for analysis in imported_module_analyses:
        callee_original_code = analysis.file_path.read_text(encoding="utf8")
        try:
            normalized_callee_original_code = normalize_code(callee_original_code)
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing code in callee module {analysis.file_path}: {e}")
            logger.info("Skipping optimization due to helper file error.")
            return None
        validated_original_code[analysis.file_path] = ValidCode(
            source_code=callee_original_code, normalized_code=normalized_callee_original_code
        )

    return validated_original_code, original_module_ast


def resolve_python_function_ast(
    function_name: str, parents: list, module_ast: ast.Module
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Look up a function/method AST node in a parsed Python module."""
    from codeflash.languages.python.static_analysis.static_analysis import get_first_top_level_function_or_method_ast

    return get_first_top_level_function_or_method_ast(function_name, parents, module_ast)
