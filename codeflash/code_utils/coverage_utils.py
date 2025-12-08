from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeflash.code_utils.code_utils import get_run_tmp_file

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext


def extract_dependent_function(main_function: str, code_context: CodeOptimizationContext) -> str | Literal[False]:
    """Extract the single dependent function from the code context excluding the main function."""
    dependent_functions = set()
    for code_string in code_context.testgen_context.code_strings:
        ast_tree = ast.parse(code_string.code)
        dependent_functions.update(
            {node.name for node in ast_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        )

    if main_function in dependent_functions:
        dependent_functions.discard(main_function)

    if not dependent_functions:
        return False

    if len(dependent_functions) != 1:
        return False

    return build_fully_qualified_name(dependent_functions.pop(), code_context)


def build_fully_qualified_name(function_name: str, code_context: CodeOptimizationContext) -> str:
    full_name = function_name
    for obj_name, parents in code_context.preexisting_objects:
        if obj_name == function_name:
            for parent in parents:
                if parent.type == "ClassDef":
                    full_name = f"{parent.name}.{full_name}"
            break
    return full_name


def generate_candidates(source_code_path: Path) -> set[str]:
    """Generate all the possible candidates for coverage data based on the source code path."""
    candidates = set()
    # Add the filename as a candidate
    name = source_code_path.name
    candidates.add(name)

    # Precompute parts for efficient candidate path construction
    parts = source_code_path.parts
    n = len(parts)

    # Walk up the directory structure without creating Path objects or repeatedly converting to posix
    last_added = name
    # Start from the last parent and move up to the root, exclusive (skip the root itself)
    for i in range(n - 2, 0, -1):
        # Combine the ith part with the accumulated path (last_added)
        candidate_path = f"{parts[i]}/{last_added}"
        candidates.add(candidate_path)
        last_added = candidate_path

    # Add the absolute posix path as a candidate
    candidates.add(source_code_path.as_posix())
    return candidates


def prepare_coverage_files() -> tuple[Path, Path]:
    """
    Prepare coverage configuration and output files.
    
    Returns tuple of (coverage_database_file, coverage_config_file).
    """
    from codeflash.cli_cmds.console import logger
    
    logger.info("!lsp|coverage_utils.prepare_coverage_files: Starting coverage file preparation")
    
    logger.info("!lsp|coverage_utils.prepare_coverage_files: Getting coverage database file path")
    coverage_database_file = get_run_tmp_file(Path(".coverage"))
    logger.info(f"!lsp|coverage_utils.prepare_coverage_files: Coverage database file: {coverage_database_file}")
    
    logger.info("!lsp|coverage_utils.prepare_coverage_files: Getting coverage config file path")
    coveragercfile = get_run_tmp_file(Path(".coveragerc"))
    logger.info(f"!lsp|coverage_utils.prepare_coverage_files: Coverage config file: {coveragercfile}")
    
    coveragerc_content = f"[run]\n branch = True\ndata_file={coverage_database_file}\n"
    logger.info("!lsp|coverage_utils.prepare_coverage_files: Writing coverage config content")
    logger.debug(f"coverage_utils.prepare_coverage_files: Config content:\n{coveragerc_content}")
    
    try:
        coveragercfile.write_text(coveragerc_content)
        logger.info("!lsp|coverage_utils.prepare_coverage_files: Coverage config file written successfully")
    except Exception as e:
        logger.error(
            f"!lsp|coverage_utils.prepare_coverage_files: Failed to write coverage config file: "
            f"{type(e).__name__}: {e}"
        )
        raise
    
    logger.info(
        f"!lsp|coverage_utils.prepare_coverage_files: Coverage files prepared successfully. "
        f"database={coverage_database_file}, config={coveragercfile}"
    )
    return coverage_database_file, coveragercfile
