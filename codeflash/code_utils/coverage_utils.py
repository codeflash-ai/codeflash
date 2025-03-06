from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeflash.code_utils.code_utils import get_run_tmp_file

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext


def extract_dependent_function(main_function: str, code_context: CodeOptimizationContext) -> str | Literal[False]:
    """Extract the single dependent function from the code context excluding the main function."""
    ast_tree = ast.parse(code_context.testgen_context_code)

    dependent_functions = {node.name for node in ast_tree.body if isinstance(node, ast.FunctionDef)}

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


def generate_candidates(source_code_path: Path) -> list[str]:
    """Generate all the possible candidates for coverage data based on the source code path."""
    candidates = [source_code_path.name]
    current_path = source_code_path.parent

    while current_path != current_path.parent:
        candidate_path = str(Path(current_path.name) / candidates[-1])
        candidates.append(candidate_path)
        current_path = current_path.parent

    return candidates


def prepare_coverage_files() -> tuple[Path, Path]:
    """Prepare coverage configuration and output files."""
    coverage_database_file = get_run_tmp_file(Path(".coverage"))
    coveragercfile = get_run_tmp_file(Path(".coveragerc"))
    coveragerc_content = f"[run]\n branch = True\ndata_file={coverage_database_file}\n"
    coveragercfile.write_text(coveragerc_content)
    return coverage_database_file, coveragercfile
