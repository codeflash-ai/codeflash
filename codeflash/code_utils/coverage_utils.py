from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeflash.code_utils.code_utils import get_run_tmp_file

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext


def extract_dependent_function(main_function: str, code_context: CodeOptimizationContext) -> str | Literal[False]:
    """Extract the single dependent function from the code context excluding the main function."""
    ast_tree = ast.parse(code_context.code_to_optimize_with_helpers)

    dependent_functions = {node.name for node in ast_tree.body if isinstance(node, ast.FunctionDef)}

    if main_function in dependent_functions:
        dependent_functions.discard(main_function)

    if not dependent_functions:
        return False

    if len(dependent_functions) != 1:
        return False

    return dependent_functions.pop()


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
    coverage_out_file = get_run_tmp_file(Path("coverage.json"))
    coveragercfile = get_run_tmp_file(Path(".coveragerc"))
    coveragerc_content = f"[run]\n branch = True\n [json]\n output = {coverage_out_file.as_posix()}\n"
    coveragercfile.write_text(coveragerc_content)
    return coverage_out_file, coveragercfile
