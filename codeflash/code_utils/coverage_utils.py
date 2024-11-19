from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic.dataclasses import dataclass

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


def grab_dependent_function_from_coverage_data(
    dependent_function_name: str, coverage_data: dict[str, dict[str, Any]], original_cov_data: dict[str, dict[str, Any]]
) -> FunctionCoverage:
    """Grab the dependent function from the coverage data."""
    try:
        return FunctionCoverage(
            name=dependent_function_name,
            coverage=coverage_data[dependent_function_name]["summary"]["percent_covered"],
            executed_lines=coverage_data[dependent_function_name]["executed_lines"],
            unexecuted_lines=coverage_data[dependent_function_name]["missing_lines"],
            executed_branches=coverage_data[dependent_function_name]["executed_branches"],
            unexecuted_branches=coverage_data[dependent_function_name]["missing_branches"],
        )
    except KeyError:
        msg = f"Coverage data not found for dependent function {dependent_function_name} in the coverage data"
        try:
            files = original_cov_data["files"]
            for file in files:
                functions = files[file]["functions"]
                for function in functions:
                    if dependent_function_name in function:
                        return FunctionCoverage(
                            name=dependent_function_name,
                            coverage=functions[function]["summary"]["percent_covered"],
                            executed_lines=functions[function]["executed_lines"],
                            unexecuted_lines=functions[function]["missing_lines"],
                            executed_branches=functions[function]["executed_branches"],
                            unexecuted_branches=functions[function]["missing_branches"],
                        )
            msg = f"Coverage data not found for dependent function {dependent_function_name} in the original coverage data"
        except KeyError:
            raise ValueError(msg) from None

    return FunctionCoverage(
        name=dependent_function_name,
        coverage=0,
        executed_lines=[],
        unexecuted_lines=[],
        executed_branches=[],
        unexecuted_branches=[],
    )


def prepare_coverage_files() -> tuple[Path, Path]:
    """Prepare coverage configuration and output files."""
    coverage_out_file = get_run_tmp_file(Path("coverage.json"))
    coveragercfile = get_run_tmp_file(Path(".coveragerc"))
    coveragerc_content = f"[run]\n branch = True\n [json]\n output = {coverage_out_file.as_posix()}\n"
    coveragercfile.write_text(coveragerc_content)
    return coverage_out_file, coveragercfile


@dataclass
class FunctionCoverage:
    """Represents the coverage data for a specific function in a source file."""

    name: str
    coverage: float
    executed_lines: list[int]
    unexecuted_lines: list[int]
    executed_branches: list[list[int]]
    unexecuted_branches: list[list[int]]
