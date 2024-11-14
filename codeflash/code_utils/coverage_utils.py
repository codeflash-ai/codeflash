from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

from pydantic.dataclasses import dataclass

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import get_run_tmp_file
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
                        )
            msg = f"Coverage data not found for dependent function {dependent_function_name} in the original coverage data"
        except KeyError:
            raise ValueError(msg) from None

    return FunctionCoverage(name=dependent_function_name, coverage=0, executed_lines=[], unexecuted_lines=[])


def prepare_coverage_files(project_root: Path) -> tuple[Path, Path]:
    """Prepare coverage configuration and output files."""
    coverage_out_file = get_run_tmp_file(Path("coverage.json"))
    coveragercfile = get_run_tmp_file(Path(".coveragerc"))
    coveragerc_content = (
        "[run]\n"
        f"source = {project_root.as_posix()}\n"
        "branch = True\n"
        "[json]\n"
        f"output = {coverage_out_file.as_posix()}\n"
    )
    coveragercfile.write_text(coveragerc_content)
    return coverage_out_file, coveragercfile


@dataclass
class FunctionCoverage:
    """Represents the coverage data for a specific function in a source file."""

    name: str
    coverage: float
    executed_lines: list[int]
    unexecuted_lines: list[int]


@dataclass
class CoverageData:
    """Represents the coverage data for a specific function in a source file, using one or more test files."""

    file_path: Path
    coverage: float
    function_name: str
    functions_being_tested: list[str]
    graph: dict[str, dict[str, set[int]]]
    code_context: CodeOptimizationContext
    main_func_coverage: FunctionCoverage
    dependent_func_coverage: Union[FunctionCoverage, None]
    blank_re = re.compile(r"\s*(#|$)")
    else_re = re.compile(r"\s*else\s*:\s*(#|$)")

    @staticmethod
    def load_from_coverage_file(
        coverage_file_path: Path, source_code_path: Path, function_name: str, code_context: CodeOptimizationContext
    ) -> CoverageData:
        """Load coverage data, including main function and its dependencies."""
        from json import load

        with coverage_file_path.open() as f:
            original_coverage_data = load(f)  # we can remove this once we're done debugging
        coverage_data = CoverageData._parse_coverage_file(coverage_file_path, source_code_path)
        main_func_coverage, dependent_func_coverage = CoverageData._fetch_function_coverages(
            function_name, code_context, coverage_data, original_cov_data=original_coverage_data
        )

        total_executed_lines, total_unexecuted_lines = CoverageData._aggregate_coverage(
            main_func_coverage, dependent_func_coverage
        )

        total_lines = total_executed_lines | total_unexecuted_lines
        coverage = len(total_executed_lines) / len(total_lines) * 100 if total_lines else 0.0
        # coverage = (lines covered of the original function + its 1 level deep helpers) / (lines spanned by original function + its 1 level deep helpers), if no helpers then just the original function coverage

        functions_being_tested = [main_func_coverage.name]
        if dependent_func_coverage:
            functions_being_tested.append(dependent_func_coverage.name)

        graph = CoverageData._build_graph(main_func_coverage, dependent_func_coverage)

        return CoverageData(
            file_path=source_code_path,
            coverage=coverage,
            function_name=function_name,
            functions_being_tested=functions_being_tested,
            graph=graph,
            code_context=code_context,
            main_func_coverage=main_func_coverage,
            dependent_func_coverage=dependent_func_coverage,
        )

    @staticmethod
    def _parse_coverage_file(coverage_file_path: Path, source_code_path: Path) -> dict[str, dict[str, Any]]:
        with coverage_file_path.open() as f:
            coverage_data = json.load(f)

        candidates = generate_candidates(source_code_path)

        logger.debug(f"Looking for coverage data in {' -> '.join(candidates)}")
        for candidate in candidates:
            try:
                cov: dict[str, dict[str, Any]] = coverage_data["files"][candidate]["functions"]
                logger.debug(f"Coverage data found for {source_code_path} in {candidate}")
                break
            except KeyError:
                continue
        else:
            logger.debug(f"No coverage data found for {source_code_path} in {candidates}")
            cov = {}
        return cov

    @staticmethod
    def _fetch_function_coverages(
        function_name: str,
        code_context: CodeOptimizationContext,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
    ) -> tuple[FunctionCoverage, Union[FunctionCoverage, None]]:
        try:
            main_function_coverage = FunctionCoverage(
                name=function_name,
                coverage=coverage_data[function_name]["summary"]["percent_covered"],
                executed_lines=coverage_data[function_name]["executed_lines"],
                unexecuted_lines=coverage_data[function_name]["missing_lines"],
            )
        except KeyError:
            main_function_coverage = FunctionCoverage(
                name=function_name, coverage=0, executed_lines=[], unexecuted_lines=[]
            )

        dependent_function = extract_dependent_function(function_name, code_context)
        dependent_func_coverage = (
            grab_dependent_function_from_coverage_data(dependent_function, coverage_data, original_cov_data)
            if dependent_function
            else None
        )

        return main_function_coverage, dependent_func_coverage

    @staticmethod
    def _aggregate_coverage(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: Union[FunctionCoverage, None]
    ) -> tuple[set[int], set[int]]:
        total_executed_lines = set(main_func_coverage.executed_lines)
        total_unexecuted_lines = set(main_func_coverage.unexecuted_lines)

        if dependent_func_coverage:
            total_executed_lines.update(dependent_func_coverage.executed_lines)
            total_unexecuted_lines.update(dependent_func_coverage.unexecuted_lines)

        return total_executed_lines, total_unexecuted_lines

    @staticmethod
    def _build_graph(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: Union[FunctionCoverage, None]
    ) -> dict[str, dict[str, set[int]]]:
        graph = {
            main_func_coverage.name: {
                "executed_lines": set(main_func_coverage.executed_lines),
                "unexecuted_lines": set(main_func_coverage.unexecuted_lines),
            }
        }
        if dependent_func_coverage:
            graph[dependent_func_coverage.name] = {
                "executed_lines": set(dependent_func_coverage.executed_lines),
                "unexecuted_lines": set(dependent_func_coverage.unexecuted_lines),
            }
        return graph

    def log_coverage(self) -> None:  # noqa: C901, PLR0912
        """Annotate the source code with the coverage data."""
        if not self.coverage:
            logger.debug(self)
            console.rule(f"No coverage data found for {self.function_name}")
            return

        console.rule(f"Coverage data for {self.function_name}: {self.coverage:.2f}%")

        if self.dependent_func_coverage:
            console.rule(
                f"Dependent function {self.dependent_func_coverage.name}: {self.dependent_func_coverage.coverage:.2f}%"
            )
        # TODO: fix this eventually to get a visual representation of the coverage data, will make it easier to grasp the coverage data and our impact on it
        # from rich.panel import Panel
        # from rich.syntax import Syntax

        # union_executed_lines = sorted(
        #     {line for func in self.functions_being_tested for line in self.graph[func]["executed_lines"]}
        # )
        # union_unexecuted_lines = sorted(
        #     {line for func in self.functions_being_tested for line in self.graph[func]["unexecuted_lines"]}
        # )
        # # adapted from nedbat/coveragepy/coverage/annotate.py:annotate_file
        # # src = self.code_context.code_to_optimize_with_helpers.splitlines()
        # src = self.main_func_coverage.
        # output = ""
        # for i, line in enumerate(src, 1):
        #     if i in union_executed_lines:
        #         output += f"✅ {line}"
        #     elif i in union_unexecuted_lines:
        #         output += f"❌ {line}"
        #     else:
        #         output += line
        #     output += "\n"

        # panel = Panel(
        #     Syntax(output, "python", line_numbers=True, theme="github-dark"),
        #     title=f"Coverage: {self.coverage}%",
        #     subtitle=f"Functions tested: {', '.join(self.functions_being_tested)}",
        # )

        # console.print(panel)
