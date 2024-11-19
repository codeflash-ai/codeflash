from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Collection, Optional, Union

from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.coverage_utils import (
    FunctionCoverage,
    extract_dependent_function,
    generate_candidates,
    grab_dependent_function_from_coverage_data,
)
from codeflash.models.models import CodeOptimizationContext
from codeflash.verification.test_results import TestResults


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CoverageData:
    """Represents the coverage data for a specific function in a source file, using one or more test files."""

    file_path: Path
    coverage: float
    function_name: str
    functions_being_tested: list[str]
    graph: dict[str, dict[str, Collection[object]]]
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
                executed_branches=coverage_data[function_name]["executed_branches"],
                unexecuted_branches=coverage_data[function_name]["missing_branches"],
            )
        except KeyError:
            main_function_coverage = FunctionCoverage(
                name=function_name,
                coverage=0,
                executed_lines=[],
                unexecuted_lines=[],
                executed_branches=[],
                unexecuted_branches=[],
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
    ) -> dict[str, dict[str, Collection[object]]]:
        graph = {
            main_func_coverage.name: {
                "executed_lines": set(main_func_coverage.executed_lines),
                "unexecuted_lines": set(main_func_coverage.unexecuted_lines),
                "executed_branches": main_func_coverage.executed_branches,
                "unexecuted_branches": main_func_coverage.unexecuted_branches,
            }
        }

        if dependent_func_coverage:
            graph[dependent_func_coverage.name] = {
                "executed_lines": set(dependent_func_coverage.executed_lines),
                "unexecuted_lines": set(dependent_func_coverage.unexecuted_lines),
                "executed_branches": dependent_func_coverage.executed_branches,
                "unexecuted_branches": dependent_func_coverage.unexecuted_branches,
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


class OriginalCodeBaseline(BaseModel):
    generated_test_results: TestResults
    existing_test_results: TestResults
    overall_test_results: Optional[TestResults]
    runtime: int
    coverage_results: Optional[CoverageData]
