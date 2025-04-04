from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Union

import sentry_sdk
from coverage.exceptions import NoDataError

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.coverage_utils import (
    build_fully_qualified_name,
    extract_dependent_function,
    generate_candidates,
)
from codeflash.models.models import CoverageData, CoverageStatus, FunctionCoverage

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from codeflash.models.models import CodeOptimizationContext


class CoverageUtils:
    """Coverage utils class for interfacing with Coverage."""

    @staticmethod
    def load_from_sqlite_database(
        database_path: Path,
        config_path: Path,
        function_name: str,
        code_context: CodeOptimizationContext,
        source_code_path: Path,
    ) -> CoverageData:
        """Load coverage data from an SQLite database, mimicking the behavior of load_from_coverage_file."""
        from coverage import Coverage
        from coverage.jsonreport import JsonReporter

        cov = Coverage(data_file=database_path, config_file=config_path, data_suffix=True, auto_data=True, branch=True)

        if not database_path.stat().st_size or not database_path.exists():
            logger.debug(f"Coverage database {database_path} is empty or does not exist")
            sentry_sdk.capture_message(f"Coverage database {database_path} is empty or does not exist")
            return CoverageUtils.create_empty(source_code_path, function_name, code_context)
        cov.load()

        reporter = JsonReporter(cov)
        temp_json_file = database_path.with_suffix(".report.json")
        with temp_json_file.open("w") as f:
            try:
                reporter.report(morfs=[source_code_path.as_posix()], outfile=f)
            except NoDataError:
                sentry_sdk.capture_message(f"No coverage data found for {function_name} in {source_code_path}")
                return CoverageUtils.create_empty(source_code_path, function_name, code_context)
        with temp_json_file.open() as f:
            original_coverage_data = json.load(f)

        coverage_data, status = CoverageUtils._parse_coverage_file(temp_json_file, source_code_path)

        main_func_coverage, dependent_func_coverage = CoverageUtils._fetch_function_coverages(
            function_name, code_context, coverage_data, original_cov_data=original_coverage_data
        )

        total_executed_lines, total_unexecuted_lines = CoverageUtils._aggregate_coverage(
            main_func_coverage, dependent_func_coverage
        )

        total_lines = total_executed_lines | total_unexecuted_lines
        coverage = len(total_executed_lines) / len(total_lines) * 100 if total_lines else 0.0
        # coverage = (lines covered of the original function + its 1 level deep helpers) / (lines spanned by original function + its 1 level deep helpers), if no helpers then just the original function coverage

        functions_being_tested = [main_func_coverage.name]
        if dependent_func_coverage:
            functions_being_tested.append(dependent_func_coverage.name)

        graph = CoverageUtils._build_graph(main_func_coverage, dependent_func_coverage)
        temp_json_file.unlink()

        return CoverageData(
            file_path=source_code_path,
            coverage=coverage,
            function_name=function_name,
            functions_being_tested=functions_being_tested,
            graph=graph,
            code_context=code_context,
            main_func_coverage=main_func_coverage,
            dependent_func_coverage=dependent_func_coverage,
            status=status,
        )

    @staticmethod
    def _parse_coverage_file(
        coverage_file_path: Path, source_code_path: Path
    ) -> tuple[dict[str, dict[str, Any]], CoverageStatus]:
        with coverage_file_path.open() as f:
            coverage_data = json.load(f)

        candidates = generate_candidates(source_code_path)

        logger.debug(f"Looking for coverage data in {' -> '.join(candidates)}")
        for candidate in candidates:
            try:
                cov: dict[str, dict[str, Any]] = coverage_data["files"][candidate]["functions"]
                logger.debug(f"Coverage data found for {source_code_path} in {candidate}")
                status = CoverageStatus.PARSED_SUCCESSFULLY
                break
            except KeyError:
                continue
        else:
            logger.debug(f"No coverage data found for {source_code_path} in {candidates}")
            cov = {}
            status = CoverageStatus.NOT_FOUND
        return cov, status

    @staticmethod
    def _fetch_function_coverages(
        function_name: str,
        code_context: CodeOptimizationContext,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
    ) -> tuple[FunctionCoverage, Union[FunctionCoverage, None]]:
        resolved_name = build_fully_qualified_name(function_name, code_context)
        try:
            main_function_coverage = FunctionCoverage(
                name=resolved_name,
                coverage=coverage_data[resolved_name]["summary"]["percent_covered"],
                executed_lines=coverage_data[resolved_name]["executed_lines"],
                unexecuted_lines=coverage_data[resolved_name]["missing_lines"],
                executed_branches=coverage_data[resolved_name]["executed_branches"],
                unexecuted_branches=coverage_data[resolved_name]["missing_branches"],
            )
        except KeyError:
            main_function_coverage = FunctionCoverage(
                name=resolved_name,
                coverage=0,
                executed_lines=[],
                unexecuted_lines=[],
                executed_branches=[],
                unexecuted_branches=[],
            )

        dependent_function = extract_dependent_function(function_name, code_context)
        dependent_func_coverage = (
            CoverageUtils.grab_dependent_function_from_coverage_data(
                dependent_function, coverage_data, original_cov_data
            )
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

    @staticmethod
    def grab_dependent_function_from_coverage_data(
        dependent_function_name: str,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
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
