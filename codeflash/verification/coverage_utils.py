from __future__ import annotations

import json
import xml.etree.ElementTree as ET
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


# TODO:{self} Needs cleanup for jest logic check for coverage algorithm here and if we need to move it to /support
class JestCoverageUtils:
    """Coverage utils class for interfacing with Jest coverage output."""

    @staticmethod
    def load_from_jest_json(
        coverage_json_path: Path, function_name: str, code_context: CodeOptimizationContext, source_code_path: Path
    ) -> CoverageData:
        """Load coverage data from Jest's coverage-final.json file.

        Args:
            coverage_json_path: Path to coverage-final.json
            function_name: Name of the function being tested
            code_context: Code optimization context
            source_code_path: Path to the source file being tested

        Returns:
            CoverageData object with parsed coverage information

        """
        if not coverage_json_path or not coverage_json_path.exists():
            logger.debug(f"Jest coverage file not found: {coverage_json_path}")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        try:
            with coverage_json_path.open(encoding="utf-8") as f:
                coverage_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse Jest coverage file: {e}")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        # Find the file entry in coverage data
        # Jest uses absolute paths as keys
        file_coverage = None
        source_path_str = str(source_code_path.resolve())

        for file_path, file_data in coverage_data.items():
            if file_path == source_path_str or file_path.endswith(source_code_path.name):
                file_coverage = file_data
                break

        if not file_coverage:
            logger.debug(f"No coverage data found for {source_code_path} in Jest coverage")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        # Extract line coverage from statement map and execution counts
        statement_map = file_coverage.get("statementMap", {})
        statement_counts = file_coverage.get("s", {})
        fn_map = file_coverage.get("fnMap", {})
        fn_counts = file_coverage.get("f", {})
        branch_map = file_coverage.get("branchMap", {})
        branch_counts = file_coverage.get("b", {})

        # Find the function in fnMap
        function_entry = None
        function_idx = None
        for idx, fn_data in fn_map.items():
            if fn_data.get("name") == function_name:
                function_entry = fn_data
                function_idx = idx
                break

        # Get function line range
        if function_entry:
            fn_start_line = function_entry.get("loc", {}).get("start", {}).get("line", 1)
            fn_end_line = function_entry.get("loc", {}).get("end", {}).get("line", 999999)
        else:
            # If function not found in fnMap, use entire file
            fn_start_line = 1
            fn_end_line = 999999
            logger.debug(f"Function {function_name} not found in Jest fnMap, using file coverage")

        # Calculate executed and unexecuted lines within the function
        executed_lines = []
        unexecuted_lines = []

        for stmt_idx, stmt_data in statement_map.items():
            stmt_start = stmt_data.get("start", {}).get("line", 0)
            stmt_end = stmt_data.get("end", {}).get("line", 0)

            # Check if statement is within function bounds
            if stmt_start >= fn_start_line and stmt_end <= fn_end_line:
                count = statement_counts.get(stmt_idx, 0)
                if count > 0:
                    # Add all lines covered by this statement
                    for line in range(stmt_start, stmt_end + 1):
                        if line not in executed_lines:
                            executed_lines.append(line)
                else:
                    for line in range(stmt_start, stmt_end + 1):
                        if line not in unexecuted_lines and line not in executed_lines:
                            unexecuted_lines.append(line)

        # Extract branch coverage
        executed_branches = []
        unexecuted_branches = []

        for branch_idx, branch_data in branch_map.items():
            branch_line = branch_data.get("loc", {}).get("start", {}).get("line", 0)
            if fn_start_line <= branch_line <= fn_end_line:
                branch_hits = branch_counts.get(branch_idx, [])
                for i, hit_count in enumerate(branch_hits):
                    if hit_count > 0:
                        executed_branches.append([branch_line, i])
                    else:
                        unexecuted_branches.append([branch_line, i])

        # Calculate coverage percentage
        total_lines = set(executed_lines) | set(unexecuted_lines)
        coverage_pct = (len(executed_lines) / len(total_lines) * 100) if total_lines else 0.0

        main_func_coverage = FunctionCoverage(
            name=function_name,
            coverage=coverage_pct,
            executed_lines=sorted(executed_lines),
            unexecuted_lines=sorted(unexecuted_lines),
            executed_branches=executed_branches,
            unexecuted_branches=unexecuted_branches,
        )

        graph = {
            function_name: {
                "executed_lines": set(executed_lines),
                "unexecuted_lines": set(unexecuted_lines),
                "executed_branches": executed_branches,
                "unexecuted_branches": unexecuted_branches,
            }
        }

        return CoverageData(
            file_path=source_code_path,
            coverage=coverage_pct,
            function_name=function_name,
            functions_being_tested=[function_name],
            graph=graph,
            code_context=code_context,
            main_func_coverage=main_func_coverage,
            dependent_func_coverage=None,
            status=CoverageStatus.PARSED_SUCCESSFULLY,
        )


class JacocoCoverageUtils:
    """Coverage utils class for parsing JaCoCo XML reports (Java)."""

    @staticmethod
    def load_from_jacoco_xml(
        jacoco_xml_path: Path,
        function_name: str,
        code_context: CodeOptimizationContext,
        source_code_path: Path,
        _class_name: str | None = None,
    ) -> CoverageData:
        """Load coverage data from JaCoCo XML report.

        JaCoCo XML structure:
        <report>
          <package name="com/example">
            <class name="com/example/Calculator" sourcefilename="Calculator.java">
              <method name="add" desc="(II)I" line="10">
                <counter type="INSTRUCTION" missed="0" covered="5"/>
                <counter type="BRANCH" missed="0" covered="2"/>
                <counter type="LINE" missed="0" covered="3"/>
              </method>
            </class>
            <sourcefile name="Calculator.java">
              <line nr="10" mi="0" ci="2" mb="0" cb="0"/>
              <line nr="11" mi="0" ci="1" mb="0" cb="2"/>
            </sourcefile>
          </package>
        </report>

        Args:
            jacoco_xml_path: Path to jacoco.xml report file.
            function_name: Name of the function/method being tested.
            code_context: Code optimization context.
            source_code_path: Path to the source file being tested.
            class_name: Optional fully qualified class name (e.g., "com.example.Calculator").

        Returns:
            CoverageData object with parsed coverage information.

        """
        if not jacoco_xml_path or not jacoco_xml_path.exists():
            logger.warning(f"JaCoCo XML file not found at path: {jacoco_xml_path}")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        # Log file info for debugging
        file_size = jacoco_xml_path.stat().st_size
        logger.info(f"Parsing JaCoCo XML file: {jacoco_xml_path} (size: {file_size} bytes)")

        if file_size == 0:
            logger.warning(f"JaCoCo XML file is empty: {jacoco_xml_path}")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        try:
            tree = ET.parse(jacoco_xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            # Log detailed debugging info
            try:
                with jacoco_xml_path.open(encoding="utf-8") as f:
                    content_preview = f.read(500)
                logger.warning(
                    f"Failed to parse JaCoCo XML file at '{jacoco_xml_path}' "
                    f"(size: {file_size} bytes, exists: {jacoco_xml_path.exists()}): {e}. "
                    f"File preview: {content_preview!r}"
                )
            except Exception as read_err:
                logger.warning(
                    f"Failed to parse JaCoCo XML file at '{jacoco_xml_path}': {e}. Could not read file: {read_err}"
                )
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        # Determine expected source file name from path
        source_filename = source_code_path.name

        # Find the matching sourcefile element and collect all method start lines
        sourcefile_elem = None
        method_elem = None
        method_start_line = None
        all_method_start_lines: list[int] = []

        for package in root.findall(".//package"):
            # Look for the sourcefile matching our source file
            for sf in package.findall("sourcefile"):
                if sf.get("name") == source_filename:
                    sourcefile_elem = sf
                    break

            # Look for the class and method, collect all method start lines
            for cls in package.findall("class"):
                cls_source = cls.get("sourcefilename")
                if cls_source == source_filename:
                    # Collect all method start lines for boundary detection
                    for method in cls.findall("method"):
                        method_line = int(method.get("line", 0))
                        if method_line > 0:
                            all_method_start_lines.append(method_line)

                        # Check if this is our target method
                        method_name = method.get("name")
                        if method_name == function_name:
                            method_elem = method
                            method_start_line = method_line

            if sourcefile_elem is not None:
                break

        if sourcefile_elem is None:
            logger.debug(f"No coverage data found for {source_filename} in JaCoCo report")
            return CoverageData.create_empty(source_code_path, function_name, code_context)

        # Sort method start lines to determine boundaries
        all_method_start_lines = sorted(set(all_method_start_lines))

        # Parse line-level coverage from sourcefile
        executed_lines: list[int] = []
        unexecuted_lines: list[int] = []
        executed_branches: list[list[int]] = []
        unexecuted_branches: list[list[int]] = []

        # Get all line data
        line_data: dict[int, dict[str, int]] = {}
        for line in sourcefile_elem.findall("line"):
            line_nr = int(line.get("nr", 0))
            line_data[line_nr] = {
                "mi": int(line.get("mi", 0)),  # missed instructions
                "ci": int(line.get("ci", 0)),  # covered instructions
                "mb": int(line.get("mb", 0)),  # missed branches
                "cb": int(line.get("cb", 0)),  # covered branches
            }

        # Determine method boundaries
        if method_start_line:
            # Find the next method's start line to determine this method's end
            method_end_line = None
            for start_line in all_method_start_lines:
                if start_line > method_start_line:
                    # Next method starts here, so our method ends before this
                    method_end_line = start_line - 1
                    break

            # If no next method found, use the max line in the file
            if method_end_line is None:
                all_lines = sorted(line_data.keys())
                method_end_line = max(all_lines) if all_lines else method_start_line

            # Filter to lines within the method boundaries
            for line_nr, data in sorted(line_data.items()):
                if method_start_line <= line_nr <= method_end_line:
                    # Line is covered if it has covered instructions
                    if data["ci"] > 0:
                        executed_lines.append(line_nr)
                    elif data["mi"] > 0:
                        unexecuted_lines.append(line_nr)

                    # Branch coverage
                    if data["cb"] > 0:
                        # Covered branches - each branch is [line, branch_id]
                        for i in range(data["cb"]):
                            executed_branches.append([line_nr, i])
                    if data["mb"] > 0:
                        # Missed branches
                        for i in range(data["mb"]):
                            unexecuted_branches.append([line_nr, data["cb"] + i])
        else:
            # No method found - use all lines in the file
            for line_nr, data in sorted(line_data.items()):
                if data["ci"] > 0:
                    executed_lines.append(line_nr)
                elif data["mi"] > 0:
                    unexecuted_lines.append(line_nr)

                if data["cb"] > 0:
                    for i in range(data["cb"]):
                        executed_branches.append([line_nr, i])
                if data["mb"] > 0:
                    for i in range(data["mb"]):
                        unexecuted_branches.append([line_nr, data["cb"] + i])

        # Calculate coverage percentage
        total_lines = set(executed_lines) | set(unexecuted_lines)
        coverage_pct = (len(executed_lines) / len(total_lines) * 100) if total_lines else 0.0

        # If we found method-level counters, use them as the authoritative source
        if method_elem is not None:
            for counter in method_elem.findall("counter"):
                if counter.get("type") == "LINE":
                    missed = int(counter.get("missed", 0))
                    covered = int(counter.get("covered", 0))
                    if missed + covered > 0:
                        coverage_pct = covered / (missed + covered) * 100
                    break

        main_func_coverage = FunctionCoverage(
            name=function_name,
            coverage=coverage_pct,
            executed_lines=sorted(executed_lines),
            unexecuted_lines=sorted(unexecuted_lines),
            executed_branches=executed_branches,
            unexecuted_branches=unexecuted_branches,
        )

        graph = {
            function_name: {
                "executed_lines": set(executed_lines),
                "unexecuted_lines": set(unexecuted_lines),
                "executed_branches": executed_branches,
                "unexecuted_branches": unexecuted_branches,
            }
        }

        return CoverageData(
            file_path=source_code_path,
            coverage=coverage_pct,
            function_name=function_name,
            functions_being_tested=[function_name],
            graph=graph,
            code_context=code_context,
            main_func_coverage=main_func_coverage,
            dependent_func_coverage=None,
            status=CoverageStatus.PARSED_SUCCESSFULLY,
        )


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

        if not database_path.exists() or not database_path.stat().st_size:
            logger.debug(f"Coverage database {database_path} is empty or does not exist")
            sentry_sdk.capture_message(f"Coverage database {database_path} is empty or does not exist")
            return CoverageData.create_empty(source_code_path, function_name, code_context)
        cov.load()

        reporter = JsonReporter(cov)
        temp_json_file = database_path.with_suffix(".report.json")
        with temp_json_file.open("w", encoding="utf-8") as f:
            try:
                reporter.report(morfs=[source_code_path.as_posix()], outfile=f)
            except NoDataError:
                sentry_sdk.capture_message(f"No coverage data found for {function_name} in {source_code_path}")
                return CoverageData.create_empty(source_code_path, function_name, code_context)
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
        with coverage_file_path.open(encoding="utf-8") as f:
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
