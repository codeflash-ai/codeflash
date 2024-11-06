from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import dill as pickle
from junitparser.xunit2 import JUnitXml
from lxml.etree import XMLParser, parse

from codeflash.cli_cmds.console import DEBUG_MODE, console, logger
from codeflash.code_utils.code_utils import (
    file_name_from_test_module_name,
    file_path_from_module_name,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.discovery.discover_unit_tests import discover_parameters_unittest
from codeflash.verification.test_results import FunctionTestInvocation, InvocationId, TestResults

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import CodeOptimizationContext, TestFiles
    from codeflash.verification.verification_utils import TestConfig


def parse_func(file_path: Path) -> XMLParser:
    """Parse the XML file with lxml.etree.XMLParser as the backend."""
    xml_parser = XMLParser(huge_tree=True)
    return parse(file_path, xml_parser)


def extract_dependent_function(main_function: str, code_context: CodeOptimizationContext) -> str | Literal[False]:
    """Extract the single dependent function from the code context excluding the main function."""
    ast_tree = ast.parse(code_context.code_to_optimize_with_helpers)

    dependent_functions = {node.name for node in ast_tree.body if isinstance(node, ast.FunctionDef)}

    if main_function in dependent_functions:
        dependent_functions.discard(main_function)

    if not dependent_functions:
        return False

    if len(dependent_functions) != 1:
        msg = f"Expected exactly one dependent function, found {len(dependent_functions)}"
        raise ValueError(msg)

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
    dependent_func_coverage: FunctionCoverage | None
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
        coverage = len(total_executed_lines) / len(total_lines) * 100 if total_lines else 0
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

        logger.info(f"Looking for coverage data in {' -> '.join(candidates)}")
        for candidate in candidates:
            try:
                cov: dict[str, dict[str, Any]] = coverage_data["files"][candidate]["functions"]
                logger.info(f"Coverage data found for {source_code_path} in {candidate}")
                break
            except KeyError:
                continue
        else:
            console.print(coverage_data)
            msg = f"Coverage data not found for {source_code_path} in {candidates}"
            raise ValueError(msg)
        return cov

    @staticmethod
    def _fetch_function_coverages(
        function_name: str,
        code_context: CodeOptimizationContext,
        coverage_data: dict[str, dict[str, Any]],
        original_cov_data: dict[str, dict[str, Any]],
    ) -> tuple[FunctionCoverage, FunctionCoverage | None]:
        try:
            main_function_coverage = FunctionCoverage(
                name=function_name,
                coverage=coverage_data[function_name]["summary"]["percent_covered"],
                executed_lines=coverage_data[function_name]["executed_lines"],
                unexecuted_lines=coverage_data[function_name]["excluded_lines"],
            )
        except KeyError:
            msg = f"Coverage data not found for {function_name} in {original_cov_data}"
            raise ValueError(msg) from None

        dependent_function = extract_dependent_function(function_name, code_context)
        try:
            if dependent_function:
                dependent_function_coverage = FunctionCoverage(
                    name=dependent_function,
                    coverage=coverage_data[dependent_function]["summary"]["percent_covered"],
                    executed_lines=coverage_data[dependent_function]["executed_lines"],
                    unexecuted_lines=coverage_data[dependent_function]["excluded_lines"],
                )
                return main_function_coverage, dependent_function_coverage
        except KeyError:
            msg = f"Coverage data not found for {dependent_function} in {original_cov_data}"
            raise ValueError(msg) from None
        return main_function_coverage, None

    @staticmethod
    def _aggregate_coverage(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: FunctionCoverage | None
    ) -> tuple[set[int], set[int]]:
        total_executed_lines = set(main_func_coverage.executed_lines)
        total_unexecuted_lines = set(main_func_coverage.unexecuted_lines)

        if dependent_func_coverage:
            total_executed_lines.update(dependent_func_coverage.executed_lines)
            total_unexecuted_lines.update(dependent_func_coverage.unexecuted_lines)

        return total_executed_lines, total_unexecuted_lines

    @staticmethod
    def _build_graph(
        main_func_coverage: FunctionCoverage, dependent_func_coverage: FunctionCoverage | None
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
            logger.info(self)
            logger.info(f"Coverage: {self.coverage}%, skipping")
            return

        from rich.panel import Panel
        from rich.syntax import Syntax

        union_executed_lines = sorted(
            {line for func in self.functions_being_tested for line in self.graph[func]["executed_lines"]}
        )
        union_unexecuted_lines = sorted(
            {line for func in self.functions_being_tested for line in self.graph[func]["unexecuted_lines"]}
        )
        # adapted from nedbat/coveragepy/coverage/annotate.py:annotate_file
        src = self.code_context.code_to_optimize_with_helpers
        output = ""
        i = j = 0
        covered = True
        for lineno, line in enumerate(src.splitlines(keepends=True), start=1):
            while i < len(union_executed_lines) and union_executed_lines[i] < lineno:
                i += 1
            while j < len(union_unexecuted_lines) and union_unexecuted_lines[j] < lineno:
                j += 1
            if i < len(union_executed_lines) and union_executed_lines[i] == lineno:
                covered = j >= len(union_unexecuted_lines) or union_unexecuted_lines[j] > lineno
            if self.blank_re.match(line):
                output += "  "
            elif self.else_re.match(line):
                if j >= len(union_unexecuted_lines):
                    output += "✅ "
                elif union_executed_lines[i] == union_unexecuted_lines[j]:
                    output += "❌ "
                else:
                    output += "> "
            elif lineno in union_unexecuted_lines:
                output += "- "
            elif covered:
                output += "✅ "
            else:
                output += "❌ "

            output += line

        panel = Panel(
            Syntax(output, "python", line_numbers=True, theme="github-dark"),
            title=f"Coverage: {self.coverage}%",
            subtitle=f"Functions tested: {', '.join(self.functions_being_tested)}",
        )
        console.print(panel)


def parse_test_return_values_bin(file_location: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not file_location.exists():
        logger.warning(f"No test results for {file_location} found.")
        return test_results

    with file_location.open("rb") as file:
        while file:
            len_next_bytes = file.read(4)
            if not len_next_bytes:
                return test_results
            len_next = int.from_bytes(len_next_bytes, byteorder="big")
            encoded_test_name = file.read(len_next).decode("ascii")
            duration_bytes = file.read(8)
            duration = int.from_bytes(duration_bytes, byteorder="big")
            len_next_bytes = file.read(4)
            if not len_next_bytes:
                return test_results
            len_next = int.from_bytes(len_next_bytes, byteorder="big")
            try:
                test_pickle_bin = file.read(len_next)
            except Exception as e:
                if DEBUG_MODE:
                    logger.exception(f"Failed to load pickle file. Exception: {e}")
                return test_results
            loop_index_bytes = file.read(8)
            loop_index = int.from_bytes(loop_index_bytes, byteorder="big")
            len_next_bytes = file.read(4)
            len_next = int.from_bytes(len_next_bytes, byteorder="big")
            invocation_id = file.read(len_next).decode("ascii")

            invocation_id_object = InvocationId.from_str_id(encoded_test_name, invocation_id)
            test_file_path = file_path_from_module_name(
                invocation_id_object.test_module_path, test_config.tests_project_rootdir
            )

            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            try:
                test_pickle = pickle.loads(test_pickle_bin) if loop_index == 1 else None
            except Exception as e:
                if DEBUG_MODE:
                    logger.exception(f"Failed to load pickle file. Exception: {e}")
                return test_results
            assert test_type is not None, f"Test type not found for {test_file_path}"
            test_results.add(
                function_test_invocation=FunctionTestInvocation(
                    loop_index=loop_index,
                    id=invocation_id_object,
                    file_name=test_file_path,
                    did_pass=True,
                    runtime=duration,
                    test_framework=test_config.test_framework,
                    test_type=test_type,
                    return_value=test_pickle,
                    timed_out=False,
                )
            )
    return test_results


def parse_sqlite_test_results(sqlite_file_path: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not sqlite_file_path.exists():
        logger.warning(f"No test results for {sqlite_file_path} found.")
        return test_results
    try:
        db = sqlite3.connect(sqlite_file_path)
        cur = db.cursor()
        data = cur.execute(
            "SELECT test_module_path, test_class_name, test_function_name, "
            "function_getting_tested, loop_index, iteration_id, runtime, return_value FROM test_results"
        ).fetchall()
    finally:
        db.close()
    for val in data:
        try:
            test_module_path = val[0]
            test_file_path = file_path_from_module_name(test_module_path, test_config.tests_project_rootdir)
            # TODO : this is because sqlite writes original file module path. Should make it consistent
            test_type = test_files.get_test_type_by_original_file_path(test_file_path)
            loop_index = val[4]
            try:
                ret_val = (pickle.loads(val[7]) if loop_index == 1 else None,)
            except Exception:
                continue
            test_results.add(
                function_test_invocation=FunctionTestInvocation(
                    loop_index=loop_index,
                    id=InvocationId(
                        test_module_path=val[0],
                        test_class_name=val[1],
                        test_function_name=val[2],
                        function_getting_tested=val[3],
                        iteration_id=val[5],
                    ),
                    file_name=test_file_path,
                    did_pass=True,
                    runtime=val[6],
                    test_framework=test_config.test_framework,
                    test_type=test_type,
                    return_value=ret_val,
                    timed_out=False,
                )
            )
        except Exception:
            logger.exception(f"Failed to parse sqlite test results for {sqlite_file_path}")
        # Hardcoding the test result to True because the test did execute and we are only interested in the return values,
        # the did_pass comes from the xml results file
    return test_results


def parse_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    test_results = TestResults()
    # Parse unittest output
    if not test_xml_file_path.exists():
        logger.warning(f"No test results for {test_xml_file_path} found.")
        return test_results
    try:
        xml = JUnitXml.fromfile(str(test_xml_file_path), parse_func=parse_func)
    except Exception as e:
        logger.warning(f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}")
        return test_results
    base_dir = (
        test_config.tests_project_rootdir if test_config.test_framework == "pytest" else test_config.project_root_path
    )
    for suite in xml:
        for testcase in suite:
            class_name = testcase.classname
            test_file_name = suite._elem.attrib.get("file")
            if (
                test_file_name == f"unittest{os.sep}loader.py"
                and class_name == "unittest.loader._FailedTest"
                and suite.errors == 1
                and suite.tests == 1
            ):
                # This means that the test failed to load, so we don't want to crash on it
                logger.info("Test failed to load, skipping it.")
                if run_result is not None:
                    if isinstance(run_result.stdout, str) and isinstance(run_result.stderr, str):
                        logger.info(f"Test log - STDOUT : {run_result.stdout} \n STDERR : {run_result.stderr}")
                    else:
                        logger.info(
                            f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}"
                        )
                return test_results

            test_class_path = testcase.classname
            try:
                test_function = testcase.name.split("[", 1)[0] if "[" in testcase.name else testcase.name
            except (AttributeError, TypeError) as e:
                msg = (
                    f"Accessing testcase.name in parse_test_xml for testcase {testcase!r} in file"
                    f" {test_xml_file_path} has exception: {e}"
                )
                logger.exception(msg)
                continue
            if test_file_name is None:
                if test_class_path:
                    # TODO : This might not be true if the test is organized under a class
                    test_file_path = file_name_from_test_module_name(test_class_path, base_dir)
                    if test_file_path is None:
                        logger.warning(f"Could not find the test for file name - {test_class_path} ")
                        continue
                else:
                    test_file_path = file_path_from_module_name(test_function, base_dir)
            else:
                test_file_path = base_dir / test_file_name
            assert test_file_path, f"Test file path not found for {test_file_name}"

            if not test_file_path.exists():
                logger.warning(f"Could not find the test for file name - {test_file_path} ")
                continue
            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            assert test_type is not None, f"Test type not found for {test_file_path}"
            test_module_path = module_name_from_file_path(test_file_path, test_config.tests_project_rootdir)
            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[len(test_module_path) + 1 :]  # +1 for the dot, gets Unittest class name

            loop_index = 1

            timed_out = False
            if test_config.test_framework == "pytest":
                loop_index = int(testcase.name.split("[ ")[-1][:-2]) if "[" in testcase.name else 1
                if len(testcase.result) > 1:
                    logger.warning(f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!")
                if len(testcase.result) == 1:
                    message = testcase.result[0].message.lower()
                    if "failed: timeout >" in message:
                        timed_out = True
            else:
                if len(testcase.result) > 1:
                    logger.warning(f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!")
                if len(testcase.result) == 1:
                    message = testcase.result[0].message.lower()
                    if "timed out" in message:
                        timed_out = True
            matches = re.findall(r"!######(.*?):(.*?)([^\.:]*?):(.*?):(.*?):(.*?)######!", testcase.system_out or "")
            if not matches or not len(matches):
                test_results.add(
                    FunctionTestInvocation(
                        loop_index=loop_index,
                        id=InvocationId(
                            test_module_path=test_module_path,
                            test_class_name=test_class,
                            test_function_name=test_function,
                            function_getting_tested="",  # FIXME
                            iteration_id=None,
                        ),
                        file_name=test_file_path,
                        runtime=None,
                        test_framework=test_config.test_framework,
                        did_pass=result,
                        test_type=test_type,
                        return_value=None,
                        timed_out=timed_out,
                    )
                )

            else:
                for match in matches:
                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=int(match[4]),
                            id=InvocationId(
                                test_module_path=match[0],
                                test_class_name=None if match[1] == "" else match[1][:-1],
                                test_function_name=match[2],
                                function_getting_tested=match[3],
                                iteration_id=match[5],
                            ),
                            file_name=test_file_path,
                            runtime=None,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                        )
                    )

    if not test_results:
        logger.info(
            f"Tests '{[test_file.original_file_path for test_file in test_files.test_files]}' failed to run, skipping"
        )
        if run_result is not None:
            stdout, stderr = "", ""
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.stderr
            logger.debug(f"Test log - STDOUT : {stdout} \n STDERR : {stderr}")
    return test_results


def merge_test_results(
    xml_test_results: TestResults, bin_test_results: TestResults, test_framework: str
) -> TestResults:
    merged_test_results = TestResults()

    grouped_xml_results: defaultdict[str, TestResults] = defaultdict(TestResults)
    grouped_bin_results: defaultdict[str, TestResults] = defaultdict(TestResults)

    # This is done to match the right iteration_id which might not be available in the xml
    for result in xml_test_results:
        if test_framework == "pytest":
            if result.id.test_function_name.endswith("]") and "[" in result.id.test_function_name:  # parameterized test
                test_function_name = result.id.test_function_name[: result.id.test_function_name.index("[")]
            else:
                test_function_name = result.id.test_function_name

        if test_framework == "unittest":
            test_function_name = result.id.test_function_name
            is_parameterized, new_test_function_name, _ = discover_parameters_unittest(test_function_name)
            if is_parameterized:  # handle parameterized test
                test_function_name = new_test_function_name

        grouped_xml_results[
            result.id.test_module_path
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + test_function_name
            + ":"
            + str(result.loop_index)
        ].add(result)

    for result in bin_test_results:
        grouped_bin_results[
            result.id.test_module_path
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + result.id.test_function_name
            + ":"
            + str(result.loop_index)
        ].add(result)

    for result_id in grouped_xml_results:
        xml_results = grouped_xml_results[result_id]
        bin_results = grouped_bin_results.get(result_id)
        if not bin_results:
            merged_test_results.merge(xml_results)
            continue

        if len(xml_results) == 1:
            xml_result = xml_results[0]
            # This means that we only have one FunctionTestInvocation for this test xml. Match them to the bin results
            # Either a whole test function fails or passes.
            for result_bin in bin_results:
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=xml_result.loop_index,
                        id=result_bin.id,
                        file_name=xml_result.file_name,
                        runtime=result_bin.runtime,
                        test_framework=xml_result.test_framework,
                        did_pass=xml_result.did_pass,
                        test_type=xml_result.test_type,
                        return_value=result_bin.return_value,
                        timed_out=xml_result.timed_out,
                    )
                )
        elif xml_results.test_results[0].id.iteration_id is not None:
            # This means that we have multiple iterations of the same test function
            # We need to match the iteration_id to the bin results
            for xml_result in xml_results.test_results:
                try:
                    bin_result = bin_results.get_by_id(xml_result.id)
                except AttributeError:
                    bin_result = None
                if bin_result is None:
                    merged_test_results.add(xml_result)
                    continue
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=xml_result.loop_index,
                        id=xml_result.id,
                        file_name=xml_result.file_name,
                        runtime=bin_result.runtime,
                        test_framework=xml_result.test_framework,
                        did_pass=bin_result.did_pass,
                        test_type=xml_result.test_type,
                        return_value=bin_result.return_value,
                        timed_out=xml_result.timed_out
                        if bin_result.runtime is None
                        else False,  # If runtime was measured in the bin file, then the testcase did not time out
                    )
                )
        else:
            # Should happen only if the xml did not have any test invocation id info
            for i, bin_result in enumerate(bin_results.test_results):
                try:
                    xml_result = xml_results.test_results[i]
                except IndexError:
                    xml_result = None
                if xml_result is None:
                    merged_test_results.add(bin_result)
                    continue
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=bin_result.loop_index,
                        id=bin_result.id,
                        file_name=bin_result.file_name,
                        runtime=bin_result.runtime,
                        test_framework=bin_result.test_framework,
                        did_pass=bin_result.did_pass,
                        test_type=bin_result.test_type,
                        return_value=bin_result.return_value,
                        timed_out=xml_result.timed_out,  # only the xml gets the timed_out flag
                    )
                )

    return merged_test_results


def parse_test_results(
    test_xml_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    optimization_iteration: int,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    test_results_xml = parse_test_xml(
        test_xml_path, test_files=test_files, test_config=test_config, run_result=run_result
    )
    try:
        bin_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin"))
        test_results_bin_file = (
            parse_test_return_values_bin(bin_results_file, test_files=test_files, test_config=test_config)
            if bin_results_file.exists()
            else TestResults()
        )
    except AttributeError as e:
        logger.exception(e)
        test_results_bin_file = TestResults()
        get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin")).unlink(missing_ok=True)

    try:
        sql_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite"))
        if sql_results_file.exists():
            test_results_sqlite_file = parse_sqlite_test_results(
                sqlite_file_path=sql_results_file, test_files=test_files, test_config=test_config
            )
            test_results_bin_file.merge(test_results_sqlite_file)
    except AttributeError as e:
        logger.exception(e)

    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin")).unlink(missing_ok=True)

    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite")).unlink(missing_ok=True)
    return merge_test_results(test_results_xml, test_results_bin_file, test_config.test_framework)
