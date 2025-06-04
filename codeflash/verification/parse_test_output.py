from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

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
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType, VerificationType
from codeflash.verification.coverage_utils import CoverageUtils

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import CodeOptimizationContext, CoverageData, TestFiles
    from codeflash.verification.verification_utils import TestConfig


def parse_func(file_path: Path) -> XMLParser:
    """Parse the XML file with lxml.etree.XMLParser as the backend."""
    xml_parser = XMLParser(huge_tree=True)
    return parse(file_path, xml_parser)


matches_re_start = re.compile(r"!\$######(.*?):(.*?)([^\.:]*?):(.*?):(.*?):(.*?)######\$!\n")
matches_re_end = re.compile(r"!######(.*?):(.*?)([^\.:]*?):(.*?):(.*?):(.*?)######!")


def parse_test_return_values_bin(file_location: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not file_location.exists():
        logger.warning(f"No test results for {file_location} found.")
        console.rule()
        return test_results

    with file_location.open("rb") as file:
        try:
            while file:
                len_next_bytes = file.read(4)
                if not len_next_bytes:
                    return test_results
                len_next = int.from_bytes(len_next_bytes, byteorder="big")
                encoded_test_bytes = file.read(len_next)
                encoded_test_name = encoded_test_bytes.decode("ascii")
                duration_bytes = file.read(8)
                duration = int.from_bytes(duration_bytes, byteorder="big")
                len_next_bytes = file.read(4)
                len_next = int.from_bytes(len_next_bytes, byteorder="big")
                test_pickle_bin = file.read(len_next)
                loop_index_bytes = file.read(8)
                loop_index = int.from_bytes(loop_index_bytes, byteorder="big")
                len_next_bytes = file.read(4)
                len_next = int.from_bytes(len_next_bytes, byteorder="big")
                invocation_id_bytes = file.read(len_next)
                invocation_id = invocation_id_bytes.decode("ascii")

                invocation_id_object = InvocationId.from_str_id(encoded_test_name, invocation_id)
                test_file_path = file_path_from_module_name(
                    invocation_id_object.test_module_path, test_config.tests_project_rootdir
                )

                test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
                try:
                    test_pickle = pickle.loads(test_pickle_bin) if loop_index == 1 else None
                except Exception as e:
                    if DEBUG_MODE:
                        logger.exception(f"Failed to load pickle file for {encoded_test_name} Exception: {e}")
                    continue
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
                        verification_type=VerificationType.FUNCTION_CALL,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to parse test results from {file_location}. Exception: {e}")
            return test_results
    return test_results


def parse_sqlite_test_results(sqlite_file_path: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not sqlite_file_path.exists():
        logger.warning(f"No test results for {sqlite_file_path} found.")
        console.rule()
        return test_results
    db = None
    try:
        db = sqlite3.connect(sqlite_file_path)
        cur = db.cursor()
        data = cur.execute(
            "SELECT test_module_path, test_class_name, test_function_name, "
            "function_getting_tested, loop_index, iteration_id, runtime, return_value,verification_type FROM test_results"
        ).fetchall()
    except Exception as e:
        logger.warning(f"Failed to parse test results from {sqlite_file_path}. Exception: {e}")
        if db is not None:
            db.close()
        return test_results
    finally:
        db.close()
    for val in data:
        try:
            test_module_path = val[0]
            test_class_name = val[1] if val[1] else None
            test_function_name = val[2] if val[2] else None
            function_getting_tested = val[3]
            test_file_path = file_path_from_module_name(test_module_path, test_config.tests_project_rootdir)
            loop_index = val[4]
            iteration_id = val[5]
            runtime = val[6]
            verification_type = val[8]
            if verification_type in {VerificationType.INIT_STATE_FTO, VerificationType.INIT_STATE_HELPER}:
                test_type = TestType.INIT_STATE_TEST
            else:
                # TODO : this is because sqlite writes original file module path. Should make it consistent
                test_type = test_files.get_test_type_by_original_file_path(test_file_path)
            try:
                ret_val = (pickle.loads(val[7]) if loop_index == 1 else None,)
            except Exception:  # noqa: S112
                continue
            test_results.add(
                function_test_invocation=FunctionTestInvocation(
                    loop_index=loop_index,
                    id=InvocationId(
                        test_module_path=test_module_path,
                        test_class_name=test_class_name,
                        test_function_name=test_function_name,
                        function_getting_tested=function_getting_tested,
                        iteration_id=iteration_id,
                    ),
                    file_name=test_file_path,
                    did_pass=True,
                    runtime=runtime,
                    test_framework=test_config.test_framework,
                    test_type=test_type,
                    return_value=ret_val,
                    timed_out=False,
                    verification_type=VerificationType(verification_type) if verification_type else None,
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
    unittest_loop_index: int | None = None,
) -> TestResults:
    test_results = TestResults()
    # Parse unittest output
    if not test_xml_file_path.exists():
        logger.warning(f"No test results for {test_xml_file_path} found.")
        console.rule()
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
            test_file_name = suite._elem.attrib.get("file")  # noqa: SLF001
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

            loop_index = unittest_loop_index if unittest_loop_index is not None else 1

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

            sys_stdout = testcase.system_out or ""
            begin_matches = list(matches_re_start.finditer(sys_stdout))
            end_matches = {}
            for match in matches_re_end.finditer(sys_stdout):
                groups = match.groups()
                if len(groups[5].split(":")) > 1:
                    iteration_id = groups[5].split(":")[0]
                    groups = groups[:5] + (iteration_id,)
                end_matches[groups] = match

            if not begin_matches or not begin_matches:
                test_results.add(
                    FunctionTestInvocation(
                        loop_index=loop_index,
                        id=InvocationId(
                            test_module_path=test_module_path,
                            test_class_name=test_class,
                            test_function_name=test_function,
                            function_getting_tested="",  # TODO: Fix this
                            iteration_id="",
                        ),
                        file_name=test_file_path,
                        runtime=None,
                        test_framework=test_config.test_framework,
                        did_pass=result,
                        test_type=test_type,
                        return_value=None,
                        timed_out=timed_out,
                        stdout="",
                    )
                )

            else:
                for match_index, match in enumerate(begin_matches):
                    groups = match.groups()
                    end_match = end_matches.get(groups)
                    iteration_id, runtime = groups[5], None
                    if end_match:
                        stdout = sys_stdout[match.end() : end_match.start()]
                        split_val = end_match.groups()[5].split(":")
                        if len(split_val) > 1:
                            iteration_id = split_val[0]
                            runtime = int(split_val[1])
                        else:
                            iteration_id, runtime = split_val[0], None
                    elif match_index == len(begin_matches) - 1:
                        stdout = sys_stdout[match.end() :]
                    else:
                        stdout = sys_stdout[match.end() : begin_matches[match_index + 1].start()]

                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=int(groups[4]),
                            id=InvocationId(
                                test_module_path=groups[0],
                                test_class_name=None if groups[1] == "" else groups[1][:-1],
                                test_function_name=groups[2],
                                function_getting_tested=groups[3],
                                iteration_id=iteration_id,
                            ),
                            file_name=test_file_path,
                            runtime=runtime,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                            stdout=stdout,
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
            (result.id.test_module_path or "")
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + (test_function_name or "")
            + ":"
            + str(result.loop_index)
        ].add(result)

    for result in bin_test_results:
        grouped_bin_results[
            (result.id.test_module_path or "")
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + (result.id.test_function_name or "")
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
                        verification_type=VerificationType(result_bin.verification_type)
                        if result_bin.verification_type
                        else None,
                        stdout=xml_result.stdout,
                    )
                )
        elif xml_results.test_results[0].id.iteration_id is not None:
            # This means that we have multiple iterations of the same test function
            # We need to match the iteration_id to the bin results
            for xml_result in xml_results.test_results:
                try:
                    bin_result = bin_results.get_by_unique_invocation_loop_id(xml_result.unique_invocation_loop_id)
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
                        verification_type=VerificationType(bin_result.verification_type)
                        if bin_result.verification_type
                        else None,
                        stdout=xml_result.stdout,
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
                        verification_type=VerificationType(bin_result.verification_type)
                        if bin_result.verification_type
                        else None,
                        stdout=xml_result.stdout,
                    )
                )

    return merged_test_results


def parse_test_results(
    test_xml_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    optimization_iteration: int,
    function_name: str | None,
    source_file: Path | None,
    coverage_database_file: Path | None,
    coverage_config_file: Path | None,
    code_context: CodeOptimizationContext | None = None,
    run_result: subprocess.CompletedProcess | None = None,
    unittest_loop_index: int | None = None,
) -> tuple[TestResults, CoverageData | None]:
    test_results_xml = parse_test_xml(
        test_xml_path,
        test_files=test_files,
        test_config=test_config,
        run_result=run_result,
        unittest_loop_index=unittest_loop_index,
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

    get_run_tmp_file(Path("pytest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path("unittest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite")).unlink(missing_ok=True)
    results = merge_test_results(test_results_xml, test_results_bin_file, test_config.test_framework)

    all_args = False
    if coverage_database_file and source_file and code_context and function_name:
        all_args = True
        coverage = CoverageUtils.load_from_sqlite_database(
            database_path=coverage_database_file,
            config_path=coverage_config_file,
            source_code_path=source_file,
            code_context=code_context,
            function_name=function_name,
        )
        coverage.log_coverage()
    return results, coverage if all_args else None
