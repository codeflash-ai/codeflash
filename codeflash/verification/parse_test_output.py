from __future__ import annotations

import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import dill as pickle
from junitparser.xunit2 import JUnitXml

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import (
    file_path_from_module_name,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.discovery.discover_unit_tests import discover_parameters_unittest
from codeflash.verification.test_results import (
    FunctionTestInvocation,
    InvocationId,
    TestResults,
)

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import TestFiles
    from codeflash.verification.verification_utils import TestConfig


def parse_test_return_values_bin(
    file_location: Path,
    test_files: TestFiles,
    test_config: TestConfig,
) -> TestResults:
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
                logger.exception(f"Failed to load pickle file. Exception: {e}")
                return test_results
            loop_index_bytes = file.read(8)
            loop_index = int.from_bytes(loop_index_bytes, byteorder="big")
            len_next_bytes = file.read(4)
            len_next = int.from_bytes(len_next_bytes, byteorder="big")
            invocation_id = file.read(len_next).decode("ascii")

            invocation_id_object = InvocationId.from_str_id(encoded_test_name, invocation_id)
            test_file_path = file_path_from_module_name(
                invocation_id_object.test_module_path,
                test_config.project_root_path,
            )

            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)

            test_pickle = pickle.loads(test_pickle_bin) if loop_index == 1 else None
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
                ),
            )
    return test_results


def parse_sqlite_test_results(
    sqlite_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
) -> TestResults:
    test_results = TestResults()
    if not sqlite_file_path.exists():
        logger.warning(f"No test results for {sqlite_file_path} found.")
        return test_results
    try:
        db = sqlite3.connect(sqlite_file_path)
        cur = db.cursor()
        data = cur.execute(
            "SELECT test_module_path, test_class_name, test_function_name, "
            "function_getting_tested, loop_index, iteration_id, runtime, return_value FROM test_results",
        ).fetchall()
    finally:
        db.close()
    for val in data:
        try:
            test_module_path = val[0]
            test_file_path = file_path_from_module_name(test_module_path, test_config.project_root_path)
            # TODO : this is because sqlite writes original file module path. Should make it consistent
            test_type = test_files.get_test_type_by_original_file_path(test_file_path)
            loop_index = val[4]
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
                    return_value=pickle.loads(val[7]) if loop_index == 1 else None,
                    timed_out=False,
                ),
            )
        except Exception:
            logger.exception("Failed to load pickle file.")
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
        xml = JUnitXml.fromfile(str(test_xml_file_path))
    except Exception as e:
        logger.warning(
            f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}",
        )
        return test_results

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
                        logger.info(
                            f"Test log - STDOUT : {run_result.stdout} \n STDERR : {run_result.stderr}",
                        )
                    else:
                        logger.info(
                            f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}",
                        )
                return test_results

            test_class_path = testcase.classname
            test_function = testcase.name.split("[", 1)[0] if "[" in testcase.name else testcase.name
            if test_file_name is None:
                if test_class_path:
                    # TODO : This might not be true if the test is organized under a class
                    test_file_path = file_path_from_module_name(
                        test_class_path,
                        test_config.project_root_path,
                    )
                else:
                    test_file_path = file_path_from_module_name(test_function, test_config.project_root_path)
            else:
                test_file_path = test_config.project_root_path / test_file_name
            if not test_file_path.exists():
                logger.warning(f"Could not find the test for file name - {test_file_path} ")
                continue
            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            assert test_type is not None, f"Test type not found for {test_file_path}"
            test_module_path = module_name_from_file_path(test_file_path, test_config.project_root_path)
            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[
                    len(test_module_path) + 1 :
                ]  # +1 for the dot, gets Unittest class name

            loop_index = 1
            if test_function is None:
                msg = f"testcase.name is None in parse_test_xml for testcase {testcase!r} in file {test_xml_file_path}"
                logger.warning(msg)
                continue
            timed_out = False
            if test_config.test_framework == "pytest":
                loop_index = int(testcase.name.split("[ ", 1)[1][:-2]) if "[" in testcase.name else 1
                if len(testcase.result) > 1:
                    logger.warning(
                        f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!",
                    )
                if len(testcase.result) == 1:
                    message = testcase.result[0].message.lower()
                    if "failed: timeout >" in message:
                        timed_out = True
            else:
                if len(testcase.result) > 1:
                    logger.warning(
                        f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!",
                    )
                if len(testcase.result) == 1:
                    message = testcase.result[0].message.lower()
                    if "timed out" in message:
                        timed_out = True
            matches = re.findall(
                r"!######(.*?):(.*?)([^\.:]*?):(.*?):(.*?):(.*?)######!",
                testcase.system_out or "",
            )
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
                    ),
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
                        ),
                    )

    if not test_results:
        logger.info(
            f"Tests '{[test_file.original_file_path for test_file in test_files.test_files]}' failed to run, skipping",
        )
        if run_result is not None:
            stdout, stderr = "", ""
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.stderr
            logger.debug(
                f"Test log - STDOUT : {stdout} \n STDERR : {stderr}",
            )
    return test_results


def merge_test_results(
    xml_test_results: TestResults,
    bin_test_results: TestResults,
    test_framework: str,
) -> TestResults:
    merged_test_results = TestResults()

    grouped_xml_results: defaultdict[str, TestResults] = defaultdict(TestResults)
    grouped_bin_results: defaultdict[str, TestResults] = defaultdict(TestResults)

    # This is done to match the right iteration_id which might not be available in the xml
    for result in xml_test_results:
        if test_framework == "pytest":
            if (
                result.id.test_function_name.endswith("]") and "[" in result.id.test_function_name
            ):  # parameterized test
                test_function_name = result.id.test_function_name[: result.id.test_function_name.index("[")]
            else:
                test_function_name = result.id.test_function_name

        if test_framework == "unittest":
            test_function_name = result.id.test_function_name
            is_parameterized, new_test_function_name, _ = discover_parameters_unittest(
                test_function_name,
            )
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
                    ),
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
                    ),
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
                    ),
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
        test_xml_path,
        test_files=test_files,
        test_config=test_config,
        run_result=run_result,
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
                sqlite_file_path=sql_results_file,
                test_files=test_files,
                test_config=test_config,
            )
            test_results_bin_file.merge(test_results_sqlite_file)
    except AttributeError as e:
        logger.exception(e)

    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin")).unlink(missing_ok=True)

    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite")).unlink(missing_ok=True)

    return merge_test_results(
        test_results_xml,
        test_results_bin_file,
        test_config.test_framework,
    )
