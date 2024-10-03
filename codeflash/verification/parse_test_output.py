from __future__ import annotations

import logging
import os
import pathlib
import re
import sqlite3
from collections import defaultdict
from typing import TYPE_CHECKING

import dill as pickle
from junitparser.xunit2 import JUnitXml

from codeflash.code_utils.code_utils import (
    file_path_from_module_name,
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.discovery.discover_unit_tests import discover_parameters_unittest
from codeflash.models.models import TestFiles
from codeflash.verification.test_results import (
    FunctionTestInvocation,
    InvocationId,
    TestResults,
    TestType,
)

if TYPE_CHECKING:
    import subprocess

    from codeflash.verification.verification_utils import TestConfig


def parse_test_return_values_bin(
    file_location: str,
    test_files: TestFiles,
    test_config: TestConfig,
) -> TestResults:
    test_results = TestResults()
    if not os.path.exists(file_location):
        logging.warning(f"No test results for {file_location} found.")
        return test_results

    with open(file_location, "rb") as file:
        while file:
            len_next = file.read(4)
            if not len_next:
                return test_results
            len_next = int.from_bytes(len_next, byteorder="big")
            encoded_test_name = file.read(len_next).decode("ascii")
            len_next = file.read(8)
            duration = int.from_bytes(len_next, byteorder="big")
            len_next = file.read(4)
            if not len_next:
                return test_results
            len_next = int.from_bytes(len_next, byteorder="big")
            try:
                test_pickle_bin = file.read(len_next)
            except Exception as e:
                logging.exception(f"Failed to load pickle file. Exception: {e}")
                return test_results
            len_next = file.read(4)
            len_next = int.from_bytes(len_next, byteorder="big")
            loop_index = file.read(len_next).decode("ascii")
            len_next = file.read(4)
            len_next = int.from_bytes(len_next, byteorder="big")
            invocation_id = file.read(len_next).decode("ascii")

            invocation_id_object = InvocationId.from_str_id(encoded_test_name, invocation_id)
            test_file_path = file_path_from_module_name(
                invocation_id_object.test_module_path,
                test_config.project_root_path,
            )
            # test_type = test_types[test_file_paths.index(test_file_path)]

            test_type = next(
                (
                    test_file.test_type
                    for test_file in test_files.test_files
                    if test_file.instrumented_file_path == test_file_path
                ),
                None,
            )

            # instrumented_file_path: str
            # original_file_path: Optional[str]
            # original_source: Optional[str]
            # test_type: TestType

            test_pickle = pickle.loads(test_pickle_bin) if loop_index == "1" else None
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
    sqlite_file_path: str,
    test_file_paths: list[str],
    test_types: list[TestType],
    test_config: TestConfig,
) -> TestResults:
    test_results = TestResults()
    if not os.path.exists(sqlite_file_path):
        logging.warning(f"No test results for {sqlite_file_path} found.")
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
            test_type = test_types[test_file_paths.index(test_file_path)]
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
                    return_value=pickle.loads(val[7]) if loop_index == "1" else None,
                    timed_out=False,
                ),
            )
        except Exception:
            logging.exception("Failed to load pickle file.")
        # Hardcoding the test result to True because the test did execute and we are only interested in the return values,
        # the did_pass comes from the xml results file
    return test_results


def parse_test_xml(
    test_xml_file_path: str,
    test_py_file_paths: list[str],
    test_types: list[TestType],
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    test_results = TestResults()
    # Parse unittest output
    if not os.path.exists(test_xml_file_path):
        logging.warning(f"No test results for {test_xml_file_path} found.")
        return test_results
    try:
        xml = JUnitXml.fromfile(test_xml_file_path)
    except Exception as e:
        logging.warning(
            f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}",
        )
        return test_results
    test_module_paths = []
    for file_path in test_py_file_paths:
        assert os.path.exists(file_path), f"File {file_path} doesn't exist."
        test_module_path = module_name_from_file_path(file_path, test_config.project_root_path)
        if test_module_path.endswith("__perfinstrumented"):
            test_module_path = test_module_path[: -len("__perfinstrumented")]
        test_module_paths.append(test_module_path)

    for suite in xml:
        for testcase in suite:
            class_name = testcase.classname
            test_module_path = (
                class_name[: -len("__perfinstrumented")]
                if class_name.endswith("__perfinstrumented")
                else class_name
            )
            test_type = test_types[test_module_paths.index(test_module_path)]
            file_name = file_path_from_module_name(test_module_path, test_config.project_root_path)
            if (
                file_name == f"unittest{os.sep}loader.py"
                and class_name == "unittest.loader._FailedTest"
                and suite.errors == 1
                and suite.tests == 1
            ):
                # This means that the test failed to load, so we don't want to crash on it
                logging.info("Test failed to load, skipping it.")
                if run_result is not None:
                    logging.info(
                        f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}",
                    )
                return test_results

            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_class = None
            if class_name is not None:
                for test_module_path in test_module_paths:
                    if class_name.startswith(test_module_path):
                        test_class = class_name[len(test_module_path) + 1 :]

            test_function = testcase.name.split("[", 1)[0] if "[" in testcase.name else testcase.name
            loop_index = "1"
            if test_function is None:
                logging.warning(
                    f"testcase.name is None in parse_test_xml for testcase {testcase!r} in file {test_xml_file_path}",
                )
                continue
            timed_out = False
            if test_config.test_framework == "pytest":
                loop_index = testcase.name.split("[ ", 1)[1][:-2] if "[" in testcase.name else "1"
                if len(testcase.result) > 1:
                    print(
                        f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!",
                    )
                if len(testcase.result) == 1:
                    message = testcase.result[0].message.lower()
                    if "failed: timeout >" in message:
                        timed_out = True
            else:
                if len(testcase.result) > 1:
                    print(
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
                (
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
                            file_name=file_name,
                            runtime=None,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                        ),
                    ),
                )
            else:
                for match in matches:
                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=match[4],
                            id=InvocationId(
                                test_module_path=match[0],
                                test_class_name=None if match[1] == "" else match[1][:-1],
                                test_function_name=match[2],
                                function_getting_tested=match[3],
                                iteration_id=match[5],
                            ),
                            file_name=file_name,
                            runtime=None,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                        ),
                    )

    if not test_results:
        logging.info(f"Tests '{test_py_file_paths}' failed to run, skipping")
        if run_result is not None:
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.std_result.stderr
            logging.debug(
                f"Test log - STDOUT : {stdout} \n STDERR : {stderr}",
            )
    return test_results


def merge_test_results(
    xml_test_results: TestResults,
    bin_test_results: TestResults,
    test_framework: str,
) -> TestResults:
    merged_test_results = TestResults()

    grouped_xml_results = defaultdict(TestResults)
    grouped_bin_results = defaultdict(TestResults)

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
            + result.loop_index
        ].add(result)

    for result in bin_test_results:
        grouped_bin_results[
            result.id.test_module_path
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + result.id.test_function_name
            + ":"
            + result.loop_index
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
    test_xml_path: str,
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
        test_results_bin_file = parse_test_return_values_bin(
            get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin"),
            test_files=test_files,
            test_config=test_config,
        )
    except AttributeError as e:
        logging.exception(e)
        test_results_bin_file = TestResults()
        pathlib.Path(
            get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin"),
        ).unlink(missing_ok=True)

    try:
        test_results_bin_file.merge(
            parse_sqlite_test_results(
                sqlite_file_path=get_run_tmp_file(
                    f"test_return_values_{optimization_iteration}.sqlite",
                ),
                test_files=test_files,
                test_config=test_config,
            ),
        )
    except AttributeError as e:
        logging.exception(e)

    pathlib.Path(
        get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin"),
    ).unlink(
        missing_ok=True,
    )
    pathlib.Path(
        get_run_tmp_file(f"test_return_values_{optimization_iteration}.sqlite"),
    ).unlink(
        missing_ok=True,
    )

    return merge_test_results(
        test_results_xml,
        test_results_bin_file,
        test_config.test_framework,
    )
