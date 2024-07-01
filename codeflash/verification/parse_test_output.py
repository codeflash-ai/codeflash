import logging
import os
import pathlib
import re
import sqlite3
import subprocess
from collections import defaultdict
from typing import Optional

import dill as pickle
import sentry_sdk
from junitparser.xunit2 import JUnitXml

from codeflash.code_utils.code_utils import (
    get_run_tmp_file,
    module_name_from_file_path,
)
from codeflash.discovery.discover_unit_tests import discover_parameters_unittest
from codeflash.verification.test_results import (
    FunctionTestInvocation,
    InvocationId,
    TestResults,
    TestType,
)
from codeflash.verification.verification_utils import TestConfig


def parse_test_return_values_bin(
    file_location: str,
    test_framework: str,
    test_type: TestType,
    test_file_path: str,
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
                test_pickle = pickle.loads(file.read(len_next))
            except Exception as e:
                logging.exception(f"Failed to load pickle file. Exception: {e}")
                return test_results
            len_next = file.read(4)
            len_next = int.from_bytes(len_next, byteorder="big")
            invocation_id = file.read(len_next).decode("ascii")
            # TODO : Remove the fully loaded unpickled object from the test_results.
            #  replace it with a link to the pickle object. Load it only on demand.
            #  The problem is that the unpickled object might be huge. This could cause codeflash to crash
            #  due to out-of-memory. Plus as we fill memory, the benchmarking results will get skewed.
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId.from_str_id(encoded_test_name, invocation_id),
                    file_name=test_file_path,
                    did_pass=True,
                    runtime=duration,
                    test_framework=test_framework,
                    test_type=test_type,
                    return_value=test_pickle,
                    timed_out=False,
                ),
            )
            # Hardcoding the test result to True because the test did execute and we are only interested in the return values,
            # the did_pass comes from the xml results file
    return test_results


def parse_sqlite_test_results(
    sqlite_file_path: str,
    test_py_file_path: str,
    test_type: TestType,
    test_config: TestConfig,
):
    test_results = TestResults()
    if not os.path.exists(sqlite_file_path):
        logging.warning(f"No test results for {sqlite_file_path} found.")
        return test_results
    try:
        db = sqlite3.connect(sqlite_file_path)
        cur = db.cursor()
        data = cur.execute(
            "SELECT test_module_path , test_class_name , test_function_name , "
            "function_getting_tested , iteration_id , runtime, return_value  FROM test_results",
        ).fetchall()
    finally:
        db.close()
    for val in data:
        try:
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId(
                        test_module_path=val[0],
                        test_class_name=val[1],
                        test_function_name=val[2],
                        function_getting_tested=val[3],
                        iteration_id=val[4],
                    ),
                    file_name=test_py_file_path,
                    did_pass=True,
                    runtime=val[5],
                    test_framework=test_config.test_framework,
                    test_type=test_type,
                    return_value=pickle.loads(val[6]),
                    timed_out=False,
                ),
            )
        except Exception:
            logging.exception("Failed to load pickle file.")
        # return_value is only None temporarily as this is only being used for the existing tests. This should generalize
        # to read the return_value from the sqlite file as well.
        # Hardcoding the test result to True because the test did execute and we are only interested in the return values,
        # the did_pass comes from the xml results file
    return test_results


def parse_test_xml(
    test_xml_file_path: str,
    test_py_file_path: str,
    test_type: TestType,
    test_config: TestConfig,
    run_result: Optional[subprocess.CompletedProcess] = None,
) -> TestResults:
    test_results = TestResults()

    # Parse unittest output
    if not os.path.exists(test_xml_file_path):
        logging.warning(f"No test results for {test_xml_file_path} found.")
        return test_results
    try:
        xml = JUnitXml.fromfile(test_xml_file_path)
    except Exception as e:
        logging.warning(f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}")
        return test_results

    for suite in xml:
        for testcase in suite:
            class_name = testcase.classname
            file_name = suite._elem.attrib.get(
                "file",
            )  # file_path_from_module_name(generated_tests_path, test_config.project_root_path)
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
            file_name = test_py_file_path

            assert os.path.exists(file_name), f"File {file_name} doesn't exist."
            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_module_path = module_name_from_file_path(file_name, test_config.project_root_path)
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[
                    len(test_module_path) + 1 :
                ]  # +1 for the dot, gets Unittest class name
            # test_name = (test_class + "." if test_class else "") + testcase.name
            if test_module_path.endswith("__perfinstrumented"):
                test_module_path = test_module_path[: -len("__perfinstrumented")]
            test_function = testcase.name
            if test_function is None:
                with sentry_sdk.push_scope() as scope:
                    xml_file_contents = open(test_xml_file_path).read()
                    scope.set_extra("file", xml_file_contents)
                    sentry_sdk.capture_message(
                        f"testcase.name is None in parse_test_xml for testcase {testcase!r} in file {xml_file_contents}",
                    )
                continue
            timed_out = False
            if test_config.test_framework == "pytest":
                if len(testcase.result) > 1:
                    print(f"!!!!!Multiple results for {testcase.name} in {test_xml_file_path}!!!")
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
                r"!######(.*?):(.*?)([^\.:]*?):(.*?):(.*?)######!",
                testcase.system_out or "",
            )
            if not matches or not len(matches):
                test_results.add(
                    FunctionTestInvocation(
                        id=InvocationId(
                            test_module_path=test_module_path,
                            test_class_name=test_class,
                            test_function_name=test_function,
                            function_getting_tested="",  # FIXME,
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
                )
            else:
                for match in matches:
                    test_results.add(
                        FunctionTestInvocation(
                            id=InvocationId(
                                test_module_path=match[0],
                                test_class_name=None if match[1] == "" else match[1][:-1],
                                test_function_name=match[2],
                                function_getting_tested=match[3],
                                iteration_id=match[4],
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
    if len(test_results) == 0:
        logging.info(f"Test '{test_py_file_path}' failed to run, skipping it")
        if run_result is not None:
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.stdout
                stderr = run_result.stderr
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
            if "[" in result.id.test_function_name:  # handle parameterized test
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
            result.id.test_module_path + ":" + (result.id.test_class_name or "") + ":" + test_function_name
        ].add(result)
    for result in bin_test_results:
        grouped_bin_results[
            result.id.test_module_path
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + result.id.test_function_name
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
            for i in range(len(xml_results.test_results)):
                xml_result = xml_results.test_results[i]
                try:
                    bin_result = bin_results.get_by_id(xml_result.id)
                except AttributeError:
                    bin_result = None
                if bin_result is None:
                    merged_test_results.add(xml_result)
                    continue
                merged_test_results.add(
                    FunctionTestInvocation(
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
            for i in range(len(bin_results.test_results)):
                bin_result = bin_results.test_results[i]
                try:
                    xml_result = xml_results.test_results[i]
                except IndexError:
                    xml_result = None
                if xml_result is None:
                    merged_test_results.add(bin_result)
                    continue
                merged_test_results.add(
                    FunctionTestInvocation(
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
    test_py_path: str,
    test_config: TestConfig,
    test_type: TestType,
    optimization_iteration: int,
    run_result: Optional[subprocess.CompletedProcess] = None,
) -> TestResults:
    test_results_xml = parse_test_xml(
        test_xml_path,
        test_py_path,
        test_type=test_type,
        test_config=test_config,
        run_result=run_result,
    )
    # TODO: Merge these different conditions into one single unified sqlite parser
    if test_type == TestType.GENERATED_REGRESSION:
        try:
            test_results_bin_file = parse_test_return_values_bin(
                get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin"),
                test_framework=test_config.test_framework,
                test_type=TestType.GENERATED_REGRESSION,
                test_file_path=test_py_path,
            )
        except AttributeError as e:
            logging.exception(e)
            test_results_bin_file = TestResults()
            pathlib.Path(
                get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin"),
            ).unlink(missing_ok=True)
    elif test_type in [TestType.EXISTING_UNIT_TEST, TestType.REPLAY_TEST]:
        try:
            test_results_bin_file = parse_sqlite_test_results(
                get_run_tmp_file(f"test_return_values_{optimization_iteration}.sqlite"),
                test_py_file_path=test_py_path,
                test_type=test_type,
                test_config=test_config,
            )
        except AttributeError as e:
            logging.exception(e)
            test_results_bin_file = TestResults()
    else:
        raise ValueError(f"Invalid test type: {test_type}")

    # We Probably want to remove deleting this file here later, because we want to preserve the reference to the
    # pickle blob in the test_results
    pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin")).unlink(
        missing_ok=True,
    )
    pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_iteration}.sqlite")).unlink(
        missing_ok=True,
    )

    merged_results = merge_test_results(
        test_results_xml,
        test_results_bin_file,
        test_config.test_framework,
    )
    return merged_results
