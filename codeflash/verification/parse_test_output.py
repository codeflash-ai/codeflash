import logging
import os
import pathlib
import pickle
import sqlite3
import subprocess
from collections import defaultdict
from typing import Optional

from junitparser.xunit2 import JUnitXml

from codeflash.code_utils.code_utils import (
    module_name_from_file_path,
    get_run_tmp_file,
)
from codeflash.verification.test_results import (
    TestResults,
    FunctionTestInvocation,
    TestType,
    InvocationId,
)
from codeflash.verification.verification_utils import TestConfig


def parse_test_return_values_bin(
    file_location: str, test_framework: str, test_type: TestType, test_file_path: str
) -> TestResults:
    test_results = TestResults()
    if not os.path.exists(file_location):
        logging.error(f"File {file_location} doesn't exist.")
        return test_results
    with open(file_location, "rb") as file:
        while file:
            len_next = file.read(4)
            if not len_next:
                return test_results
            len_next = int.from_bytes(len_next, byteorder="big")
            encoded_invocation_id = file.read(len_next).decode("ascii")
            len_next = file.read(8)
            duration = int.from_bytes(len_next, byteorder="big")
            len_next = file.read(4)
            if not len_next:
                return test_results
            len_next = int.from_bytes(len_next, byteorder="big")
            try:
                test_pickle = pickle.loads(file.read(len_next))
            except Exception as e:
                logging.error(f"Failed to load pickle file. Exception: {e}")
                return test_results
            # TODO : Remove the fully loaded unpickled object from the test_results.
            #  replace it with a link to the pickle object. Load it only on demand.
            #  The problem is that the unpickled object might be huge. This could cause codeflash to crash
            #  due to out-of-memory. Plus as we fill memory, the benchmarking results will get skewed.
            test_results.add(
                FunctionTestInvocation(
                    id=InvocationId.from_str_id(encoded_invocation_id),
                    file_name=test_file_path,
                    did_pass=True,
                    runtime=duration,
                    test_framework=test_framework,
                    test_type=test_type,
                    return_value=test_pickle,
                )
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
        logging.error(f"File {sqlite_file_path} doesn't exist.")
        return test_results
    db = sqlite3.connect(sqlite_file_path)
    cur = db.cursor()
    data = cur.execute(
        "SELECT test_module_path , test_class_name , test_function_name , "
        "function_getting_tested , iteration_id , runtime  FROM test_results",
    ).fetchall()
    for val in data:
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
                return_value=None,
            )
        )
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
        logging.warning(f"File {test_xml_file_path} doesn't exist.")
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
                "file"
            )  # file_path_from_module_name(generated_tests_path, test_config.project_root_path)
            if (
                file_name == "unittest/loader.py"
                and class_name == "unittest.loader._FailedTest"
                and suite.errors == 1
                and suite.tests == 1
            ):
                # This means that the test failed to load, so we don't want to crash on it
                logging.info(f"Test failed to load, skipping it.")
                if run_result is not None:
                    logging.info(
                        f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}"
                    )
                return test_results
            file_name = test_py_file_path

            assert os.path.exists(file_name), f"File {file_name} doesn't exist."
            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_module_path = module_name_from_file_path(file_name, test_config.project_root_path)
            test_class = None
            if class_name.startswith(test_module_path):
                test_class = class_name[
                    len(test_module_path) + 1 :
                ]  # +1 for the dot, gets Unittest class name
            # test_name = (test_class + "." if test_class else "") + testcase.name
            if test_module_path.endswith("__perfinstrumented"):
                test_module_path = test_module_path[: -len("__perfinstrumented")]
            test_function = testcase.name
            # Parse test timing
            # system_out_content = ""
            # for system_out in testcase.system_out:

            #     system_out_content += system_out.text
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
                )
            )
    if len(test_results) == 0:
        logging.info(f"Test '{test_py_file_path}' failed to run, skipping it")
        if run_result is not None:
            logging.info(
                f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}"
            )
    return test_results


def merge_test_results(xml_test_results: TestResults, bin_test_results: TestResults) -> TestResults:
    merged_test_results = TestResults()

    grouped_xml_results = defaultdict(TestResults)
    grouped_bin_results = defaultdict(TestResults)

    # This is done to match the right iteration_id which might not be available in the xml
    for result in xml_test_results:
        if "[" in result.id.test_function_name:
            test_function_name = result.id.test_function_name[
                                 : result.id.test_function_name.index("[")
                                 ]
        else:
            test_function_name = result.id.test_function_name

        grouped_xml_results[
            result.id.test_module_path
            + ":"
            + (result.id.test_class_name or "")
            + ":"
            + test_function_name
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
                    )
                )
        else:
            for i in range(len(xml_results.test_results)):
                xml_result = xml_results.test_results[i]
                try:
                    bin_result = bin_results.test_results[i]
                except IndexError:
                    bin_result = None
                if bin_result is None:
                    # if xml_result.test_type == TestType.EXISTING_UNIT_TEST:
                    # only support
                    logging.warning(f"Could not find bin result for xml result: {xml_result.id}")
                    merged_test_results.add(xml_result)
                    continue
                merged_test_results.add(
                    FunctionTestInvocation(
                        id=bin_result.id,
                        file_name=xml_result.file_name,
                        runtime=bin_result.runtime,
                        test_framework=xml_result.test_framework,
                        did_pass=xml_result.did_pass,
                        test_type=xml_result.test_type,
                        return_value=bin_result.return_value,
                    )
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
            logging.error(e)
            test_results_bin_file = TestResults()
            pathlib.Path(
                get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin")
            ).unlink(missing_ok=True)
    elif test_type == TestType.EXISTING_UNIT_TEST:
        try:
            test_results_bin_file = parse_sqlite_test_results(
                get_run_tmp_file(f"test_return_values_{optimization_iteration}.sqlite"),
                test_py_file_path=test_py_path,
                test_type=TestType.EXISTING_UNIT_TEST,
                test_config=test_config,
            )
        except AttributeError as e:
            logging.error(e)
            test_results_bin_file = TestResults()
    else:
        raise ValueError(f"Invalid test type: {test_type}")

    # We Probably want to remove deleting this file here later, because we want to preserve the reference to the
    # pickle blob in the test_results
    pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_iteration}.bin")).unlink(
        missing_ok=True
    )
    pathlib.Path(get_run_tmp_file(f"test_return_values_{optimization_iteration}.sqlite")).unlink(
        missing_ok=True
    )

    merged_results = merge_test_results(test_results_xml, test_results_bin_file)
    return merged_results
