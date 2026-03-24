from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import dill as pickle
from lxml.etree import XMLParser, parse  # type: ignore[import-not-found]

from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType, VerificationType
from codeflash_python.code_utils.code_utils import get_run_tmp_file
from codeflash_python.verification.path_utils import file_path_from_module_name
from codeflash_python.verification.test_output_utils import merge_test_results, parse_test_failures_from_stdout

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import CodeOptimizationContext, CoverageData, TestFiles
    from codeflash_core.config import TestConfig

logger = logging.getLogger("codeflash_python")
DEBUG_MODE = os.environ.get("CODEFLASH_DEBUG", "").lower() in ("1", "true")


def parse_func(file_path: Path) -> XMLParser:
    """Parse the XML file with lxml.etree.XMLParser as the backend."""
    xml_parser = XMLParser(huge_tree=True)
    return parse(file_path, xml_parser)


def parse_test_return_values_bin(file_location: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not file_location.exists():
        logger.debug("No test results for %s found.", file_location)
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
                        logger.exception("Failed to load pickle file for %s Exception: %s", encoded_test_name, e)
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
            logger.warning("Failed to parse test results from %s. Exception: %s", file_location, e)
            return test_results
    return test_results


def parse_sqlite_test_results(sqlite_file_path: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not sqlite_file_path.exists():
        logger.warning("No test results for %s found.", sqlite_file_path)
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
        logger.warning("Failed to parse test results from %s. Exception: %s", sqlite_file_path, e)
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
                # Try original_file_path first (for existing tests that were instrumented)
                test_type = test_files.get_test_type_by_original_file_path(test_file_path)
                logger.debug("[PARSE-DEBUG] test_module=%s, test_file_path=%s", test_module_path, test_file_path)
                logger.debug("[PARSE-DEBUG]   by_original_file_path: %s", test_type)
                # If not found, try instrumented_behavior_file_path (for generated tests)
                if test_type is None:
                    test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
                    logger.debug("[PARSE-DEBUG]   by_instrumented_file_path: %s", test_type)
                if test_type is None:
                    # Skip results where test type cannot be determined
                    logger.debug("Skipping result for %s: could not determine test type", test_function_name)
                    continue
                logger.debug("[PARSE-DEBUG]   FINAL test_type=%s", test_type)

            ret_val = None
            if loop_index == 1 and val[7]:
                try:
                    ret_val = (pickle.loads(val[7]),)
                except Exception as e:
                    logger.debug("Failed to deserialize return value for %s: %s", test_function_name, e)
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
            logger.exception("Failed to parse sqlite test results for %s", sqlite_file_path)
        # Hardcoding the test result to True because the test did execute and we are only interested in the return values,
        # the did_pass comes from the xml results file
    return test_results


def parse_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    from codeflash_python.verification.parse_xml import parse_python_test_xml

    return parse_python_test_xml(test_xml_file_path, test_files, test_config, run_result)


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
) -> tuple[TestResults, CoverageData | None]:
    test_results_xml = parse_test_xml(
        test_xml_path, test_files=test_files, test_config=test_config, run_result=run_result
    )

    test_results_data = TestResults()

    try:
        sql_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite"))
        if sql_results_file.exists():
            test_results_data = parse_sqlite_test_results(
                sqlite_file_path=sql_results_file, test_files=test_files, test_config=test_config
            )
            logger.debug("Parsed %s results from SQLite", len(test_results_data.test_results))
    except Exception as e:
        logger.exception("Failed to parse SQLite test results: %s", e)

    try:
        bin_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin"))
        if bin_results_file.exists():
            bin_test_results = parse_test_return_values_bin(
                bin_results_file, test_files=test_files, test_config=test_config
            )
            for result in bin_test_results:
                test_results_data.add(result)
            logger.debug("Merged %s results from binary file", len(bin_test_results))
    except AttributeError as e:
        logger.exception(e)

    # Cleanup temp files
    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin")).unlink(missing_ok=True)

    get_run_tmp_file(Path("pytest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path("unittest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite")).unlink(missing_ok=True)

    results = merge_test_results(test_results_xml, test_results_data, test_config.test_framework)

    all_args = False
    coverage = None
    if coverage_database_file and source_file and code_context and function_name:
        all_args = True
        from codeflash_python.verification.coverage_utils import CoverageUtils

        coverage = CoverageUtils.load_from_sqlite_database(
            database_path=coverage_database_file,
            config_path=coverage_config_file,  # type: ignore[invalid-argument-type]
            source_code_path=source_file,
            code_context=code_context,
            function_name=function_name,
        )
        if coverage:
            coverage.log_coverage()
    if run_result:
        try:
            failures = parse_test_failures_from_stdout(run_result.stdout)
            results.test_failures = failures
        except Exception as e:
            logger.exception(e)

    return results, coverage if all_args else None
