r"""Python-specific JUnit XML parsing with 6-field timing markers.

Python uses extended 6-field markers:
  Start: !$######module:class_prefix.test_func:func_tested:loop_index:iteration_id######$!\n
  End:   !######module:class_prefix.test_func:func_tested:loop_index:iteration_id:runtime######!
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from junitparser.xunit2 import JUnitXml

from codeflash_python.code_utils.code_utils import module_name_from_file_path
from codeflash_python.models.models import FunctionTestInvocation, InvocationId, TestResults
from codeflash_python.verification.path_utils import file_path_from_module_name

logger = logging.getLogger("codeflash_python")

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path

    from lxml import etree  # type: ignore[import-not-found]

    from codeflash_core.config import TestConfig
    from codeflash_python.models.models import TestFiles

matches_re_start = re.compile(
    r"!\$######([^:]*)"  # group 1: module path
    r":((?:[^:.]*\.)*)"  # group 2: class prefix with trailing dot, or empty
    r"([^.:]*)"  # group 3: test function name
    r":([^:]*)"  # group 4: function being tested
    r":([^:]*)"  # group 5: loop index
    r":([^#]*)"  # group 6: iteration id
    r"######\$!\n"
)
matches_re_end = re.compile(
    r"!######([^:]*)"  # group 1: module path
    r":((?:[^:.]*\.)*)"  # group 2: class prefix with trailing dot, or empty
    r"([^.:]*)"  # group 3: test function name
    r":([^:]*)"  # group 4: function being tested
    r":([^:]*)"  # group 5: loop index
    r":([^#]*)"  # group 6: iteration_id or iteration_id:runtime
    r"######!"
)


def parse_func(file_path: Path) -> etree._ElementTree:
    from lxml.etree import XMLParser, parse  # type: ignore[import-not-found]

    xml_parser = XMLParser(huge_tree=True)
    return parse(file_path, xml_parser)


def parse_python_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    from codeflash_python.verification.test_output_utils import resolve_test_file_from_class_path

    test_results = TestResults()
    if not test_xml_file_path.exists():
        logger.warning("No test results for %s found.", test_xml_file_path)
        return test_results
    try:
        xml = JUnitXml.fromfile(str(test_xml_file_path), parse_func=parse_func)
    except Exception as e:
        logger.warning("Failed to parse %s as JUnitXml. Exception: %s", test_xml_file_path, e)
        return test_results
    base_dir = test_config.tests_project_rootdir

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
                logger.info("Test failed to load, skipping it.")
                if run_result is not None:
                    if isinstance(run_result.stdout, str) and isinstance(run_result.stderr, str):
                        logger.info("Test log - STDOUT : %s \nSTDERR : %s", run_result.stdout, run_result.stderr)
                    else:
                        logger.info(
                            "Test log - STDOUT : %s \nSTDERR : %s",
                            run_result.stdout.decode(),
                            run_result.stderr.decode(),
                        )
                return test_results

            test_class_path = testcase.classname
            if test_class_path and test_class_path.split(".")[0] in ("pytest", "_pytest"):
                logger.debug("Skipping pytest-internal test entry: %s", test_class_path)
                continue
            try:
                if testcase.name is None:
                    logger.debug(
                        "testcase.name is None for testcase %r in file %s, skipping", testcase, test_xml_file_path
                    )
                    continue
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
                    test_file_path = resolve_test_file_from_class_path(test_class_path, base_dir)
                    if test_file_path is None:
                        logger.warning("Could not find the test for file name - %s ", test_class_path)
                        continue
                else:
                    test_file_path = file_path_from_module_name(test_function, base_dir)
            else:
                test_file_path = base_dir / test_file_name
            assert test_file_path, f"Test file path not found for {test_file_name}"

            if not test_file_path.exists():
                logger.warning("Could not find the test for file name - %s ", test_file_path)
                continue
            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            if test_type is None:
                test_type = test_files.get_test_type_by_original_file_path(test_file_path)
            if test_type is None:
                registered_paths = [str(tf.instrumented_behavior_file_path) for tf in test_files.test_files]
                logger.warning(
                    "Test type not found for '%s'. Registered test files: %s. Skipping test case.",
                    test_file_path,
                    registered_paths,
                )
                continue
            test_module_path = module_name_from_file_path(test_file_path, test_config.tests_project_rootdir)
            result = testcase.is_passed
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[len(test_module_path) + 1 :]

            loop_index = int(testcase.name.split("[ ")[-1][:-2]) if testcase.name and "[" in testcase.name else 1

            timed_out = False
            if len(testcase.result) > 1:
                logger.debug("!!!!!Multiple results for %s in %s!!!", testcase.name or "<None>", test_xml_file_path)
            if len(testcase.result) == 1:
                message = testcase.result[0].message.lower()
                if "failed: timeout >" in message or "timed out" in message:
                    timed_out = True

            sys_stdout = testcase.system_out or ""

            begin_matches = list(matches_re_start.finditer(sys_stdout))
            end_matches: dict[tuple, re.Match] = {}
            for match in matches_re_end.finditer(sys_stdout):
                groups = match.groups()
                if len(groups[5].split(":")) > 1:
                    iteration_id = groups[5].split(":")[0]
                    groups = (*groups[:5], iteration_id)
                end_matches[groups] = match

            if not begin_matches:
                test_results.add(
                    FunctionTestInvocation(
                        loop_index=loop_index,
                        id=InvocationId(
                            test_module_path=test_module_path,
                            test_class_name=test_class,
                            test_function_name=test_function,
                            function_getting_tested="",
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
                    runtime = None

                    end_match = end_matches.get(groups)
                    iteration_id = groups[5]
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
            "Tests '%s' failed to run, skipping", [test_file.original_file_path for test_file in test_files.test_files]
        )
        if run_result is not None:
            stdout, stderr = "", ""
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.stderr
            logger.debug("Test log - STDOUT : %s \nSTDERR : %s", stdout, stderr)
    return test_results
