"""Java-specific JUnit XML parsing with 5-field compact timing markers.

Java uses compact 5-field markers:
  Start: !$######module:class.test:func:loop_index:iteration_id######$!
  End:   !######module:class.test:func:loop_index:iteration_id:runtime######!

Maven/Surefire may not capture per-test stdout in JUnit XML system-out,
so we also support fallback to subprocess stdout.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from junitparser.xunit2 import JUnitXml

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults

if TYPE_CHECKING:
    import subprocess
    from pathlib import Path

    from codeflash.models.models import TestFiles
    from codeflash.verification.verification_utils import TestConfig

start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")


def _parse_func(file_path: Path):
    from lxml.etree import XMLParser, parse

    xml_parser = XMLParser(huge_tree=True)
    return parse(file_path, xml_parser)


def parse_java_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    from codeflash.verification.parse_test_output import resolve_test_file_from_class_path

    test_results = TestResults()
    if not test_xml_file_path.exists():
        logger.warning(f"No test results for {test_xml_file_path} found.")
        console.rule()
        return test_results
    try:
        xml = JUnitXml.fromfile(str(test_xml_file_path), parse_func=_parse_func)
    except Exception as e:
        logger.warning(f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}")
        return test_results
    base_dir = test_config.tests_project_rootdir

    # Pre-parse fallback stdout once (not per testcase) to avoid O(n^2) complexity
    # Maven/Surefire doesn't always capture per-test stdout in JUnit XML system-out
    java_fallback_stdout = None
    java_fallback_begin_matches = None
    java_fallback_end_matches = None
    if run_result is not None:
        try:
            fallback_stdout = run_result.stdout if isinstance(run_result.stdout, str) else run_result.stdout.decode()
            _begin = list(start_pattern.finditer(fallback_stdout))
            if _begin:
                java_fallback_stdout = fallback_stdout
                java_fallback_begin_matches = _begin
                java_fallback_end_matches = {}
                for _m in end_pattern.finditer(fallback_stdout):
                    java_fallback_end_matches[_m.groups()[:5]] = _m
        except Exception:
            pass

    for suite in xml:
        for testcase in suite:
            class_name = testcase.classname
            test_file_name = suite._elem.attrib.get("file")  # noqa: SLF001

            test_class_path = testcase.classname
            try:
                if testcase.name is None:
                    logger.debug(
                        f"testcase.name is None for testcase {testcase!r} in file {test_xml_file_path}, skipping"
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
                        logger.warning(f"Could not find the test for file name - {test_class_path} ")
                        continue
                else:
                    from codeflash.code_utils.code_utils import file_path_from_module_name

                    test_file_path = file_path_from_module_name(test_function, base_dir)
            else:
                test_file_path = base_dir / test_file_name
            assert test_file_path, f"Test file path not found for {test_file_name}"

            if not test_file_path.exists():
                logger.warning(f"Could not find the test for file name - {test_file_path} ")
                continue
            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            if test_type is None:
                test_type = test_files.get_test_type_by_original_file_path(test_file_path)
            if test_type is None:
                registered_paths = [str(tf.instrumented_behavior_file_path) for tf in test_files.test_files]
                logger.warning(
                    f"Test type not found for '{test_file_path}'. "
                    f"Registered test files: {registered_paths}. Skipping test case."
                )
                continue
            test_module_path = module_name_from_file_path(test_file_path, test_config.tests_project_rootdir)
            result = testcase.is_passed
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[len(test_module_path) + 1 :]

            loop_index = 1
            if testcase.name and "[" in testcase.name:
                bracket_match = re.search(r"\[(\d+)\]", testcase.name)
                if bracket_match:
                    loop_index = int(bracket_match.group(1))

            timed_out = False
            if len(testcase.result) > 1:
                logger.debug(f"!!!!!Multiple results for {testcase.name or '<None>'} in {test_xml_file_path}!!!")
            if len(testcase.result) == 1:
                message = testcase.result[0]._elem.get("message", "").lower()
                if "failed: timeout >" in message or "timed out" in message:
                    timed_out = True

            sys_stdout = testcase.system_out or ""

            begin_matches = list(start_pattern.finditer(sys_stdout))
            end_matches: dict[tuple, re.Match] = {}
            for match in end_pattern.finditer(sys_stdout):
                end_matches[match.groups()[:5]] = match

            # Fallback to subprocess stdout when JUnit XML system-out has no markers
            if not begin_matches and java_fallback_begin_matches is not None:
                assert java_fallback_stdout is not None
                assert java_fallback_end_matches is not None
                sys_stdout = java_fallback_stdout
                begin_matches = java_fallback_begin_matches
                end_matches = java_fallback_end_matches

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

                    end_key = groups[:5]
                    end_match = end_matches.get(end_key)
                    iteration_id = groups[4]
                    loop_idx = int(groups[3])
                    test_module = groups[0]
                    class_test_field = groups[1]
                    if "." in class_test_field:
                        test_class_str, test_func = class_test_field.rsplit(".", 1)
                    else:
                        test_class_str = class_test_field
                        test_func = test_function
                    func_getting_tested = groups[2]

                    if end_match:
                        stdout = sys_stdout[match.end() : end_match.start()]
                        runtime = int(end_match.groups()[5])
                    elif match_index == len(begin_matches) - 1:
                        stdout = sys_stdout[match.end() :]
                    else:
                        stdout = sys_stdout[match.end() : begin_matches[match_index + 1].start()]

                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=loop_idx,
                            id=InvocationId(
                                test_module_path=test_module,
                                test_class_name=test_class_str,
                                test_function_name=test_func,
                                function_getting_tested=func_getting_tested,
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
