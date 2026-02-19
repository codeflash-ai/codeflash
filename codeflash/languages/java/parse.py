from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from junitparser.xunit2 import JUnitXml

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import file_path_from_module_name, module_name_from_file_path
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType, VerificationType

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import TestFiles
    from codeflash.verification.verification_utils import TestConfig


start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")


def resolve_java_test_file_from_class_path(test_class_path: str, base_dir: Path) -> Path | None:
    """Resolve Java test file path from a class path.

    Java class paths look like "com.example.TestClass" and should map to
    src/test/java/com/example/TestClass.java.
    """
    logger.debug(f"[RESOLVE] Input: test_class_path={test_class_path}, base_dir={base_dir}")
    # Convert dots to path separators
    relative_path = test_class_path.replace(".", "/") + ".java"

    # Try various locations
    # 1. Directly under base_dir
    potential_path = base_dir / relative_path
    logger.debug(f"[RESOLVE] Attempt 1: checking {potential_path}")
    if potential_path.exists():
        logger.debug(f"[RESOLVE] Attempt 1 SUCCESS: found {potential_path}")
        return potential_path

    # 2. Under src/test/java relative to project root
    project_root = base_dir.parent if base_dir.name == "java" else base_dir
    while project_root.name not in ("", "/") and not (project_root / "pom.xml").exists():
        project_root = project_root.parent
    if (project_root / "pom.xml").exists():
        potential_path = project_root / "src" / "test" / "java" / relative_path
        logger.debug(f"[RESOLVE] Attempt 2: checking {potential_path} (project_root={project_root})")
        if potential_path.exists():
            logger.debug(f"[RESOLVE] Attempt 2 SUCCESS: found {potential_path}")
            return potential_path

    # 3. Search for the file in base_dir and its subdirectories
    file_name = test_class_path.rsplit(".", maxsplit=1)[-1] + ".java"
    logger.debug(f"[RESOLVE] Attempt 3: rglob for {file_name} in {base_dir}")
    for java_file in base_dir.rglob(file_name):
        logger.debug(f"[RESOLVE] Attempt 3 SUCCESS: rglob found {java_file}")
        return java_file

    logger.warning(f"[RESOLVE] FAILED to resolve {test_class_path} in base_dir {base_dir}")
    return None


def resolve_java_test_file_from_module_path(
    test_module_path: str, test_files: TestFiles, tests_project_rootdir: Path
) -> Path:
    """Resolve Java test file path from a module/class name stored in SQLite."""
    # Java: test_module_path is the class name (e.g., "CounterTest")
    # We need to find the test file by searching for it in the test files
    test_file_path = None
    for test_file in test_files.test_files:
        # Check instrumented behavior file path
        if test_file.instrumented_behavior_file_path:
            # Java class name is stored without package prefix in SQLite
            # Check if the file name matches the module path
            file_stem = test_file.instrumented_behavior_file_path.stem
            # The instrumented file has __perfinstrumented suffix
            original_class = file_stem.replace("__perfinstrumented", "").replace("__perfonlyinstrumented", "")
            if test_module_path in (original_class, file_stem):
                test_file_path = test_file.instrumented_behavior_file_path
                break
        # Check original file path
        if test_file.original_file_path:
            if test_file.original_file_path.stem == test_module_path:
                test_file_path = test_file.original_file_path
                break
    if test_file_path is None:
        # Fallback: try to find by searching in tests_project_rootdir
        java_files = list(tests_project_rootdir.rglob(f"*{test_module_path}*.java"))
        if java_files:
            test_file_path = java_files[0]
        else:
            logger.debug(f"Could not find Java test file for module path: {test_module_path}")
            test_file_path = tests_project_rootdir / f"{test_module_path}.java"
    return test_file_path


def parse_java_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
    parse_func=None,
) -> TestResults:
    test_results = TestResults()
    if not test_xml_file_path.exists():
        logger.warning(f"No test results for {test_xml_file_path} found.")
        return test_results
    try:
        xml = JUnitXml.fromfile(str(test_xml_file_path), parse_func=parse_func)
    except Exception as e:
        logger.warning(f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}")
        return test_results

    base_dir = test_config.tests_project_rootdir
    logger.debug(f"[PARSE-XML] base_dir for resolution: {base_dir}")
    logger.debug(
        f"[PARSE-XML] Registered test files: {[str(tf.instrumented_behavior_file_path) for tf in test_files.test_files]}"
    )

    # For Java: pre-parse fallback stdout once (not per testcase) to avoid O(n²) complexity
    java_fallback_stdout = None
    java_fallback_begin_matches = None
    java_fallback_end_matches = None
    if run_result is not None:
        try:
            fallback_stdout = run_result.stdout if isinstance(run_result.stdout, str) else run_result.stdout.decode()
            begin_matches = list(start_pattern.finditer(fallback_stdout))
            if begin_matches:
                java_fallback_stdout = fallback_stdout
                java_fallback_begin_matches = begin_matches
                java_fallback_end_matches = {}
                for match in end_pattern.finditer(fallback_stdout):
                    groups = match.groups()
                    java_fallback_end_matches[groups[:5]] = match
                logger.debug(f"Java: Found {len(begin_matches)} timing markers in subprocess stdout (fallback)")
        except (AttributeError, UnicodeDecodeError):
            pass

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
                        logger.info(f"Test log - STDOUT : {run_result.stdout} \n STDERR : {run_result.stderr}")
                    else:
                        logger.info(
                            f"Test log - STDOUT : {run_result.stdout.decode()} \n STDERR : {run_result.stderr.decode()}"
                        )
                return test_results

            test_class_path = testcase.classname
            logger.debug(f"[PARSE-XML] Processing testcase: classname={test_class_path}, name={testcase.name}")
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
                    logger.debug(f"[PARSE-XML] Resolving test_class_path={test_class_path} in base_dir={base_dir}")
                    test_file_path = resolve_java_test_file_from_class_path(test_class_path, base_dir)

                    if test_file_path is None:
                        logger.error(
                            f"[PARSE-XML] ERROR: Could not resolve test_class_path={test_class_path}, base_dir={base_dir}"
                        )
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
            logger.debug(f"[PARSE-XML] Looking up test_type by instrumented_file_path: {test_file_path}")
            test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            logger.debug(f"[PARSE-XML] Lookup by instrumented path result: {test_type}")
            if test_type is None:
                logger.debug(f"[PARSE-XML] Looking up test_type by original_file_path: {test_file_path}")
                test_type = test_files.get_test_type_by_original_file_path(test_file_path)
                logger.debug(f"[PARSE-XML] Lookup by original path result: {test_type}")
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

            loop_index = int(testcase.name.split("[ ")[-1][:-2]) if testcase.name and "[" in testcase.name else 1

            timed_out = False
            if len(testcase.result) > 1:
                logger.debug(f"!!!!!Multiple results for {testcase.name or '<None>'} in {test_xml_file_path}!!!")
            if len(testcase.result) == 1:
                message = testcase.result[0].message
                if message is not None:
                    message = message.lower()
                    if "failed: timeout >" in message or "timed out" in message:
                        timed_out = True

            sys_stdout = testcase.system_out or ""

            begin_matches = list(start_pattern.finditer(sys_stdout))
            end_matches = {}
            for match in end_pattern.finditer(sys_stdout):
                groups = match.groups()
                end_matches[groups[:5]] = match

            # For Java: fallback to pre-parsed subprocess stdout when XML system-out has no timing markers
            # This happens when using JUnit Console Launcher directly (bypassing Maven)
            if not begin_matches and java_fallback_begin_matches is not None:
                sys_stdout = java_fallback_stdout
                begin_matches = java_fallback_begin_matches
                end_matches = java_fallback_end_matches

            if not begin_matches:
                runtime_from_xml = None

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
                        runtime=runtime_from_xml,
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

                    end_key = groups[:5]
                    end_match = end_matches.get(end_key)
                    iteration_id = groups[4]
                    loop_idx = int(groups[3])
                    test_module = groups[0]
                    test_class_str = groups[1]
                    test_func = test_function
                    func_getting_tested = groups[2]
                    runtime = None

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
                                test_class_name=test_class_str if test_class_str else None,
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
        test_paths_display = [
            str(test_file.instrumented_behavior_file_path or test_file.original_file_path)
            for test_file in test_files.test_files
        ]
        logger.info(f"Tests {test_paths_display} failed to run, skipping")
        if run_result is not None:
            stdout, stderr = "", ""
            try:
                stdout = run_result.stdout.decode()
                stderr = run_result.stderr.decode()
            except AttributeError:
                stdout = run_result.stderr
            logger.debug(f"Test log - STDOUT : {stdout} \n STDERR : {stderr}")
    return test_results
