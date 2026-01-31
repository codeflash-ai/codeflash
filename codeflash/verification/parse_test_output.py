from __future__ import annotations

import contextlib
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
from codeflash.languages import is_java, is_javascript
from codeflash.models.models import (
    ConcurrencyMetrics,
    FunctionTestInvocation,
    InvocationId,
    TestResults,
    TestType,
    VerificationType,
)
from codeflash.verification.coverage_utils import CoverageUtils, JacocoCoverageUtils, JestCoverageUtils

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


start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")

# Jest timing marker patterns (from codeflash-jest-helper.js console.log output)
# Format: !$######testName:testName:funcName:loopIndex:lineId######$! (start)
# Format: !######testName:testName:funcName:loopIndex:lineId:durationNs######! (end)
jest_start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")
jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")


def calculate_function_throughput_from_test_results(test_results: TestResults, function_name: str) -> int:
    """Calculate function throughput from TestResults by extracting performance stdout.

    A completed execution is defined as having both a start tag and matching end tag from performance wrappers.
    Start: !$######test_module:test_function:function_name:loop_index:iteration_id######$!
    End:   !######test_module:test_function:function_name:loop_index:iteration_id:duration######!
    """
    start_matches = start_pattern.findall(test_results.perf_stdout or "")
    end_matches = end_pattern.findall(test_results.perf_stdout or "")

    end_matches_truncated = [end_match[:5] for end_match in end_matches]
    end_matches_set = set(end_matches_truncated)

    function_throughput = 0
    for start_match in start_matches:
        if start_match in end_matches_set and len(start_match) > 2 and start_match[2] == function_name:
            function_throughput += 1
    return function_throughput


# Pattern for concurrency benchmark output:
# !@######CONC:module:class:test:function:loop_index:seq_time:conc_time:factor######@!
_concurrency_pattern = re.compile(r"!@######CONC:([^:]*):([^:]*):([^:]*):([^:]*):([^:]*):(\d+):(\d+):(\d+)######@!")


def parse_concurrency_metrics(test_results: TestResults, function_name: str) -> ConcurrencyMetrics | None:
    """Parse concurrency benchmark results from test output.

    Format: !@######CONC:module:class:test:function:loop_index:seq_time:conc_time:factor######@!

    Returns ConcurrencyMetrics with:
    - sequential_time_ns: Total time for N sequential executions
    - concurrent_time_ns: Total time for N concurrent executions
    - concurrency_factor: N (number of concurrent executions)
    - concurrency_ratio: sequential_time / concurrent_time (higher = better concurrency)
    """
    if not test_results.perf_stdout:
        return None

    matches = _concurrency_pattern.findall(test_results.perf_stdout)
    if not matches:
        return None

    # Aggregate metrics for the target function
    total_seq, total_conc, factor, count = 0, 0, 0, 0
    for match in matches:
        # match[3] is function_name
        if len(match) >= 8 and match[3] == function_name:
            total_seq += int(match[5])
            total_conc += int(match[6])
            factor = int(match[7])
            count += 1

    if count == 0:
        return None

    avg_seq = total_seq / count
    avg_conc = total_conc / count
    ratio = avg_seq / avg_conc if avg_conc > 0 else 1.0

    return ConcurrencyMetrics(
        sequential_time_ns=int(avg_seq),
        concurrent_time_ns=int(avg_conc),
        concurrency_factor=factor,
        concurrency_ratio=ratio,
    )


def resolve_test_file_from_class_path(test_class_path: str, base_dir: Path) -> Path | None:
    """Resolve test file path from pytest's test class path or Java class path.

    This function handles various cases where pytest's classname in JUnit XML
    includes parent directories that may already be part of base_dir.

    Args:
        test_class_path: The full class path from pytest (e.g., "project.tests.test_file.TestClass")
            or a file path from Jest (e.g., "tests/test_file.test.js")
            or a Java class path (e.g., "com.example.AlgorithmsTest")
        base_dir: The base directory for tests (tests project root)

    Returns:
        Path to the test file if found, None otherwise

    Examples:
        >>> # base_dir = "/path/to/tests"
        >>> # test_class_path = "code_to_optimize.tests.unittest.test_file.TestClass"
        >>> # Should find: /path/to/tests/unittest/test_file.py

    """
    # Handle Java class paths (convert dots to path and add .java extension)
    # Java class paths look like "com.example.TestClass" and should map to
    # src/test/java/com/example/TestClass.java
    if is_java():
        # Convert dots to path separators
        relative_path = test_class_path.replace(".", "/") + ".java"

        # Try various locations
        # 1. Directly under base_dir
        potential_path = base_dir / relative_path
        if potential_path.exists():
            return potential_path

        # 2. Under src/test/java relative to project root
        project_root = base_dir.parent if base_dir.name == "java" else base_dir
        while project_root.name not in ("", "/") and not (project_root / "pom.xml").exists():
            project_root = project_root.parent
        if (project_root / "pom.xml").exists():
            potential_path = project_root / "src" / "test" / "java" / relative_path
            if potential_path.exists():
                return potential_path

        # 3. Search for the file in base_dir and its subdirectories
        file_name = test_class_path.split(".")[-1] + ".java"
        for java_file in base_dir.rglob(file_name):
            return java_file

        return None

    # Handle file paths (contain slashes and extensions like .js/.ts)
    if "/" in test_class_path or "\\" in test_class_path:
        # This is a file path, not a Python module path
        # Try to resolve relative to base_dir's parent (project root)
        project_root = base_dir.parent
        potential_path = project_root / test_class_path
        if potential_path.exists():
            return potential_path
        # Also try relative to base_dir itself
        potential_path = base_dir / test_class_path
        if potential_path.exists():
            return potential_path
        # Try the path as-is if it's absolute
        potential_path = Path(test_class_path)
        if potential_path.exists():
            return potential_path
        return None

    # First try the full path (Python module path)
    test_file_path = file_name_from_test_module_name(test_class_path, base_dir)

    # If we couldn't find the file, try stripping the last component (likely a class name)
    # This handles cases like "module.TestClass" where TestClass is a class, not a module
    if test_file_path is None and "." in test_class_path:
        module_without_class = ".".join(test_class_path.split(".")[:-1])
        test_file_path = file_name_from_test_module_name(module_without_class, base_dir)

    # If still not found, progressively strip prefix components
    # This handles cases where pytest's classname includes parent directories that are
    # already part of base_dir (e.g., "project.tests.unittest.test_file.TestClass"
    # when base_dir is "/.../tests")
    if test_file_path is None:
        parts = test_class_path.split(".")
        # Try stripping 1, 2, 3, ... prefix components
        for num_to_strip in range(1, len(parts)):
            remaining = ".".join(parts[num_to_strip:])
            test_file_path = file_name_from_test_module_name(remaining, base_dir)
            if test_file_path:
                break
            # Also try without the last component (class name)
            if "." in remaining:
                remaining_no_class = ".".join(remaining.split(".")[:-1])
                test_file_path = file_name_from_test_module_name(remaining_no_class, base_dir)
                if test_file_path:
                    break

    return test_file_path


def parse_jest_json_results(
    file_location: Path, test_files: TestFiles, test_config: TestConfig, function_name: str | None = None
) -> TestResults:
    """Parse Jest test results from JSON format written by codeflash-jest-helper.

    Args:
        file_location: Path to the JSON results file.
        test_files: TestFiles object containing test file information.
        test_config: Test configuration.
        function_name: Name of the function being tested.

    Returns:
        TestResults containing parsed test invocations.

    """
    import json

    test_results = TestResults()
    if not file_location.exists():
        logger.debug(f"No Jest JSON results at {file_location}")
        return test_results

    try:
        with file_location.open("r") as f:
            data = json.load(f)

        results = data.get("results", [])
        for result in results:
            test_name = result.get("testName", "") or result.get("testFunctionName", "")
            func_name = result.get("funcName", "")
            duration_ns = result.get("durationNs", 0)
            loop_index = result.get("loopIndex", 1)
            invocation_id = result.get("invocationId", 0)
            error = result.get("error")
            result_module_path = result.get("testModulePath", "")

            # Try to find the test file from test_files by matching testModulePath
            test_file_path = None
            test_type = TestType.GENERATED_REGRESSION  # Default for Jest generated tests

            # If we have testModulePath from the result, use it to find the matching test file
            if result_module_path:
                # Convert module path to file path (e.g., "tests.test_foo.test" -> "tests/test_foo.test.js")
                expected_path = result_module_path.replace(".", "/")
                if not expected_path.endswith(".js"):
                    expected_path += ".js"

                for test_file in test_files.test_files:
                    # Check behavior path
                    if test_file.instrumented_behavior_file_path:
                        try:
                            rel_path = str(
                                test_file.instrumented_behavior_file_path.relative_to(test_config.tests_project_rootdir)
                            )
                        except ValueError:
                            rel_path = test_file.instrumented_behavior_file_path.name
                        if (
                            rel_path == expected_path
                            or rel_path.replace("/", ".").replace(".js", "") == result_module_path
                        ):
                            test_file_path = test_file.instrumented_behavior_file_path
                            test_type = test_file.test_type
                            break
                    # Check benchmarking path
                    if test_file.benchmarking_file_path:
                        try:
                            rel_path = str(
                                test_file.benchmarking_file_path.relative_to(test_config.tests_project_rootdir)
                            )
                        except ValueError:
                            rel_path = test_file.benchmarking_file_path.name
                        if (
                            rel_path == expected_path
                            or rel_path.replace("/", ".").replace(".js", "") == result_module_path
                        ):
                            test_file_path = test_file.benchmarking_file_path
                            test_type = test_file.test_type
                            break

            # Fallback: find the first test file that exists (legacy behavior)
            if test_file_path is None:
                for test_file in test_files.test_files:
                    if test_file.benchmarking_file_path and test_file.benchmarking_file_path.exists():
                        test_file_path = test_file.benchmarking_file_path
                        test_type = test_file.test_type
                        break
                    if test_file.instrumented_behavior_file_path and test_file.instrumented_behavior_file_path.exists():
                        test_file_path = test_file.instrumented_behavior_file_path
                        test_type = test_file.test_type
                        break

            if test_file_path is None:
                logger.debug(f"Could not find test file for Jest result: {test_name} (module: {result_module_path})")
                continue

            # Create invocation ID - use funcName from result or passed function_name
            function_getting_tested = func_name or function_name or "unknown"
            # For Jest tests, keep the relative file path with extension intact
            # (Python uses module_name_from_file_path which strips extensions)
            try:
                test_module_path = str(test_file_path.relative_to(test_config.tests_project_rootdir))
            except ValueError:
                test_module_path = test_file_path.name
            invocation_id_obj = InvocationId(
                test_module_path=test_module_path,
                test_class_name=None,
                test_function_name=test_name or func_name,
                function_getting_tested=function_getting_tested,
                iteration_id=str(invocation_id),
            )

            test_results.add(
                function_test_invocation=FunctionTestInvocation(
                    loop_index=loop_index,
                    id=invocation_id_obj,
                    file_name=test_file_path,
                    did_pass=error is None,
                    runtime=duration_ns,
                    test_framework=test_config.test_framework,
                    test_type=test_type,
                    return_value=result.get("returnValue"),
                    timed_out=False,
                    verification_type=VerificationType.FUNCTION_CALL,
                )
            )

    except Exception as e:
        logger.warning(f"Failed to parse Jest JSON results from {file_location}: {e}")

    return test_results


def parse_test_return_values_bin(file_location: Path, test_files: TestFiles, test_config: TestConfig) -> TestResults:
    test_results = TestResults()
    if not file_location.exists():
        logger.debug(f"No test results for {file_location} found.")
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

    # Check if this is a JavaScript or Java test (use JSON) or Python test (use pickle)
    is_jest = is_javascript()
    is_java_test = is_java()

    for val in data:
        try:
            test_module_path = val[0]
            test_class_name = val[1] if val[1] else None
            test_function_name = val[2] if val[2] else None
            function_getting_tested = val[3]

            # For Jest tests, test_module_path could be:
            # - A module-style path: "tests.fibonacci.test.ts" (dots as separators)
            # - A file path: "tests/fibonacci.test.ts" (slashes as separators)
            # For Python, it's a module path (e.g., "tests.test_foo") that needs conversion
            if is_jest:
                # Jest test file extensions (including .test.ts, .spec.ts patterns)
                jest_test_extensions = (
                    ".test.ts",
                    ".test.js",
                    ".test.tsx",
                    ".test.jsx",
                    ".spec.ts",
                    ".spec.js",
                    ".spec.tsx",
                    ".spec.jsx",
                    ".ts",
                    ".js",
                    ".tsx",
                    ".jsx",
                    ".mjs",
                    ".mts",
                )
                # Check if it's a module-style path (no slashes, has dots beyond extension)
                if "/" not in test_module_path and "\\" not in test_module_path:
                    # Find the appropriate extension to preserve
                    extension = ""
                    for ext in jest_test_extensions:
                        if test_module_path.endswith(ext):
                            extension = ext
                            break
                    if extension:
                        # Convert module-style path to file path
                        # "tests.fibonacci__perfinstrumented.test.ts" -> "tests/fibonacci__perfinstrumented.test.ts"
                        base_path = test_module_path[: -len(extension)]
                        file_path = base_path.replace(".", os.sep) + extension
                        # Check if the module path includes the tests directory name
                        tests_dir_name = test_config.tests_project_rootdir.name
                        if file_path.startswith((tests_dir_name + os.sep, tests_dir_name + "/")):
                            # Module path includes "tests." - use project root parent
                            test_file_path = test_config.tests_project_rootdir.parent / file_path
                        else:
                            # Module path doesn't include tests dir - use tests root directly
                            test_file_path = test_config.tests_project_rootdir / file_path
                    else:
                        # No recognized extension, treat as-is
                        test_file_path = test_config.tests_project_rootdir / test_module_path
                else:
                    # Already a file path
                    test_file_path = test_config.tests_project_rootdir / test_module_path
            elif is_java_test:
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
                        if original_class == test_module_path or file_stem == test_module_path:
                            test_file_path = test_file.instrumented_behavior_file_path
                            break
                    # Check original file path
                    if test_file.original_file_path:
                        if test_file.original_file_path.stem == test_module_path:
                            test_file_path = test_file.original_file_path
                            break
                if test_file_path is None:
                    # Fallback: try to find by searching in tests_project_rootdir
                    java_files = list(test_config.tests_project_rootdir.rglob(f"*{test_module_path}*.java"))
                    if java_files:
                        test_file_path = java_files[0]
                    else:
                        logger.debug(f"Could not find Java test file for module path: {test_module_path}")
                        test_file_path = test_config.tests_project_rootdir / f"{test_module_path}.java"
            else:
                # Python: convert module path to file path
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
                logger.debug(f"[PARSE-DEBUG] test_module={test_module_path}, test_file_path={test_file_path}")
                logger.debug(f"[PARSE-DEBUG]   by_original_file_path: {test_type}")
                # If not found, try instrumented_behavior_file_path (for generated tests)
                if test_type is None:
                    test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
                    logger.debug(f"[PARSE-DEBUG]   by_instrumented_file_path: {test_type}")
                # Default to GENERATED_REGRESSION for Jest/Java tests when test type can't be determined
                if test_type is None and (is_jest or is_java_test):
                    test_type = TestType.GENERATED_REGRESSION
                    logger.debug(f"[PARSE-DEBUG]   defaulting to GENERATED_REGRESSION ({'Jest' if is_jest else 'Java'})")
                elif test_type is None:
                    # Skip results where test type cannot be determined
                    logger.debug(f"Skipping result for {test_function_name}: could not determine test type")
                    continue
                logger.debug(f"[PARSE-DEBUG]   FINAL test_type={test_type}")

            # Deserialize return value
            # For Jest/Java: Store as serialized JSON - comparison happens via language-specific comparator
            # For Python: Use pickle to deserialize
            ret_val = None
            if loop_index == 1 and val[7]:
                try:
                    if is_jest or is_java_test:
                        # Jest/Java comparison happens via language-specific comparator
                        # Store a marker indicating data exists but is not deserialized in Python
                        # For Java, val[7] is a JSON string from Gson serialization
                        ret_val = ("__serialized__", val[7])
                    else:
                        # Python uses pickle serialization
                        ret_val = (pickle.loads(val[7]),)
                except Exception as e:
                    # If deserialization fails, skip this result
                    logger.debug(f"Failed to deserialize return value for {test_function_name}: {e}")
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


def _extract_jest_console_output(suite_elem) -> str:
    """Extract console output from Jest's JUnit XML system-out element.

    Jest-junit writes console.log output as a JSON array in the testsuite's system-out.
    Each entry has: {"message": "...", "origin": "...", "type": "log"}

    Args:
        suite_elem: The testsuite lxml element

    Returns:
        Concatenated message content from all log entries

    """
    import json

    system_out_elem = suite_elem.find("system-out")
    if system_out_elem is None or system_out_elem.text is None:
        return ""

    raw_content = system_out_elem.text.strip()
    if not raw_content:
        return ""

    # Jest-junit wraps console output in a JSON array
    # Try to parse as JSON first
    try:
        log_entries = json.loads(raw_content)
        if isinstance(log_entries, list):
            # Extract message field from each log entry
            messages = []
            for entry in log_entries:
                if isinstance(entry, dict) and "message" in entry:
                    messages.append(entry["message"])
            return "\n".join(messages)
    except (json.JSONDecodeError, TypeError):
        # Not JSON - return as plain text (fallback for pytest-style output)
        pass

    return raw_content


# TODO: {Claude} we need to move to the support directory.
def parse_jest_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    """Parse Jest JUnit XML test results.

    Jest-junit has a different structure than pytest:
    - system-out is at the testsuite level (not testcase)
    - system-out contains a JSON array of log entries
    - Timing markers are in the message field of log entries

    Args:
        test_xml_file_path: Path to the Jest JUnit XML file
        test_files: TestFiles object with test file information
        test_config: Test configuration
        run_result: Optional subprocess result for logging

    Returns:
        TestResults containing parsed test invocations

    """
    test_results = TestResults()

    if not test_xml_file_path.exists():
        logger.warning(f"No Jest test results for {test_xml_file_path} found.")
        return test_results

    # Log file size for debugging
    file_size = test_xml_file_path.stat().st_size
    logger.debug(f"Jest XML file size: {file_size} bytes at {test_xml_file_path}")

    try:
        xml = JUnitXml.fromfile(str(test_xml_file_path), parse_func=parse_func)
        logger.debug(f"Successfully parsed Jest JUnit XML from {test_xml_file_path}")
    except Exception as e:
        logger.warning(f"Failed to parse {test_xml_file_path} as JUnitXml. Exception: {e}")
        return test_results

    base_dir = test_config.tests_project_rootdir
    logger.debug(f"Jest XML parsing: base_dir={base_dir}, num_test_files={len(test_files.test_files)}")

    # Build lookup from instrumented file path to TestFile for direct matching
    # This handles cases where instrumented files are in temp directories
    instrumented_path_lookup: dict[str, tuple[Path, TestType]] = {}
    for test_file in test_files.test_files:
        if test_file.instrumented_behavior_file_path:
            # Store both the absolute path and resolved path as keys
            abs_path = str(test_file.instrumented_behavior_file_path.resolve())
            instrumented_path_lookup[abs_path] = (test_file.instrumented_behavior_file_path, test_file.test_type)
            # Also store the string representation in case of minor path differences
            instrumented_path_lookup[str(test_file.instrumented_behavior_file_path)] = (
                test_file.instrumented_behavior_file_path,
                test_file.test_type,
            )
            logger.debug(f"Jest XML lookup: registered {abs_path}")

    # Fallback: if JUnit XML doesn't have system-out, use subprocess stdout directly
    global_stdout = ""
    if run_result is not None:
        try:
            global_stdout = run_result.stdout if isinstance(run_result.stdout, str) else run_result.stdout.decode()
            # Debug: log if timing markers are found in stdout
            if global_stdout:
                marker_count = len(jest_start_pattern.findall(global_stdout))
                if marker_count > 0:
                    logger.debug(f"Found {marker_count} timing start markers in Jest stdout")
                else:
                    logger.debug(f"No timing start markers found in Jest stdout (len={len(global_stdout)})")
        except (AttributeError, UnicodeDecodeError):
            global_stdout = ""

    suite_count = 0
    testcase_count = 0
    for suite in xml:
        suite_count += 1
        # Extract console output from suite-level system-out (Jest specific)
        suite_stdout = _extract_jest_console_output(suite._elem)  # noqa: SLF001

        # Fallback: use subprocess stdout if XML system-out is empty
        if not suite_stdout and global_stdout:
            suite_stdout = global_stdout

        # Parse timing markers from the suite's console output
        start_matches = list(jest_start_pattern.finditer(suite_stdout))
        end_matches_dict = {}
        for match in jest_end_pattern.finditer(suite_stdout):
            # Key: (testName, testName2, funcName, loopIndex, lineId)
            key = match.groups()[:5]
            end_matches_dict[key] = match

        for testcase in suite:
            testcase_count += 1
            test_class_path = testcase.classname  # For Jest, this is the file path
            test_name = testcase.name

            if test_name is None:
                logger.debug(f"testcase.name is None in Jest XML {test_xml_file_path}, skipping")
                continue

            logger.debug(f"Jest XML: processing testcase name={test_name}, classname={test_class_path}")

            # First, try direct lookup in instrumented file paths
            # This handles cases where instrumented files are in temp directories
            test_file_path = None
            test_type = None

            if test_class_path:
                # Try exact match with classname (which should be the filepath from jest-junit)
                if test_class_path in instrumented_path_lookup:
                    test_file_path, test_type = instrumented_path_lookup[test_class_path]
                else:
                    # Try resolving the path and matching
                    try:
                        resolved_path = str(Path(test_class_path).resolve())
                        if resolved_path in instrumented_path_lookup:
                            test_file_path, test_type = instrumented_path_lookup[resolved_path]
                    except Exception:
                        pass

            # If direct lookup failed, try the file attribute
            if test_file_path is None:
                test_file_name = suite._elem.attrib.get("file") or testcase._elem.attrib.get("file")  # noqa: SLF001
                if test_file_name:
                    if test_file_name in instrumented_path_lookup:
                        test_file_path, test_type = instrumented_path_lookup[test_file_name]
                    else:
                        try:
                            resolved_path = str(Path(test_file_name).resolve())
                            if resolved_path in instrumented_path_lookup:
                                test_file_path, test_type = instrumented_path_lookup[resolved_path]
                        except Exception:
                            pass

            # Fall back to traditional path resolution if direct lookup failed
            if test_file_path is None:
                test_file_path = resolve_test_file_from_class_path(test_class_path, base_dir)
                if test_file_path is None:
                    test_file_name = suite._elem.attrib.get("file") or testcase._elem.attrib.get("file")  # noqa: SLF001
                    if test_file_name:
                        test_file_path = base_dir.parent / test_file_name
                        if not test_file_path.exists():
                            test_file_path = base_dir / test_file_name

            if test_file_path is None or not test_file_path.exists():
                logger.warning(f"Could not resolve test file for Jest test: {test_class_path}")
                continue

            # Get test type if not already set from lookup
            if test_type is None:
                test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            if test_type is None:
                # Default to GENERATED_REGRESSION for Jest tests
                test_type = TestType.GENERATED_REGRESSION

            # For Jest tests, keep the relative file path with extension intact
            # (Python uses module_name_from_file_path which strips extensions)
            try:
                test_module_path = str(test_file_path.relative_to(test_config.tests_project_rootdir))
            except ValueError:
                test_module_path = test_file_path.name
            result = testcase.is_passed

            # Check for timeout
            timed_out = False
            if len(testcase.result) >= 1:
                message = (testcase.result[0].message or "").lower()
                if "timeout" in message or "timed out" in message:
                    timed_out = True

            # Find matching timing markers for this test
            # Jest test names in markers are sanitized by codeflash-jest-helper's sanitizeTestId()
            # which replaces: !#: (space) ()[]{}|\/*?^$.+- with underscores
            # IMPORTANT: Must match Jest helper's sanitization exactly for marker matching to work
            # Pattern from capture.js: /[!#: ()\[\]{}|\\/*?^$.+\-]/g
            sanitized_test_name = re.sub(r"[!#: ()\[\]{}|\\/*?^$.+\-]", "_", test_name)
            matching_starts = [m for m in start_matches if sanitized_test_name in m.group(2)]

            # For performance tests (capturePerf), there are no START markers - only END markers with duration
            # Check for END markers directly if no START markers found
            matching_ends_direct = []
            if not matching_starts:
                # Look for END markers that match this test (performance test format)
                # END marker format: !######module:testName:funcName:loopIndex:invocationId:durationNs######!
                for end_key, end_match in end_matches_dict.items():
                    # end_key is (module, testName, funcName, loopIndex, invocationId)
                    if len(end_key) >= 2 and sanitized_test_name in end_key[1]:
                        matching_ends_direct.append(end_match)

            if not matching_starts and not matching_ends_direct:
                # No timing markers found - add basic result
                test_results.add(
                    FunctionTestInvocation(
                        loop_index=1,
                        id=InvocationId(
                            test_module_path=test_module_path,
                            test_class_name=None,
                            test_function_name=test_name,
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
            elif matching_ends_direct:
                # Performance test format: process END markers directly (no START markers)
                for end_match in matching_ends_direct:
                    groups = end_match.groups()
                    # groups: (module, testName, funcName, loopIndex, invocationId, durationNs)
                    func_name = groups[2]
                    loop_index = int(groups[3]) if groups[3].isdigit() else 1
                    line_id = groups[4]
                    try:
                        runtime = int(groups[5])
                    except (ValueError, IndexError):
                        runtime = None
                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=loop_index,
                            id=InvocationId(
                                test_module_path=test_module_path,
                                test_class_name=None,
                                test_function_name=test_name,
                                function_getting_tested=func_name,
                                iteration_id=line_id,
                            ),
                            file_name=test_file_path,
                            runtime=runtime,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                            stdout="",
                        )
                    )
            else:
                # Process each timing marker
                for match in matching_starts:
                    groups = match.groups()
                    # groups: (testName, testName2, funcName, loopIndex, lineId)
                    func_name = groups[2]
                    loop_index = int(groups[3]) if groups[3].isdigit() else 1
                    line_id = groups[4]

                    # Find matching end marker
                    end_key = groups[:5]
                    end_match = end_matches_dict.get(end_key)

                    runtime = None
                    if end_match:
                        # Duration is in the 6th group (index 5)
                        with contextlib.suppress(ValueError, IndexError):
                            runtime = int(end_match.group(6))
                    test_results.add(
                        FunctionTestInvocation(
                            loop_index=loop_index,
                            id=InvocationId(
                                test_module_path=test_module_path,
                                test_class_name=None,
                                test_function_name=test_name,
                                function_getting_tested=func_name,
                                iteration_id=line_id,
                            ),
                            file_name=test_file_path,
                            runtime=runtime,
                            test_framework=test_config.test_framework,
                            did_pass=result,
                            test_type=test_type,
                            return_value=None,
                            timed_out=timed_out,
                            stdout="",
                        )
                    )

    if not test_results:
        logger.info(
            f"No Jest test results parsed from {test_xml_file_path} "
            f"(found {suite_count} suites, {testcase_count} testcases)"
        )
        if run_result is not None:
            logger.debug(f"Jest stdout: {run_result.stdout[:1000] if run_result.stdout else 'empty'}")
    else:
        logger.debug(
            f"Jest XML parsing complete: {len(test_results.test_results)} results "
            f"from {suite_count} suites, {testcase_count} testcases"
        )

    return test_results


def parse_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
) -> TestResults:
    # Route to Jest-specific parser for JavaScript/TypeScript tests
    if is_javascript():
        return parse_jest_test_xml(test_xml_file_path, test_files, test_config, run_result)

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
    # Always use tests_project_rootdir since pytest is now the test runner for all frameworks
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
                    # TODO : This might not be true if the test is organized under a class
                    test_file_path = resolve_test_file_from_class_path(test_class_path, base_dir)

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
            if test_type is None:
                # Log registered paths for debugging
                registered_paths = [str(tf.instrumented_behavior_file_path) for tf in test_files.test_files]
                logger.warning(
                    f"Test type not found for '{test_file_path}'. "
                    f"Registered test files: {registered_paths}. Skipping test case."
                )
                continue
            test_module_path = module_name_from_file_path(test_file_path, test_config.tests_project_rootdir)
            result = testcase.is_passed  # TODO: See for the cases of ERROR and SKIPPED
            test_class = None
            if class_name is not None and class_name.startswith(test_module_path):
                test_class = class_name[len(test_module_path) + 1 :]  # +1 for the dot, gets Unittest class name

            loop_index = int(testcase.name.split("[ ")[-1][:-2]) if testcase.name and "[" in testcase.name else 1

            timed_out = False
            if len(testcase.result) > 1:
                logger.debug(f"!!!!!Multiple results for {testcase.name or '<None>'} in {test_xml_file_path}!!!")
            if len(testcase.result) == 1:
                message = testcase.result[0].message.lower()
                if "failed: timeout >" in message or "timed out" in message:
                    timed_out = True

            sys_stdout = testcase.system_out or ""

            # Use different patterns for Java (5-field start, 6-field end) vs Python (6-field both)
            # Java format: !$######module:class:func:loop:iter######$! (start)
            #              !######module:class:func:loop:iter:duration######! (end)
            if is_java():
                begin_matches = list(start_pattern.finditer(sys_stdout))
                end_matches = {}
                for match in end_pattern.finditer(sys_stdout):
                    groups = match.groups()
                    # Key is first 5 groups (module, class, func, loop, iter)
                    end_matches[groups[:5]] = match
            else:
                begin_matches = list(matches_re_start.finditer(sys_stdout))
                end_matches = {}
                for match in matches_re_end.finditer(sys_stdout):
                    groups = match.groups()
                    if len(groups[5].split(":")) > 1:
                        iteration_id = groups[5].split(":")[0]
                        groups = (*groups[:5], iteration_id)
                    end_matches[groups] = match

            if not begin_matches:
                # For Java tests, use the JUnit XML time attribute for runtime
                runtime_from_xml = None
                if is_java():
                    try:
                        # JUnit XML time is in seconds, convert to nanoseconds
                        # Use a minimum of 1000ns (1 microsecond) for any successful test
                        # to avoid 0 runtime being treated as "no runtime"
                        test_time = float(testcase.time) if hasattr(testcase, 'time') and testcase.time else 0.0
                        runtime_from_xml = max(int(test_time * 1_000_000_000), 1000)
                    except (ValueError, TypeError):
                        # If we can't get time from XML, use 1 microsecond as minimum
                        runtime_from_xml = 1000

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

                    # Java and Python have different marker formats:
                    # Java:   5 groups - (module, class, func, loop_index, iteration_id)
                    # Python: 6 groups - (module, class.test, _, func, loop_index, iteration_id)
                    if is_java():
                        # Java format: !$######module:class:func:loop:iter######$!
                        end_key = groups[:5]  # Use all 5 groups as key
                        end_match = end_matches.get(end_key)
                        iteration_id = groups[4]  # iter is at index 4
                        loop_idx = int(groups[3])  # loop is at index 3
                        test_module = groups[0]  # module
                        test_class_str = groups[1]  # class
                        test_func = test_function  # Use the testcase name from XML
                        func_getting_tested = groups[2]  # func being tested
                        runtime = None

                        if end_match:
                            stdout = sys_stdout[match.end() : end_match.start()]
                            runtime = int(end_match.groups()[5])  # duration is at index 5
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
                    else:
                        # Python format: 6 groups
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
        elif test_framework == "unittest":
            test_function_name = result.id.test_function_name
            is_parameterized, new_test_function_name, _ = discover_parameters_unittest(test_function_name)
            if is_parameterized:  # handle parameterized test
                test_function_name = new_test_function_name
        else:
            # Jest and other frameworks - use test function name as-is
            test_function_name = result.id.test_function_name

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
                # Prefer XML runtime (from stdout markers) if bin runtime is None/0
                # This is important for Jest perf tests which output timing to stdout, not SQLite
                merged_runtime = result_bin.runtime if result_bin.runtime else xml_result.runtime
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=xml_result.loop_index,
                        id=result_bin.id,
                        file_name=xml_result.file_name,
                        runtime=merged_runtime,
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
                # Prefer XML runtime (from stdout markers) if bin runtime is None/0
                # This is important for Jest perf tests which output timing to stdout, not SQLite
                merged_runtime = bin_result.runtime if bin_result.runtime else xml_result.runtime
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=xml_result.loop_index,
                        id=xml_result.id,
                        file_name=xml_result.file_name,
                        runtime=merged_runtime,
                        test_framework=xml_result.test_framework,
                        did_pass=bin_result.did_pass,
                        test_type=xml_result.test_type,
                        return_value=bin_result.return_value,
                        timed_out=xml_result.timed_out
                        if merged_runtime is None
                        else False,  # If runtime was measured, then the testcase did not time out
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
                # Prefer XML runtime (from stdout markers) if bin runtime is None/0
                # This is important for Jest perf tests which output timing to stdout, not SQLite
                merged_runtime = bin_result.runtime if bin_result.runtime else xml_result.runtime
                merged_test_results.add(
                    FunctionTestInvocation(
                        loop_index=bin_result.loop_index,
                        id=bin_result.id,
                        file_name=bin_result.file_name,
                        runtime=merged_runtime,
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


FAILURES_HEADER_RE = re.compile(r"=+ FAILURES =+")
TEST_HEADER_RE = re.compile(r"_{3,}\s*(.*?)\s*_{3,}$")


def parse_test_failures_from_stdout(stdout: str) -> dict[str, str]:
    """Extract individual pytest test failures from stdout grouped by test case qualified name, and add them to the test results."""
    lines = stdout.splitlines()
    start = end = None

    for i, line in enumerate(lines):
        if FAILURES_HEADER_RE.search(line.strip()):
            start = i
            break

    if start is None:
        return {}

    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if "short test summary info" in stripped:
            end = j
            break
        # any new === section === block
        if stripped.startswith("=") and stripped.count("=") > 3:
            end = j
            break

    # If no clear "end", just grap the rest of the string
    if end is None:
        end = len(lines)

    failure_block = lines[start:end]

    failures: dict[str, str] = {}
    current_name = None
    current_lines: list[str] = []

    for line in failure_block:
        m = TEST_HEADER_RE.match(line.strip())
        if m:
            if current_name is not None:
                failures[current_name] = "".join(current_lines)

            current_name = m.group(1)
            current_lines = []
        elif current_name:
            current_lines.append(line + "\n")

    if current_name:
        failures[current_name] = "".join(current_lines)

    return failures


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
    skip_sqlite_cleanup: bool = False,
) -> tuple[TestResults, CoverageData | None]:
    test_results_xml = parse_test_xml(
        test_xml_path, test_files=test_files, test_config=test_config, run_result=run_result
    )

    # Parse timing/behavior data from SQLite (used by both Python and Jest)
    # Jest uses SQLite exclusively via codeflash-jest-helper
    # Python can use SQLite (preferred) or legacy binary format
    test_results_data = TestResults()

    try:
        sql_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite"))
        if sql_results_file.exists():
            test_results_data = parse_sqlite_test_results(
                sqlite_file_path=sql_results_file, test_files=test_files, test_config=test_config
            )
            logger.debug(f"Parsed {len(test_results_data.test_results)} results from SQLite")
    except Exception as e:
        logger.exception(f"Failed to parse SQLite test results: {e}")

    # Also try to read legacy binary format for Python tests
    # Binary file may contain additional results (e.g., from codeflash_wrap) even if SQLite has data
    # from @codeflash_capture. We need to merge both sources.
    if not is_javascript():
        try:
            bin_results_file = get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin"))
            if bin_results_file.exists():
                bin_test_results = parse_test_return_values_bin(
                    bin_results_file, test_files=test_files, test_config=test_config
                )
                # Merge binary results with SQLite results
                for result in bin_test_results:
                    test_results_data.add(result)
                logger.debug(f"Merged {len(bin_test_results)} results from binary file")
        except AttributeError as e:
            logger.exception(e)

    # Cleanup temp files
    get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.bin")).unlink(missing_ok=True)

    get_run_tmp_file(Path("pytest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path("unittest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path("jest_results.xml")).unlink(missing_ok=True)
    get_run_tmp_file(Path("jest_perf_results.xml")).unlink(missing_ok=True)

    # For Jest tests, SQLite cleanup is deferred until after comparison
    # (comparison happens via language_support.compare_test_results)
    if not skip_sqlite_cleanup:
        get_run_tmp_file(Path(f"test_return_values_{optimization_iteration}.sqlite")).unlink(missing_ok=True)

    results = merge_test_results(test_results_xml, test_results_data, test_config.test_framework)

    all_args = False
    coverage = None
    if coverage_database_file and source_file and code_context and function_name:
        all_args = True
        if is_javascript():
            # Jest uses coverage-final.json (coverage_database_file points to this)
            coverage = JestCoverageUtils.load_from_jest_json(
                coverage_json_path=coverage_database_file,
                function_name=function_name,
                code_context=code_context,
                source_code_path=source_file,
            )
        elif is_java():
            # Java uses JaCoCo XML report (coverage_database_file points to jacoco.xml)
            coverage = JacocoCoverageUtils.load_from_jacoco_xml(
                jacoco_xml_path=coverage_database_file,
                function_name=function_name,
                code_context=code_context,
                source_code_path=source_file,
            )
        else:
            # Python uses coverage.py SQLite database
            coverage = CoverageUtils.load_from_sqlite_database(
                database_path=coverage_database_file,
                config_path=coverage_config_file,
                source_code_path=source_file,
                code_context=code_context,
                function_name=function_name,
            )
        coverage.log_coverage()
    try:
        failures = parse_test_failures_from_stdout(run_result.stdout)
        results.test_failures = failures
    except Exception as e:
        logger.exception(e)

    # Cleanup Jest coverage directory after coverage is parsed
    import shutil

    jest_coverage_dir = get_run_tmp_file(Path("jest_coverage"))
    if jest_coverage_dir.exists():
        shutil.rmtree(jest_coverage_dir, ignore_errors=True)

    return results, coverage if all_args else None
