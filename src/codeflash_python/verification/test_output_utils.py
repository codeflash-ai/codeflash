"""Throughput/concurrency metrics, test file resolution, result merging, and failure parsing."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from codeflash_python.discovery.discover_unit_tests import discover_parameters_unittest
from codeflash_python.models.models import ConcurrencyMetrics, FunctionTestInvocation, TestResults, VerificationType
from codeflash_python.verification.path_utils import file_name_from_test_module_name

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("codeflash_python")


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


start_pattern = re.compile(r"!\$######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+)######\$!")
end_pattern = re.compile(r"!######([^:]*):([^:]*):([^:]*):([^:]*):([^:]+):([^:]+)######!")


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
    """Resolve test file path from pytest's test class path.

    This function handles various cases where pytest's classname in JUnit XML
    includes parent directories that may already be part of base_dir.

    Args:
        test_class_path: The full class path from pytest (e.g., "project.tests.test_file.TestClass")
        base_dir: The base directory for tests (tests project root)

    Returns:
        Path to the test file if found, None otherwise

    Examples:
        >>> # base_dir = "/path/to/tests"
        >>> # test_class_path = "code_to_optimize.tests.unittest.test_file.TestClass"
        >>> # Should find: /path/to/tests/unittest/test_file.py

    """
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


def merge_test_results(
    xml_test_results: TestResults, bin_test_results: TestResults, test_framework: str
) -> TestResults:
    merged_test_results = TestResults()

    grouped_xml_results: defaultdict[str, TestResults] = defaultdict(TestResults)
    grouped_bin_results: defaultdict[str, TestResults] = defaultdict(TestResults)

    # This is done to match the right iteration_id which might not be available in the xml
    for result in xml_test_results:
        if test_framework == "pytest":
            if (
                result.id.test_function_name
                and result.id.test_function_name.endswith("]")
                and "[" in result.id.test_function_name
            ):  # parameterized test
                test_function_name = result.id.test_function_name[: result.id.test_function_name.index("[")]
            else:
                test_function_name = result.id.test_function_name
        elif test_framework == "unittest":
            test_function_name = result.id.test_function_name
            if test_function_name:
                is_parameterized, new_test_function_name, _ = discover_parameters_unittest(test_function_name)
                if is_parameterized:  # handle parameterized test
                    test_function_name = new_test_function_name
        else:
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
                        verification_type=VerificationType(result_bin.verification_type)
                        if result_bin.verification_type
                        else None,
                        stdout=xml_result.stdout,
                    )
                )

    return merged_test_results


TEST_HEADER_RE = re.compile(r"_{3,}\s*(.*?)\s*_{3,}$")


def parse_test_failures_from_stdout(stdout: str) -> dict[str, str]:
    """Extract individual pytest test failures from stdout grouped by test case qualified name, and add them to the test results."""
    lines = stdout.splitlines()
    start = end = None

    for i, line in enumerate(lines):
        if "= FAILURES =" in line:
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
