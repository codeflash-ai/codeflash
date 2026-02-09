"""Jest/Vitest JUnit XML parsing for JavaScript/TypeScript tests.

This module handles parsing of JUnit XML test results produced by Jest and Vitest
test runners. It extracts test results, timing information, and maps them back
to instrumented test files.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from junitparser.xunit2 import JUnitXml

from codeflash.cli_cmds.console import logger
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults, TestType

if TYPE_CHECKING:
    import subprocess

    from codeflash.models.models import TestFiles
    from codeflash.verification.verification_utils import TestConfig


# Jest timing marker patterns (from codeflash-jest-helper.js console.log output)
# Format: !$######testName:testName:funcName:loopIndex:lineId######$! (start)
# Format: !######testName:testName:funcName:loopIndex:lineId:durationNs######! (end)
jest_start_pattern = re.compile(r"!\$######([^:]+):([^:]+):([^:]+):([^:]+):([^#]+)######\$!")
jest_end_pattern = re.compile(r"!######([^:]+):([^:]+):([^:]+):([^:]+):([^:]+):(\d+)######!")


def _extract_jest_console_output(suite_elem) -> str:
    """Extract console output from Jest's JUnit XML system-out element.

    Jest-junit writes console.log output as a JSON array in the testsuite's system-out.
    Each entry has: {"message": "...", "origin": "...", "type": "log"}

    Args:
        suite_elem: The testsuite lxml element

    Returns:
        Concatenated message content from all log entries

    """
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


def parse_jest_test_xml(
    test_xml_file_path: Path,
    test_files: TestFiles,
    test_config: TestConfig,
    run_result: subprocess.CompletedProcess | None = None,
    parse_func=None,
    resolve_test_file_from_class_path=None,
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
        parse_func: XML parser function (injected to avoid circular imports)
        resolve_test_file_from_class_path: Function to resolve test file paths (injected)

    Returns:
        TestResults containing parsed test invocations

    """
    test_results = TestResults()

    if not test_xml_file_path.exists():
        logger.warning(f"No JavaScript test results for {test_xml_file_path} found.")
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
        # Add behavior instrumented file paths
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
        # Also add benchmarking file paths (perf-only instrumented tests)
        if test_file.benchmarking_file_path:
            bench_abs_path = str(test_file.benchmarking_file_path.resolve())
            instrumented_path_lookup[bench_abs_path] = (test_file.benchmarking_file_path, test_file.test_type)
            instrumented_path_lookup[str(test_file.benchmarking_file_path)] = (
                test_file.benchmarking_file_path,
                test_file.test_type,
            )
            logger.debug(f"Jest XML lookup: registered benchmark {bench_abs_path}")

    # Also build a filename-only lookup for fallback matching
    # This handles cases where JUnit XML has relative paths that don't match absolute paths
    # e.g., JUnit has "test/utils__perfinstrumented.test.ts" but lookup has absolute paths
    filename_lookup: dict[str, tuple[Path, TestType]] = {}
    for test_file in test_files.test_files:
        # Add instrumented_behavior_file_path (behavior tests)
        if test_file.instrumented_behavior_file_path:
            filename = test_file.instrumented_behavior_file_path.name
            # Only add if not already present (avoid overwrites in case of duplicate filenames)
            if filename not in filename_lookup:
                filename_lookup[filename] = (test_file.instrumented_behavior_file_path, test_file.test_type)
                logger.debug(f"Jest XML filename lookup: registered {filename}")
        # Also add benchmarking_file_path (perf-only tests) - these have different filenames
        # e.g., utils__perfonlyinstrumented.test.ts vs utils__perfinstrumented.test.ts
        if test_file.benchmarking_file_path:
            bench_filename = test_file.benchmarking_file_path.name
            if bench_filename not in filename_lookup:
                filename_lookup[bench_filename] = (test_file.benchmarking_file_path, test_file.test_type)
                logger.debug(f"Jest XML filename lookup: registered benchmark file {bench_filename}")

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
                # Check for END markers with duration (perf test markers)
                end_marker_count = len(jest_end_pattern.findall(global_stdout))
                if end_marker_count > 0:
                    logger.debug(f"[PERF-DEBUG] Found {end_marker_count} END timing markers with duration in Jest stdout")
                    # Sample a few markers to verify loop indices
                    end_samples = list(jest_end_pattern.finditer(global_stdout))[:5]
                    for sample in end_samples:
                        groups = sample.groups()
                        logger.debug(f"[PERF-DEBUG] Sample END marker: loopIndex={groups[3]}, duration={groups[5]}")
                else:
                    logger.debug(f"[PERF-DEBUG] No END markers with duration found in Jest stdout")
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

        # Debug: log suite-level END marker parsing for perf tests
        if end_matches_dict:
            # Get unique loop indices from the parsed END markers
            loop_indices = sorted(set(int(k[3]) if k[3].isdigit() else 1 for k in end_matches_dict.keys()))
            logger.debug(f"[PERF-DEBUG] Suite {suite_count}: parsed {len(end_matches_dict)} END markers from suite_stdout, loop_index range: {min(loop_indices)}-{max(loop_indices)}")

        # Also collect timing markers from testcase-level system-out (Vitest puts output at testcase level)
        for tc in suite:
            tc_system_out = tc._elem.find("system-out")  # noqa: SLF001
            if tc_system_out is not None and tc_system_out.text:
                tc_stdout = tc_system_out.text.strip()
                logger.debug(f"Vitest testcase system-out found: {len(tc_stdout)} chars, first 200: {tc_stdout[:200]}")
                end_marker_count = 0
                for match in jest_end_pattern.finditer(tc_stdout):
                    key = match.groups()[:5]
                    end_matches_dict[key] = match
                    end_marker_count += 1
                if end_marker_count > 0:
                    logger.debug(f"Found {end_marker_count} END timing markers in testcase system-out")
                start_matches.extend(jest_start_pattern.finditer(tc_stdout))

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
            if test_file_path is None and resolve_test_file_from_class_path is not None:
                test_file_path = resolve_test_file_from_class_path(test_class_path, base_dir)
                if test_file_path is None:
                    test_file_name = suite._elem.attrib.get("file") or testcase._elem.attrib.get("file")  # noqa: SLF001
                    if test_file_name:
                        test_file_path = base_dir.parent / test_file_name
                        if not test_file_path.exists():
                            test_file_path = base_dir / test_file_name

            # Fallback: try matching by filename only
            # This handles when JUnit XML has relative paths like "test/utils__perfinstrumented.test.ts"
            # that can't be resolved to absolute paths because they're relative to Jest's CWD, not parse CWD
            if test_file_path is None and test_class_path:
                # Extract filename from the path (handles both forward and back slashes)
                path_filename = Path(test_class_path).name
                if path_filename in filename_lookup:
                    test_file_path, test_type = filename_lookup[path_filename]
                    logger.debug(f"Jest XML: matched by filename {path_filename}")

            # Also try filename matching on the file attribute if classname matching failed
            if test_file_path is None:
                test_file_name = suite._elem.attrib.get("file") or testcase._elem.attrib.get("file")  # noqa: SLF001
                if test_file_name:
                    file_attr_filename = Path(test_file_name).name
                    if file_attr_filename in filename_lookup:
                        test_file_path, test_type = filename_lookup[file_attr_filename]
                        logger.debug(f"Jest XML: matched by file attr filename {file_attr_filename}")

            # For Jest tests in monorepos, test files may not exist after cleanup
            # but we can still parse results and infer test type from the path
            if test_file_path is None:
                logger.warning(f"Could not resolve test file for Jest test: {test_class_path}")
                continue

            # Get test type if not already set from lookup
            if test_type is None and test_file_path.exists():
                test_type = test_files.get_test_type_by_instrumented_file_path(test_file_path)
            if test_type is None:
                # Infer test type from filename pattern
                filename = test_file_path.name
                if "__perf_test_" in filename or "_perf_test_" in filename:
                    test_type = TestType.GENERATED_PERFORMANCE
                elif "__unit_test_" in filename or "_unit_test_" in filename:
                    test_type = TestType.GENERATED_REGRESSION
                else:
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

            # Debug: log which branch we're taking
            logger.debug(
                f"[FLOW-DEBUG] Testcase '{test_name[:50]}': "
                f"total_start_matches={len(start_matches)}, matching_starts={len(matching_starts)}, "
                f"total_end_matches={len(end_matches_dict)}"
            )

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
                # Debug: log matching results for perf tests
                if matching_ends_direct:
                    loop_indices = [int(m.groups()[3]) if m.groups()[3].isdigit() else 1 for m in matching_ends_direct]
                    logger.debug(
                        f"[PERF-MATCH] Testcase '{test_name[:40]}': matched {len(matching_ends_direct)} END markers, "
                        f"loop_index range: {min(loop_indices)}-{max(loop_indices)}"
                    )
                elif end_matches_dict:
                    # No matches but we have END markers - check why
                    sample_keys = list(end_matches_dict.keys())[:3]
                    logger.debug(
                        f"[PERF-MISMATCH] Testcase '{test_name[:40]}': no matches found. "
                        f"sanitized_test_name='{sanitized_test_name[:50]}', "
                        f"sample end_keys={[k[1][:30] if len(k) >= 2 else k for k in sample_keys]}"
                    )

            # Log if we're skipping the matching_ends_direct branch
            if matching_starts and end_matches_dict:
                logger.debug(
                    f"[FLOW-SKIP] Testcase '{test_name[:40]}': has {len(matching_starts)} START markers, "
                    f"skipping {len(end_matches_dict)} END markers (behavior test mode)"
                )

            if not matching_starts and not matching_ends_direct:
                # No timing markers found - use JUnit XML time attribute as fallback
                # The time attribute is in seconds (e.g., "0.00077875"), convert to nanoseconds
                runtime = None
                try:
                    time_attr = testcase._elem.attrib.get("time")  # noqa: SLF001
                    if time_attr:
                        time_seconds = float(time_attr)
                        runtime = int(time_seconds * 1_000_000_000)  # Convert seconds to nanoseconds
                        logger.debug(f"Jest XML: using time attribute for {test_name}: {time_seconds}s = {runtime}ns")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Jest XML: could not parse time attribute: {e}")

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
                        runtime=runtime,
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
                loop_indices_found = []
                for end_match in matching_ends_direct:
                    groups = end_match.groups()
                    # groups: (module, testName, funcName, loopIndex, invocationId, durationNs)
                    func_name = groups[2]
                    loop_index = int(groups[3]) if groups[3].isdigit() else 1
                    loop_indices_found.append(loop_index)
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
                if loop_indices_found:
                    logger.debug(
                        f"[LOOP-DEBUG] Testcase '{test_name}': processed {len(matching_ends_direct)} END markers, "
                        f"loop_index range: {min(loop_indices_found)}-{max(loop_indices_found)}, "
                        f"total results so far: {len(test_results.test_results)}"
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
        # Debug: show loop_index distribution for perf analysis
        if test_results.test_results:
            loop_indices = [r.loop_index for r in test_results.test_results]
            unique_loop_indices = sorted(set(loop_indices))
            min_idx, max_idx = min(unique_loop_indices), max(unique_loop_indices)
            logger.debug(
                f"[LOOP-SUMMARY] Results loop_index: min={min_idx}, max={max_idx}, "
                f"unique_count={len(unique_loop_indices)}, total_results={len(loop_indices)}"
            )
            if max_idx == 1 and len(loop_indices) > 1:
                logger.warning(
                    f"[LOOP-WARNING] All {len(loop_indices)} results have loop_index=1. "
                    "Perf test markers may not have been parsed correctly."
                )

    return test_results
