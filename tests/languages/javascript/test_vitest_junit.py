"""Tests for Vitest JUnit XML output parsing and compatibility.

These tests verify that Vitest's JUnit XML output can be parsed
by the existing parsing infrastructure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from junitparser import JUnitXml

from codeflash.verification.parse_test_output import jest_end_pattern, jest_start_pattern


class TestVitestJunitXmlFormat:
    """Tests for Vitest JUnit XML format compatibility."""

    def test_can_parse_vitest_junit_xml(self) -> None:
        """Should be able to parse Vitest JUnit XML with junitparser."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="4" failures="1" errors="0" time="0.537">
    <testsuite name="tests/fibonacci.test.ts" timestamp="2026-01-30T18:03:49.433Z" hostname="localhost" tests="3" failures="0" errors="0" skipped="0" time="0.008">
        <testcase classname="tests/fibonacci.test.ts" name="fibonacci &gt; returns 0 for n=0" time="0.001">
        </testcase>
        <testcase classname="tests/fibonacci.test.ts" name="fibonacci &gt; returns 1 for n=1" time="0.0005">
        </testcase>
        <testcase classname="tests/fibonacci.test.ts" name="fibonacci &gt; returns 55 for n=10" time="0.0001">
        </testcase>
    </testsuite>
    <testsuite name="tests/string_utils.test.ts" timestamp="2026-01-30T18:03:49.438Z" hostname="localhost" tests="1" failures="1" errors="0" skipped="0" time="0.01">
        <testcase classname="tests/string_utils.test.ts" name="reverseString &gt; reverses a simple string" time="0.0007">
            <failure message="expected &apos;olleh&apos; to equal &apos;hello&apos;" type="AssertionError">AssertionError: expected 'olleh' to equal 'hello'</failure>
        </testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            assert xml is not None
            test_count = sum(len(list(suite)) for suite in xml)
            assert test_count == 4

    def test_extracts_test_suite_names(self) -> None:
        """Should extract test suite names from Vitest JUnit XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="2" failures="0" errors="0" time="0.1">
    <testsuite name="tests/fibonacci.test.ts" tests="1" failures="0" time="0.01">
        <testcase classname="tests/fibonacci.test.ts" name="test1" time="0.001"></testcase>
    </testsuite>
    <testsuite name="tests/string_utils.test.ts" tests="1" failures="0" time="0.01">
        <testcase classname="tests/string_utils.test.ts" name="test2" time="0.001"></testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            suite_names = [suite.name for suite in xml]
            assert suite_names == ["tests/fibonacci.test.ts", "tests/string_utils.test.ts"]

    def test_extracts_test_case_names_with_vitest_separator(self) -> None:
        """Should extract test case names from Vitest JUnit XML (uses > as separator)."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="2" failures="0" errors="0" time="0.1">
    <testsuite name="tests/fibonacci.test.ts" tests="2" failures="0" time="0.01">
        <testcase classname="tests/fibonacci.test.ts" name="fibonacci &gt; returns 0 for n=0" time="0.001"></testcase>
        <testcase classname="tests/fibonacci.test.ts" name="fibonacci &gt; returns 1 for n=1" time="0.001"></testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            test_names = []
            for suite in xml:
                for case in suite:
                    test_names.append(case.name)

            assert test_names == ["fibonacci > returns 0 for n=0", "fibonacci > returns 1 for n=1"]

    def test_extracts_classname_as_file_path(self) -> None:
        """Should extract classname which contains file path in Vitest."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="1" failures="0" errors="0" time="0.1">
    <testsuite name="tests/fibonacci.test.ts" tests="1" failures="0" time="0.01">
        <testcase classname="tests/fibonacci.test.ts" name="test1" time="0.001"></testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            for suite in xml:
                for case in suite:
                    assert case.classname == "tests/fibonacci.test.ts"

    def test_extracts_test_time_as_float(self) -> None:
        """Should extract test execution time as float from Vitest JUnit XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="1" failures="0" errors="0" time="0.1">
    <testsuite name="tests/test.ts" tests="1" failures="0" time="0.01">
        <testcase classname="tests/test.ts" name="test1" time="0.0015"></testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            for suite in xml:
                for case in suite:
                    assert isinstance(case.time, float)
                    assert case.time == 0.0015

    def test_detects_failures(self) -> None:
        """Should detect test failures in Vitest JUnit XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="2" failures="1" errors="0" time="0.1">
    <testsuite name="tests/test.ts" tests="2" failures="1" time="0.01">
        <testcase classname="tests/test.ts" name="passing test" time="0.001"></testcase>
        <testcase classname="tests/test.ts" name="failing test" time="0.001">
            <failure message="expected true to be false" type="AssertionError">AssertionError: expected true to be false</failure>
        </testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            failures = []
            for suite in xml:
                for case in suite:
                    if not case.is_passed:
                        failures.append(case.name)

            assert failures == ["failing test"]

    def test_extracts_failure_message(self) -> None:
        """Should extract failure message from Vitest JUnit XML."""
        xml_content = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="1" failures="1" errors="0" time="0.1">
    <testsuite name="tests/test.ts" tests="1" failures="1" time="0.01">
        <testcase classname="tests/test.ts" name="failing test" time="0.001">
            <failure message="expected 'actual' to equal 'expected'" type="AssertionError">AssertionError: expected 'actual' to equal 'expected'</failure>
        </testcase>
    </testsuite>
</testsuites>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            f.flush()
            junit_file = Path(f.name)

            xml = JUnitXml.fromfile(str(junit_file))

            for suite in xml:
                for case in suite:
                    if not case.is_passed:
                        for result in case.result:
                            if hasattr(result, "message"):
                                assert result.message == "expected 'actual' to equal 'expected'"


class TestVitestTimingMarkers:
    """Tests for Vitest timing marker extraction.

    Timing markers are used to measure function execution time during benchmarking.
    The format is the same for Jest and Vitest since they use the same codeflash helper.
    """

    def test_parses_start_timing_marker(self) -> None:
        """Should parse start timing marker from Vitest output."""
        output = "!$######fibonacci.test.ts:returns 0 for n=0:fibonacci:1:line_0######$!"

        matches = jest_start_pattern.findall(output)

        assert len(matches) == 1
        test_file, test_name, func_name, loop_index, line_id = matches[0]
        assert test_file == "fibonacci.test.ts"
        assert test_name == "returns 0 for n=0"
        assert func_name == "fibonacci"
        assert loop_index == "1"
        assert line_id == "line_0"

    def test_parses_end_timing_marker(self) -> None:
        """Should parse end timing marker from Vitest output."""
        output = "!######fibonacci.test.ts:returns 0 for n=0:fibonacci:1:line_0:123456######!"

        matches = jest_end_pattern.findall(output)

        assert len(matches) == 1
        test_file, test_name, func_name, loop_index, line_id, duration = matches[0]
        assert test_file == "fibonacci.test.ts"
        assert test_name == "returns 0 for n=0"
        assert func_name == "fibonacci"
        assert loop_index == "1"
        assert line_id == "line_0"
        assert duration == "123456"

    def test_extracts_multiple_timing_markers(self) -> None:
        """Should extract multiple timing markers from Vitest output."""
        output = """Running tests...
!$######test.ts:test1:func:1:id1######$!
executing...
!######test.ts:test1:func:1:id1:100000######!
!$######test.ts:test2:func:1:id2######$!
executing...
!######test.ts:test2:func:1:id2:200000######!
Done."""

        start_matches = jest_start_pattern.findall(output)
        end_matches = jest_end_pattern.findall(output)

        assert len(start_matches) == 2
        assert len(end_matches) == 2

        durations = [int(m[5]) for m in end_matches]
        assert durations == [100000, 200000]

    def test_timing_marker_with_special_characters_in_test_name(self) -> None:
        """Should handle test names with special characters."""
        output = "!$######test.ts:handles_n=0_correctly:fibonacci:1:id######$!"

        matches = jest_start_pattern.findall(output)

        assert len(matches) == 1
        assert matches[0][1] == "handles_n=0_correctly"
