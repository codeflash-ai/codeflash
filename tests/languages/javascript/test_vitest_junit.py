"""Tests for Vitest JUnit XML output parsing and compatibility.

These tests verify that Vitest's JUnit XML output can be parsed
by the existing parsing infrastructure.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestVitestJunitXmlFormat:
    """Tests for Vitest JUnit XML format compatibility."""

    @pytest.fixture
    def vitest_junit_xml(self, tmp_path: Path) -> Path:
        """Create a sample Vitest JUnit XML file."""
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
        junit_file = tmp_path / "vitest-results.xml"
        junit_file.write_text(xml_content)
        return junit_file

    def test_can_parse_vitest_junit_xml(self, vitest_junit_xml: Path) -> None:
        """Should be able to parse Vitest JUnit XML with junitparser."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        assert xml is not None
        # Count test cases
        test_count = sum(len(list(suite)) for suite in xml)
        assert test_count == 4

    def test_extracts_test_suite_names(self, vitest_junit_xml: Path) -> None:
        """Should extract test suite names from Vitest JUnit XML."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        suite_names = [suite.name for suite in xml]
        assert "tests/fibonacci.test.ts" in suite_names
        assert "tests/string_utils.test.ts" in suite_names

    def test_extracts_test_case_names(self, vitest_junit_xml: Path) -> None:
        """Should extract test case names from Vitest JUnit XML."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        test_names = []
        for suite in xml:
            for case in suite:
                test_names.append(case.name)

        # Vitest uses > as separator
        assert "fibonacci > returns 0 for n=0" in test_names
        assert "reverseString > reverses a simple string" in test_names

    def test_extracts_classname_as_file_path(self, vitest_junit_xml: Path) -> None:
        """Should extract classname which contains file path in Vitest."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        classnames = set()
        for suite in xml:
            for case in suite:
                classnames.add(case.classname)

        # Vitest uses file path as classname
        assert "tests/fibonacci.test.ts" in classnames
        assert "tests/string_utils.test.ts" in classnames

    def test_extracts_test_time(self, vitest_junit_xml: Path) -> None:
        """Should extract test execution time from Vitest JUnit XML."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        for suite in xml:
            for case in suite:
                # Time should be a float
                assert isinstance(case.time, float)
                assert case.time >= 0

    def test_detects_failures(self, vitest_junit_xml: Path) -> None:
        """Should detect test failures in Vitest JUnit XML."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        failures = []
        for suite in xml:
            for case in suite:
                if not case.is_passed:
                    failures.append(case.name)

        assert len(failures) == 1
        assert "reverseString > reverses a simple string" in failures

    def test_extracts_failure_message(self, vitest_junit_xml: Path) -> None:
        """Should extract failure message from Vitest JUnit XML."""
        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(vitest_junit_xml))

        for suite in xml:
            for case in suite:
                if not case.is_passed:
                    # Get failure element
                    for result in case.result:
                        if hasattr(result, "message"):
                            assert "expected" in result.message.lower()


class TestVitestJunitXmlResolution:
    """Tests for resolving test file paths from Vitest JUnit XML."""

    def test_resolves_test_file_from_vitest_classname(self, tmp_path: Path) -> None:
        """Should resolve test file path from Vitest classname."""
        from codeflash.verification.parse_test_output import resolve_test_file_from_class_path

        # Create test directory structure
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "fibonacci.test.ts"
        test_file.write_text("// test file")

        # Vitest uses file paths as classname
        classname = "tests/fibonacci.test.ts"

        result = resolve_test_file_from_class_path(classname, tmp_path)

        assert result is not None
        assert result.exists()

    def test_handles_nested_test_paths(self, tmp_path: Path) -> None:
        """Should handle nested test paths from Vitest."""
        from codeflash.verification.parse_test_output import resolve_test_file_from_class_path

        # Create nested test directory structure
        tests_dir = tmp_path / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        test_file = tests_dir / "fibonacci.test.ts"
        test_file.write_text("// test file")

        # Vitest uses file paths as classname
        classname = "tests/unit/fibonacci.test.ts"

        result = resolve_test_file_from_class_path(classname, tmp_path)

        assert result is not None
        assert result.exists()


class TestVitestTimingMarkers:
    """Tests for Vitest timing marker extraction.

    Timing markers are used to measure function execution time during benchmarking.
    The format is the same for Jest and Vitest since they use the same codeflash helper.
    """

    def test_parses_start_timing_marker(self) -> None:
        """Should parse start timing marker from Vitest output."""
        from codeflash.verification.parse_test_output import jest_start_pattern

        # Timing marker format: !$######testName:testName:funcName:loopIndex:lineId######$!
        output = "!$######fibonacci.test.ts:returns 0 for n=0:fibonacci:1:line_0######$!"

        matches = jest_start_pattern.findall(output)

        assert len(matches) == 1
        match = matches[0]
        assert match[0] == "fibonacci.test.ts"  # test file
        assert match[1] == "returns 0 for n=0"  # test name
        assert match[2] == "fibonacci"  # function name
        assert match[3] == "1"  # loop index
        assert match[4] == "line_0"  # line id

    def test_parses_end_timing_marker(self) -> None:
        """Should parse end timing marker from Vitest output."""
        from codeflash.verification.parse_test_output import jest_end_pattern

        # End marker format: !######testName:testName:funcName:loopIndex:lineId:durationNs######!
        output = "!######fibonacci.test.ts:returns 0 for n=0:fibonacci:1:line_0:123456######!"

        matches = jest_end_pattern.findall(output)

        assert len(matches) == 1
        match = matches[0]
        assert match[0] == "fibonacci.test.ts"  # test file
        assert match[1] == "returns 0 for n=0"  # test name
        assert match[2] == "fibonacci"  # function name
        assert match[3] == "1"  # loop index
        assert match[4] == "line_0"  # line id
        assert match[5] == "123456"  # duration in nanoseconds

    def test_extracts_multiple_timing_markers(self) -> None:
        """Should extract multiple timing markers from Vitest output."""
        from codeflash.verification.parse_test_output import jest_end_pattern, jest_start_pattern

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

        # Verify durations
        durations = [int(m[5]) for m in end_matches]
        assert durations == [100000, 200000]


class TestVitestRealJunitOutput:
    """Tests using real Vitest JUnit output from the test project."""

    @pytest.fixture
    def vitest_project_dir(self):
        """Get the Vitest sample project directory."""
        project_root = Path(__file__).parent.parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return vitest_dir

    def test_parses_real_vitest_junit_output(self, vitest_project_dir: Path) -> None:
        """Should parse real Vitest JUnit output from test project."""
        junit_file = vitest_project_dir / ".codeflash" / "vitest-results.xml"
        if not junit_file.exists():
            pytest.skip("Vitest JUnit output not found - run npm test first")

        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(junit_file))

        # Should have parsed without errors
        assert xml is not None

        # Should have multiple test suites
        suite_count = len(list(xml))
        assert suite_count >= 2

        # All tests should pass in the sample project
        for suite in xml:
            for case in suite:
                assert case.is_passed, f"Test {case.name} should pass"

    def test_counts_tests_in_real_output(self, vitest_project_dir: Path) -> None:
        """Should count all tests in real Vitest JUnit output."""
        junit_file = vitest_project_dir / ".codeflash" / "vitest-results.xml"
        if not junit_file.exists():
            pytest.skip("Vitest JUnit output not found - run npm test first")

        from junitparser import JUnitXml

        xml = JUnitXml.fromfile(str(junit_file))

        test_count = sum(len(list(suite)) for suite in xml)

        # We have 22 tests in fibonacci.test.ts and 21 in string_utils.test.ts
        assert test_count >= 40
