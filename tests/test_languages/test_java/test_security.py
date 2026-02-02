"""Tests for Java security and input validation."""

from pathlib import Path

import pytest

from codeflash.languages.java.test_runner import (
    _validate_java_class_name,
    _validate_test_filter,
    get_test_run_command,
)


class TestInputValidation:
    """Tests for input validation to prevent command injection."""

    def test_validate_java_class_name_valid(self):
        """Test validation of valid Java class names."""
        valid_names = [
            "MyTest",
            "com.example.MyTest",
            "com.example.sub.MyTest",
            "MyTest$InnerClass",
            "_MyTest",
            "$MyTest",
            "Test123",
            "com.example.Test_123",
        ]

        for name in valid_names:
            assert _validate_java_class_name(name), f"Should accept: {name}"

    def test_validate_java_class_name_invalid(self):
        """Test rejection of invalid Java class names."""
        invalid_names = [
            "My Test",  # Space
            "My-Test",  # Hyphen
            "My;Test",  # Semicolon (command injection)
            "My&Test",  # Ampersand (command injection)
            "My|Test",  # Pipe (command injection)
            "My`Test",  # Backtick (command injection)
            "My$(whoami)Test",  # Command substitution
            "../../../etc/passwd",  # Path traversal
            "Test\nmalicious",  # Newline
            "",  # Empty
        ]

        for name in invalid_names:
            assert not _validate_java_class_name(name), f"Should reject: {name}"

    def test_validate_test_filter_single_class(self):
        """Test validation of single test class filter."""
        valid_filter = "com.example.MyTest"
        result = _validate_test_filter(valid_filter)
        assert result == valid_filter

    def test_validate_test_filter_multiple_classes(self):
        """Test validation of multiple test classes."""
        valid_filter = "MyTest,OtherTest,com.example.ThirdTest"
        result = _validate_test_filter(valid_filter)
        assert result == valid_filter

    def test_validate_test_filter_wildcards(self):
        """Test validation of wildcard patterns."""
        valid_patterns = [
            "My*Test",
            "*Test",
            "com.example.*Test",
            "com.example.**",
        ]

        for pattern in valid_patterns:
            result = _validate_test_filter(pattern)
            assert result == pattern, f"Should accept wildcard: {pattern}"

    def test_validate_test_filter_rejects_invalid(self):
        """Test rejection of malicious test filters."""
        malicious_filters = [
            "Test;rm -rf /",
            "Test&&whoami",
            "Test|cat /etc/passwd",
            "Test`whoami`",
            "Test$(whoami)",
            "../../../etc/passwd",
        ]

        for malicious in malicious_filters:
            with pytest.raises(ValueError, match="Invalid test class name"):
                _validate_test_filter(malicious)

    def test_get_test_run_command_validates_input(self, tmp_path: Path):
        """Test that get_test_run_command validates test class names."""
        # Valid class names should work
        cmd = get_test_run_command(tmp_path, ["MyTest", "OtherTest"])
        assert "-Dtest=MyTest,OtherTest" in " ".join(cmd)

        # Invalid class names should raise ValueError
        with pytest.raises(ValueError, match="Invalid test class name"):
            get_test_run_command(tmp_path, ["My;Test"])

        with pytest.raises(ValueError, match="Invalid test class name"):
            get_test_run_command(tmp_path, ["Test$(whoami)"])

    def test_special_characters_in_valid_java_names(self):
        """Test that valid Java special characters are allowed."""
        # Dollar sign is valid (inner classes)
        assert _validate_java_class_name("Outer$Inner")

        # Underscore is valid
        assert _validate_java_class_name("_Private")

        # Numbers are valid (but not at start)
        assert _validate_java_class_name("Test123")

        # Numbers at start are invalid
        assert not _validate_java_class_name("123Test")


class TestXMLParsingSecurity:
    """Tests for secure XML parsing."""

    def test_parse_malformed_surefire_report(self, tmp_path: Path):
        """Test handling of malformed XML in Surefire reports."""
        from codeflash.languages.java.build_tools import _parse_surefire_reports

        surefire_dir = tmp_path / "surefire-reports"
        surefire_dir.mkdir()

        # Create a malformed XML file
        malformed_xml = surefire_dir / "TEST-Malformed.xml"
        malformed_xml.write_text("<testsuite><testcase>no closing tag")

        # Should not crash, should log warning and return 0
        tests_run, failures, errors, skipped = _parse_surefire_reports(surefire_dir)
        assert tests_run == 0
        assert failures == 0
        assert errors == 0
        assert skipped == 0

    def test_parse_surefire_report_invalid_numbers(self, tmp_path: Path):
        """Test handling of invalid numeric attributes in XML."""
        from codeflash.languages.java.build_tools import _parse_surefire_reports

        surefire_dir = tmp_path / "surefire-reports"
        surefire_dir.mkdir()

        # Create XML with invalid numeric values
        invalid_xml = surefire_dir / "TEST-Invalid.xml"
        invalid_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<testsuite tests="abc" failures="xyz" errors="foo" skipped="bar">
    <testcase name="test1" classname="MyTest" time="0.001"/>
</testsuite>
""")

        # Should handle gracefully and default to 0
        tests_run, failures, errors, skipped = _parse_surefire_reports(surefire_dir)
        assert tests_run == 0  # Invalid "abc" defaulted to 0
        assert failures == 0   # Invalid "xyz" defaulted to 0
        assert errors == 0     # Invalid "foo" defaulted to 0
        assert skipped == 0    # Invalid "bar" defaulted to 0

    def test_parse_valid_surefire_report(self, tmp_path: Path):
        """Test parsing of valid Surefire report."""
        from codeflash.languages.java.build_tools import _parse_surefire_reports

        surefire_dir = tmp_path / "surefire-reports"
        surefire_dir.mkdir()

        # Create valid XML
        valid_xml = surefire_dir / "TEST-Valid.xml"
        valid_xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<testsuite tests="5" failures="1" errors="2" skipped="1" time="1.234">
    <testcase name="test1" classname="MyTest" time="0.001"/>
    <testcase name="test2" classname="MyTest" time="0.002">
        <failure>Expected true but was false</failure>
    </testcase>
    <testcase name="test3" classname="MyTest" time="0.003">
        <error>NullPointerException</error>
    </testcase>
    <testcase name="test4" classname="MyTest" time="0.004">
        <error>IllegalArgumentException</error>
    </testcase>
    <testcase name="test5" classname="MyTest" time="0.005">
        <skipped/>
    </testcase>
</testsuite>
""")

        tests_run, failures, errors, skipped = _parse_surefire_reports(surefire_dir)
        assert tests_run == 5
        assert failures == 1
        assert errors == 2
        assert skipped == 1

    def test_parse_multiple_surefire_reports(self, tmp_path: Path):
        """Test parsing of multiple Surefire reports."""
        from codeflash.languages.java.build_tools import _parse_surefire_reports

        surefire_dir = tmp_path / "surefire-reports"
        surefire_dir.mkdir()

        # Create multiple valid XML files
        for i in range(3):
            xml_file = surefire_dir / f"TEST-Suite{i}.xml"
            xml_file.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite tests="{i+1}" failures="0" errors="0" skipped="0">
    <testcase name="test1" classname="Suite{i}" time="0.001"/>
</testsuite>
""")

        tests_run, failures, errors, skipped = _parse_surefire_reports(surefire_dir)
        assert tests_run == 1 + 2 + 3  # Sum of all tests
        assert failures == 0
        assert errors == 0
        assert skipped == 0


class TestErrorHandling:
    """Tests for robust error handling."""

    def test_empty_test_class_name(self):
        """Test handling of empty test class name."""
        assert not _validate_java_class_name("")

    def test_whitespace_test_class_name(self):
        """Test handling of whitespace-only test class name."""
        assert not _validate_java_class_name("   ")

    def test_test_filter_with_spaces(self):
        """Test handling of test filter with spaces (should be rejected)."""
        with pytest.raises(ValueError):
            _validate_test_filter("My Test")

    def test_test_filter_empty_after_split(self):
        """Test handling of empty patterns after comma split."""
        # Empty patterns between commas should raise ValueError
        with pytest.raises(ValueError, match="Invalid test class name"):
            _validate_test_filter("Test1,,Test2")
