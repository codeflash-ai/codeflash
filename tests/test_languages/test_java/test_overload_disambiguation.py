"""Tests for method overload disambiguation in test discovery."""

import logging
from pathlib import Path

import pytest

from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.test_discovery import (
    disambiguate_overloads,
    discover_tests,
)


class TestOverloadDisambiguation:
    """Tests for method overload disambiguation in test discovery."""

    def test_overload_disambiguation_by_type_name(self, tmp_path: Path):
        """Overloaded methods in the same class share qualified_name."""
        src_file = tmp_path / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public String add(String a, String b) { return a + b; }
}
""")

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;
public class CalculatorTest {
    @Test
    public void testAddIntegers() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        add_funcs = [f for f in source_functions if f.function_name == "add"]
        assert len(add_funcs) == 2, "Should find both add overloads"
        assert all(f.qualified_name == "Calculator.add" for f in add_funcs)

        result = discover_tests(test_dir, source_functions)
        assert "Calculator.add" in result

    def test_overload_ambiguous_keeps_all_matches(self, tmp_path: Path):
        """Generic test name still matches overloaded functions."""
        src_file = tmp_path / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public String add(String a, String b) { return a + b; }
}
""")

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;
public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        result = discover_tests(test_dir, source_functions)

        assert "Calculator.add" in result
        assert len(result["Calculator.add"]) == 1

    def test_no_overload_single_match(self, tmp_path: Path):
        """Single function add(int, int), test testAdd. Only one match."""
        src_file = tmp_path / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
""")

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;
public class CalculatorTest {
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(2, 2));
    }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        result = discover_tests(test_dir, source_functions)
        assert "Calculator.add" in result
        assert len(result["Calculator.add"]) == 1

    def test_overload_disambiguation_logs_info_on_ambiguity(self, caplog):
        """When overloaded methods are detected, info log fires."""
        matched_names = ["Calculator.add", "StringUtils.add"]
        with caplog.at_level(logging.INFO):
            result = disambiguate_overloads(
                matched_names, "testAdd", "some test source code"
            )

        assert result == matched_names
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Ambiguous overload" in msg for msg in info_messages), (
            f"Expected info log about ambiguous overload match, got: {info_messages}"
        )

    def test_disambiguate_overloads_single_match_returns_unchanged(self):
        """Single match goes through disambiguation unchanged."""
        result = disambiguate_overloads(["Calculator.add"], "testAdd", "source code")
        assert result == ["Calculator.add"]
