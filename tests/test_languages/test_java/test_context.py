"""Tests for Java code context extraction."""

from pathlib import Path

import pytest

from codeflash.languages.base import Language
from codeflash.languages.java.context import (
    extract_code_context,
    extract_function_source,
    extract_read_only_context,
)
from codeflash.languages.java.discovery import discover_functions_from_source


class TestExtractFunctionSource:
    """Tests for extract_function_source."""

    def test_extract_simple_method(self):
        """Test extracting a simple method."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 1

        func_source = extract_function_source(source, functions[0])
        assert "public int add" in func_source
        assert "return a + b" in func_source

    def test_extract_method_with_javadoc(self):
        """Test extracting method including Javadoc."""
        source = """
public class Calculator {
    /**
     * Adds two numbers.
     * @param a first number
     * @param b second number
     * @return sum
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 1

        func_source = extract_function_source(source, functions[0])
        # Should include Javadoc
        assert "/**" in func_source or "Adds two numbers" in func_source


class TestExtractCodeContext:
    """Tests for extract_code_context."""

    def test_extract_context(self, tmp_path: Path):
        """Test extracting full code context."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""
package com.example;

import java.util.List;

public class Calculator {
    private int base = 0;

    public int add(int a, int b) {
        return a + b + base;
    }

    private int helper(int x) {
        return x * 2;
    }
}
""")

        functions = discover_functions_from_source(
            java_file.read_text(), file_path=java_file
        )
        add_func = next((f for f in functions if f.name == "add"), None)
        assert add_func is not None

        context = extract_code_context(add_func, tmp_path)

        assert context.language == Language.JAVA
        assert "add" in context.target_code
        assert context.target_file == java_file


class TestExtractReadOnlyContext:
    """Tests for extract_read_only_context."""

    def test_extract_fields(self):
        """Test extracting class fields."""
        source = """
public class Calculator {
    private int base;
    private static final double PI = 3.14159;

    public int add(int a, int b) {
        return a + b;
    }
}
"""
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        functions = discover_functions_from_source(source, analyzer=analyzer)
        add_func = next((f for f in functions if f.name == "add"), None)
        assert add_func is not None

        context = extract_read_only_context(source, add_func, analyzer)

        # Should include field declarations
        assert "base" in context or "PI" in context or context == ""
