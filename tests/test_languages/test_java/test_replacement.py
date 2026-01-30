"""Tests for Java code replacement."""

from pathlib import Path

import pytest

from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.replacement import (
    add_runtime_comments,
    insert_method,
    remove_method,
    remove_test_functions,
    replace_function,
    replace_method_body,
)


class TestReplaceFunction:
    """Tests for replace_function."""

    def test_replace_simple_method(self):
        """Test replacing a simple method."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 1

        new_method = """    public int add(int a, int b) {
        // Optimized version
        return a + b;
    }"""

        result = replace_function(source, functions[0], new_method)

        assert "Optimized version" in result
        assert "Calculator" in result

    def test_replace_preserves_other_methods(self):
        """Test that other methods are preserved."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
        functions = discover_functions_from_source(source)
        add_func = next(f for f in functions if f.name == "add")

        new_method = """    public int add(int a, int b) {
        return a + b; // optimized
    }"""

        result = replace_function(source, add_func, new_method)

        assert "subtract" in result
        assert "optimized" in result


class TestReplaceMethodBody:
    """Tests for replace_method_body."""

    def test_replace_body(self):
        """Test replacing method body."""
        source = """
public class Example {
    public int getValue() {
        return 42;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 1

        result = replace_method_body(source, functions[0], "return 100;")

        assert "100" in result
        assert "getValue" in result


class TestInsertMethod:
    """Tests for insert_method."""

    def test_insert_at_end(self):
        """Test inserting method at end of class."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        new_method = """public int multiply(int a, int b) {
    return a * b;
}"""

        result = insert_method(source, "Calculator", new_method, position="end")

        assert "multiply" in result
        assert "add" in result


class TestRemoveMethod:
    """Tests for remove_method."""

    def test_remove_method(self):
        """Test removing a method."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
        functions = discover_functions_from_source(source)
        add_func = next(f for f in functions if f.name == "add")

        result = remove_method(source, add_func)

        assert "add" not in result or result.count("add") < source.count("add")
        assert "subtract" in result


class TestRemoveTestFunctions:
    """Tests for remove_test_functions."""

    def test_remove_test_functions(self):
        """Test removing specific test functions."""
        source = """
public class CalculatorTest {
    @Test
    public void testAdd() {
        assertEquals(4, calc.add(2, 2));
    }

    @Test
    public void testSubtract() {
        assertEquals(0, calc.subtract(2, 2));
    }
}
"""
        result = remove_test_functions(source, ["testAdd"])

        # testAdd should be removed, testSubtract should remain
        assert "testSubtract" in result


class TestAddRuntimeComments:
    """Tests for add_runtime_comments."""

    def test_add_comments(self):
        """Test adding runtime comments."""
        source = """
import org.junit.jupiter.api.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assertEquals(4, calc.add(2, 2));
    }
}
"""
        original_runtimes = {"inv1": 1000000}  # 1ms
        optimized_runtimes = {"inv1": 500000}  # 0.5ms

        result = add_runtime_comments(source, original_runtimes, optimized_runtimes)

        # Should contain performance comment
        assert "Performance" in result or "ms" in result
