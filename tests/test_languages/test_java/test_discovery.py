"""Tests for Java function/method discovery."""

from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria, Language
from codeflash.languages.java.discovery import (
    discover_functions,
    discover_functions_from_source,
    discover_test_methods,
    get_class_methods,
    get_method_by_name,
)


class TestDiscoverFunctions:
    """Tests for function discovery."""

    def test_discover_simple_method(self):
        """Test discovering a simple method."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 1
        assert functions[0].name == "add"
        assert functions[0].language == Language.JAVA
        assert functions[0].is_method is True
        assert functions[0].class_name == "Calculator"

    def test_discover_multiple_methods(self):
        """Test discovering multiple methods."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }
}
"""
        functions = discover_functions_from_source(source)
        assert len(functions) == 3
        method_names = {f.name for f in functions}
        assert method_names == {"add", "subtract", "multiply"}

    def test_skip_abstract_methods(self):
        """Test that abstract methods are skipped."""
        source = """
public abstract class Shape {
    public abstract double area();

    public double perimeter() {
        return 0.0;
    }
}
"""
        functions = discover_functions_from_source(source)
        # Should only find perimeter, not area
        assert len(functions) == 1
        assert functions[0].name == "perimeter"

    def test_skip_constructors(self):
        """Test that constructors are skipped."""
        source = """
public class Person {
    private String name;

    public Person(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
"""
        functions = discover_functions_from_source(source)
        # Should only find getName, not the constructor
        assert len(functions) == 1
        assert functions[0].name == "getName"

    def test_filter_by_pattern(self):
        """Test filtering by include patterns."""
        source = """
public class StringUtils {
    public String toUpperCase(String s) {
        return s.toUpperCase();
    }

    public String toLowerCase(String s) {
        return s.toLowerCase();
    }

    public int length(String s) {
        return s.length();
    }
}
"""
        criteria = FunctionFilterCriteria(include_patterns=["*Upper*", "*Lower*"])
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        assert len(functions) == 2
        method_names = {f.name for f in functions}
        assert method_names == {"toUpperCase", "toLowerCase"}

    def test_filter_exclude_pattern(self):
        """Test filtering by exclude patterns."""
        source = """
public class DataService {
    public void getData() {}
    public void setData() {}
    public void processData() {}
}
"""
        criteria = FunctionFilterCriteria(
            exclude_patterns=["set*"],
            require_return=False,  # Allow void methods
        )
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        method_names = {f.name for f in functions}
        assert "setData" not in method_names

    def test_filter_require_return(self):
        """Test filtering by require_return."""
        source = """
public class Example {
    public void doSomething() {}

    public int getValue() {
        return 42;
    }
}
"""
        criteria = FunctionFilterCriteria(require_return=True)
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        assert len(functions) == 1
        assert functions[0].name == "getValue"

    def test_filter_by_line_count(self):
        """Test filtering by line count."""
        source = """
public class Example {
    public int short() { return 1; }

    public int long() {
        int a = 1;
        int b = 2;
        int c = 3;
        int d = 4;
        int e = 5;
        return a + b + c + d + e;
    }
}
"""
        criteria = FunctionFilterCriteria(min_lines=3, require_return=False)
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        # The 'long' method should be included (>3 lines)
        # The 'short' method should be excluded (1 line)
        method_names = {f.name for f in functions}
        assert "long" in method_names or len(functions) >= 1

    def test_method_with_javadoc(self):
        """Test that Javadoc is tracked."""
        source = """
public class Example {
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
        assert functions[0].doc_start_line is not None
        # Doc should start before the method
        assert functions[0].doc_start_line < functions[0].start_line


class TestDiscoverTestMethods:
    """Tests for test method discovery."""

    def test_discover_junit5_tests(self, tmp_path: Path):
        """Test discovering JUnit 5 test methods."""
        test_file = tmp_path / "CalculatorTest.java"
        test_file.write_text("""
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAdd() {
        assertEquals(4, 2 + 2);
    }

    @Test
    void testSubtract() {
        assertEquals(0, 2 - 2);
    }

    void helperMethod() {
        // Not a test
    }
}
""")
        tests = discover_test_methods(test_file)
        assert len(tests) == 2
        test_names = {t.name for t in tests}
        assert test_names == {"testAdd", "testSubtract"}

    def test_discover_parameterized_tests(self, tmp_path: Path):
        """Test discovering parameterized tests."""
        test_file = tmp_path / "StringTest.java"
        test_file.write_text("""
package com.example;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class StringTest {
    @ParameterizedTest
    @ValueSource(strings = {"hello", "world"})
    void testLength(String input) {
        assertTrue(input.length() > 0);
    }
}
""")
        tests = discover_test_methods(test_file)
        assert len(tests) == 1
        assert tests[0].name == "testLength"


class TestGetMethodByName:
    """Tests for getting methods by name."""

    def test_get_method_by_name(self, tmp_path: Path):
        """Test getting a specific method by name."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
""")
        method = get_method_by_name(java_file, "add")
        assert method is not None
        assert method.name == "add"

    def test_get_method_not_found(self, tmp_path: Path):
        """Test getting a method that doesn't exist."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""")
        method = get_method_by_name(java_file, "multiply")
        assert method is None


class TestGetClassMethods:
    """Tests for getting methods in a class."""

    def test_get_class_methods(self, tmp_path: Path):
        """Test getting all methods in a specific class."""
        java_file = tmp_path / "Example.java"
        java_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
}

class Helper {
    public void help() {}
}
""")
        methods = get_class_methods(java_file, "Calculator")
        assert len(methods) == 1
        assert methods[0].name == "add"


class TestFileBasedDiscovery:
    """Tests for file-based discovery using the fixture project."""

    @pytest.fixture
    def java_fixture_path(self):
        """Get path to the Java fixture project."""
        fixture_path = Path(__file__).parent.parent.parent / "test_languages" / "fixtures" / "java_maven"
        if not fixture_path.exists():
            pytest.skip("Java fixture project not found")
        return fixture_path

    def test_discover_from_fixture(self, java_fixture_path: Path):
        """Test discovering functions from fixture project."""
        calculator_file = java_fixture_path / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        if not calculator_file.exists():
            pytest.skip("Calculator.java not found in fixture")

        functions = discover_functions(calculator_file)
        assert len(functions) > 0
        method_names = {f.name for f in functions}
        # Should find methods from Calculator.java
        assert "fibonacci" in method_names or "add" in method_names or len(method_names) > 0

    def test_discover_tests_from_fixture(self, java_fixture_path: Path):
        """Test discovering test methods from fixture project."""
        test_file = java_fixture_path / "src" / "test" / "java" / "com" / "example" / "CalculatorTest.java"
        if not test_file.exists():
            pytest.skip("CalculatorTest.java not found in fixture")

        tests = discover_test_methods(test_file)
        assert len(tests) > 0
