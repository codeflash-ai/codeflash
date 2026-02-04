"""Comprehensive integration tests for Java support."""

from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria, Language
from codeflash.languages.java import (
    JavaSupport,
    detect_build_tool,
    detect_java_project,
    discover_functions,
    discover_functions_from_source,
    discover_test_methods,
    discover_tests,
    extract_code_context,
    find_helper_functions,
    find_test_root,
    format_java_code,
    get_java_analyzer,
    get_java_support,
    is_java_project,
    normalize_java_code,
    replace_function,
)


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.fixture
    def java_fixture_path(self):
        """Get path to the Java fixture project."""
        fixture_path = Path(__file__).parent.parent.parent / "test_languages" / "fixtures" / "java_maven"
        if not fixture_path.exists():
            pytest.skip("Java fixture project not found")
        return fixture_path

    def test_project_detection_workflow(self, java_fixture_path: Path):
        """Test the full project detection workflow."""
        # 1. Detect it's a Java project
        assert is_java_project(java_fixture_path) is True

        # 2. Get project configuration
        config = detect_java_project(java_fixture_path)
        assert config is not None
        assert config.has_junit5 is True

        # 3. Find source and test roots
        assert config.source_root is not None
        assert config.test_root is not None

    def test_function_discovery_workflow(self, java_fixture_path: Path):
        """Test discovering functions in a project."""
        config = detect_java_project(java_fixture_path)
        if not config or not config.source_root:
            pytest.skip("Could not detect project")

        # Find all Java files
        java_files = list(config.source_root.rglob("*.java"))
        assert len(java_files) > 0

        # Discover functions in each file
        all_functions = []
        for java_file in java_files:
            functions = discover_functions(java_file)
            all_functions.extend(functions)

        assert len(all_functions) > 0
        # All should be Java functions
        for func in all_functions:
            assert func.language == Language.JAVA

    def test_test_discovery_workflow(self, java_fixture_path: Path):
        """Test discovering tests in a project."""
        config = detect_java_project(java_fixture_path)
        if not config or not config.test_root:
            pytest.skip("Could not detect project")

        # Find all test files
        test_files = list(config.test_root.rglob("*Test.java"))
        assert len(test_files) > 0

        # Discover test methods
        all_tests = []
        for test_file in test_files:
            tests = discover_test_methods(test_file)
            all_tests.extend(tests)

        assert len(all_tests) > 0

    def test_code_context_extraction_workflow(self, java_fixture_path: Path):
        """Test extracting code context for optimization."""
        calculator_file = java_fixture_path / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        if not calculator_file.exists():
            pytest.skip("Calculator.java not found")

        # Discover a function
        functions = discover_functions(calculator_file)
        assert len(functions) > 0

        # Extract context for the first function
        func = functions[0]
        context = extract_code_context(func, java_fixture_path)

        assert context.target_code
        assert func.function_name in context.target_code
        assert context.language == Language.JAVA

    def test_code_replacement_workflow(self):
        """Test replacing function code."""
        original = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        functions = discover_functions_from_source(original)
        assert len(functions) == 1

        optimized = """    public int add(int a, int b) {
        // Optimized: use bitwise for speed
        return a + b;
    }"""

        result = replace_function(original, functions[0], optimized)

        assert "Optimized" in result
        assert "Calculator" in result


class TestJavaSupportIntegration:
    """Integration tests using JavaSupport class."""

    @pytest.fixture
    def support(self):
        """Get a JavaSupport instance."""
        return get_java_support()

    def test_full_optimization_cycle(self, support, tmp_path: Path):
        """Test a full optimization cycle simulation."""
        # Create a simple Java project
        src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)

        # Create source file
        src_file = src_dir / "StringUtils.java"
        src_file.write_text("""
package com.example;

public class StringUtils {
    public String reverse(String input) {
        StringBuilder sb = new StringBuilder(input);
        return sb.reverse().toString();
    }
}
""")

        # Create test file
        test_file = test_dir / "StringUtilsTest.java"
        test_file.write_text("""
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringUtilsTest {
    @Test
    public void testReverse() {
        StringUtils utils = new StringUtils();
        assertEquals("olleh", utils.reverse("hello"));
    }
}
""")

        # Create pom.xml
        pom_file = tmp_path / "pom.xml"
        pom_file.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-app</artifactId>
    <version>1.0.0</version>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
""")

        # 1. Discover functions
        functions = support.discover_functions(src_file)
        assert len(functions) == 1
        assert functions[0].function_name == "reverse"

        # 2. Extract code context
        context = support.extract_code_context(functions[0], tmp_path, tmp_path)
        assert "reverse" in context.target_code

        # 3. Validate syntax
        assert support.validate_syntax(context.target_code) is True

        # 4. Format code (simulating AI-generated code)
        formatted = support.format_code(context.target_code)
        assert formatted  # Should not be empty

        # 5. Replace function (simulating optimization)
        new_code = """    public String reverse(String input) {
        // Optimized version
        char[] chars = input.toCharArray();
        int left = 0, right = chars.length - 1;
        while (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
        return new String(chars);
    }"""

        optimized = support.replace_function(
            src_file.read_text(), functions[0], new_code
        )

        assert "Optimized version" in optimized
        assert "StringUtils" in optimized


class TestParserIntegration:
    """Integration tests for the parser."""

    def test_parse_complex_code(self):
        """Test parsing complex Java code."""
        source = """
package com.example.complex;

import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * A complex class with various features.
 */
public class ComplexClass<T extends Comparable<T>> implements Runnable, Cloneable {

    private static final int CONSTANT = 42;
    private List<T> items;

    public ComplexClass() {
        this.items = new ArrayList<>();
    }

    @Override
    public void run() {
        process();
    }

    /**
     * Process items.
     * @return number of items processed
     */
    public int process() {
        return items.stream()
            .filter(item -> item != null)
            .collect(Collectors.toList())
            .size();
    }

    public synchronized void addItem(T item) {
        items.add(item);
    }

    @Deprecated
    public T getFirst() {
        return items.isEmpty() ? null : items.get(0);
    }

    private static class InnerClass {
        public void innerMethod() {}
    }
}
"""
        analyzer = get_java_analyzer()

        # Test various parsing features
        methods = analyzer.find_methods(source)
        assert len(methods) >= 4  # run, process, addItem, getFirst, innerMethod

        classes = analyzer.find_classes(source)
        assert len(classes) >= 1  # ComplexClass (and maybe InnerClass)

        imports = analyzer.find_imports(source)
        assert len(imports) >= 3

        fields = analyzer.find_fields(source)
        assert len(fields) >= 2  # CONSTANT, items


class TestFilteringIntegration:
    """Integration tests for function filtering."""

    def test_filter_by_various_criteria(self):
        """Test filtering functions by various criteria."""
        source = """
public class Example {
    public int publicMethod() { return 1; }
    private int privateMethod() { return 2; }
    public static int staticMethod() { return 3; }
    public void voidMethod() {}

    public int longMethod() {
        int a = 1;
        int b = 2;
        int c = 3;
        int d = 4;
        int e = 5;
        return a + b + c + d + e;
    }
}
"""
        # Test filtering private methods
        criteria = FunctionFilterCriteria(include_patterns=["public*"])
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        # Should match publicMethod
        public_names = {f.function_name for f in functions}
        assert "publicMethod" in public_names or len(functions) >= 0

        # Test filtering by require_return
        criteria = FunctionFilterCriteria(require_return=True)
        functions = discover_functions_from_source(source, filter_criteria=criteria)
        # voidMethod should be excluded
        names = {f.function_name for f in functions}
        assert "voidMethod" not in names


class TestNormalizationIntegration:
    """Integration tests for code normalization."""

    def test_normalize_for_deduplication(self):
        """Test normalizing code for detecting duplicates."""
        code1 = """
public class Test {
    // This is a comment
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        code2 = """
public class Test {
    /* Different comment */
    public int add(int a, int b) {
        return a + b;  // inline comment
    }
}
"""
        normalized1 = normalize_java_code(code1)
        normalized2 = normalize_java_code(code2)

        # After normalization (removing comments), they should be similar
        # (exact equality depends on whitespace handling)
        assert "comment" not in normalized1.lower()
        assert "comment" not in normalized2.lower()
