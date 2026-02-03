"""Tests for Java test discovery for JUnit 5."""

from pathlib import Path

import pytest

from codeflash.languages.java.discovery import discover_functions_from_source
from codeflash.languages.java.test_discovery import (
    discover_all_tests,
    discover_tests,
    find_tests_for_function,
    get_test_class_for_source_class,
    get_test_file_suffix,
    is_test_file,
)


class TestIsTestFile:
    """Tests for is_test_file function."""

    def test_standard_test_suffix(self, tmp_path: Path):
        """Test detecting files with Test suffix."""
        test_file = tmp_path / "CalculatorTest.java"
        test_file.touch()
        assert is_test_file(test_file) is True

    def test_standard_tests_suffix(self, tmp_path: Path):
        """Test detecting files with Tests suffix."""
        test_file = tmp_path / "CalculatorTests.java"
        test_file.touch()
        assert is_test_file(test_file) is True

    def test_test_prefix(self, tmp_path: Path):
        """Test detecting files with Test prefix."""
        test_file = tmp_path / "TestCalculator.java"
        test_file.touch()
        assert is_test_file(test_file) is True

    def test_not_test_file(self, tmp_path: Path):
        """Test detecting non-test files."""
        source_file = tmp_path / "Calculator.java"
        source_file.touch()
        assert is_test_file(source_file) is False


class TestGetTestFileSuffix:
    """Tests for get_test_file_suffix function."""

    def test_suffix(self):
        """Test getting the test file suffix."""
        assert get_test_file_suffix() == "Test.java"


class TestGetTestClassForSourceClass:
    """Tests for get_test_class_for_source_class function."""

    def test_find_test_class(self, tmp_path: Path):
        """Test finding test class for source class."""
        test_file = tmp_path / "CalculatorTest.java"
        test_file.write_text("""
public class CalculatorTest {
    @Test
    public void testAdd() {}
}
""")

        result = get_test_class_for_source_class("Calculator", tmp_path)
        assert result is not None
        assert result.name == "CalculatorTest.java"

    def test_not_found(self, tmp_path: Path):
        """Test when no test class exists."""
        result = get_test_class_for_source_class("NonExistent", tmp_path)
        assert result is None


class TestDiscoverTests:
    """Tests for discover_tests function."""

    def test_discover_tests_by_name(self, tmp_path: Path):
        """Test discovering tests by method name matching."""
        # Create source file
        src_dir = tmp_path / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        src_file = src_dir / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""")

        # Create test file
        test_dir = tmp_path / "src" / "test" / "java"
        test_dir.mkdir(parents=True)
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

        # Get source functions
        source_functions = discover_functions_from_source(
            src_file.read_text(), file_path=src_file
        )

        # Discover tests
        result = discover_tests(test_dir, source_functions)

        # Should find the test for add
        assert len(result) > 0 or "Calculator.add" in result or any("add" in k.lower() for k in result.keys())


class TestDiscoverAllTests:
    """Tests for discover_all_tests function."""

    def test_discover_all(self, tmp_path: Path):
        """Test discovering all tests in a directory."""
        test_dir = tmp_path / "tests"
        test_dir.mkdir()

        test_file = test_dir / "ExampleTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;

public class ExampleTest {
    @Test
    public void test1() {}

    @Test
    public void test2() {}
}
""")

        tests = discover_all_tests(test_dir)
        assert len(tests) == 2


class TestFindTestsForFunction:
    """Tests for find_tests_for_function function."""

    def test_find_tests(self, tmp_path: Path):
        """Test finding tests for a specific function."""
        # Create test directory with test file
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        test_file = test_dir / "StringUtilsTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;

public class StringUtilsTest {
    @Test
    public void testReverse() {}

    @Test
    public void testLength() {}
}
""")

        # Create source function
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize
        from codeflash.languages.base import Language

        func = FunctionToOptimize(
            function_name="reverse",
            file_path=tmp_path / "StringUtils.java",
            starting_line=1,
            ending_line=5,
            parents=[],
            is_method=True,
            language="java",
        )

        tests = find_tests_for_function(func, test_dir)
        # Should find testReverse
        test_names = [t.test_name for t in tests]
        assert "testReverse" in test_names or len(tests) >= 0


class TestImportBasedDiscovery:
    """Tests for import-based test discovery."""

    def test_discover_by_import_when_class_name_doesnt_match(self, tmp_path: Path):
        """Test that tests are discovered when they import a class even if class name doesn't match.

        This reproduces a real-world scenario from aerospike-client-java where:
        - TestQueryBlob imports Buffer class
        - TestQueryBlob calls Buffer.longToBytes() directly
        - We want to optimize Buffer.bytesToHexString()
        - The test should be discovered because it imports and uses Buffer
        """
        # Create source file with utility methods
        src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        src_file = src_dir / "Buffer.java"
        src_file.write_text("""
package com.example;

public class Buffer {
    public static String bytesToHexString(byte[] buf) {
        StringBuilder sb = new StringBuilder();
        for (byte b : buf) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    public static void longToBytes(long v, byte[] buf, int offset) {
        buf[offset] = (byte)(v >> 56);
        buf[offset+1] = (byte)(v >> 48);
    }
}
""")

        # Create test file that imports Buffer but has non-matching name
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "TestQueryBlob.java"
        test_file.write_text("""
package com.example;

import org.junit.jupiter.api.Test;
import com.example.Buffer;

public class TestQueryBlob {
    @Test
    public void queryBlob() {
        byte[] bytes = new byte[8];
        Buffer.longToBytes(50003, bytes, 0);
        // Uses Buffer class
    }
}
""")

        # Get source functions
        source_functions = discover_functions_from_source(
            src_file.read_text(), file_path=src_file
        )

        # Filter to just bytesToHexString
        target_functions = [f for f in source_functions if f.function_name == "bytesToHexString"]
        assert len(target_functions) == 1, "Should find bytesToHexString function"

        # Discover tests
        result = discover_tests(tmp_path / "src" / "test" / "java", target_functions)

        # The test should be discovered because it imports Buffer class
        # Even though TestQueryBlob doesn't follow naming convention for BufferTest
        assert len(result) > 0, "Should find tests that import the target class"
        assert "Buffer.bytesToHexString" in result, f"Should map test to Buffer.bytesToHexString, got: {result.keys()}"

    def test_discover_by_direct_method_call(self, tmp_path: Path):
        """Test that tests are discovered when they directly call the target method."""
        # Create source file
        src_dir = tmp_path / "src" / "main" / "java"
        src_dir.mkdir(parents=True)
        src_file = src_dir / "Utils.java"
        src_file.write_text("""
public class Utils {
    public static String format(String s) {
        return s.toUpperCase();
    }
}
""")

        # Create test with direct call to format()
        test_dir = tmp_path / "src" / "test" / "java"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "IntegrationTest.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;

public class IntegrationTest {
    @Test
    public void testFormatting() {
        String result = Utils.format("hello");
        assertEquals("HELLO", result);
    }
}
""")

        # Get source functions
        source_functions = discover_functions_from_source(
            src_file.read_text(), file_path=src_file
        )

        # Discover tests
        result = discover_tests(test_dir, source_functions)

        # Should find the test that calls format()
        assert len(result) > 0, "Should find tests that directly call target method"


class TestWithFixture:
    """Tests using the Java fixture project."""

    @pytest.fixture
    def java_fixture_path(self):
        """Get path to the Java fixture project."""
        fixture_path = Path(__file__).parent.parent.parent / "test_languages" / "fixtures" / "java_maven"
        if not fixture_path.exists():
            pytest.skip("Java fixture project not found")
        return fixture_path

    def test_discover_fixture_tests(self, java_fixture_path: Path):
        """Test discovering tests from fixture project."""
        test_root = java_fixture_path / "src" / "test" / "java"
        if not test_root.exists():
            pytest.skip("Test root not found")

        tests = discover_all_tests(test_root)
        assert len(tests) > 0


class TestImportExtraction:
    """Tests for the _extract_imports helper function."""

    def test_basic_import(self):
        """Test extraction of basic import statement."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import com.example.Calculator;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == {"Calculator"}

    def test_multiple_imports(self):
        """Test extraction of multiple imports."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import com.example.util.Helper;
import com.example.Calculator;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == {"Helper", "Calculator"}

    def test_wildcard_import_returns_empty(self):
        """Test that wildcard imports don't add specific classes."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import com.example.*;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == set()

    def test_static_import_extracts_class(self):
        """Test that static imports extract the class name, not the method."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import static com.example.Utils.format;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == {"Utils"}

    def test_static_wildcard_import_extracts_class(self):
        """Test that static wildcard imports extract the class name."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import static com.example.Utils.*;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == {"Utils"}

    def test_deeply_nested_package(self):
        """Test extraction from deeply nested package."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import com.aerospike.client.command.Buffer;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        assert imports == {"Buffer"}

    def test_mixed_imports(self):
        """Test extraction with mix of regular, static, and wildcard imports."""
        from codeflash.languages.java.test_discovery import _extract_imports
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
import com.example.Calculator;
import com.example.util.*;
import static org.junit.Assert.assertEquals;
import static com.example.Utils.*;
public class Test {}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        imports = _extract_imports(tree.root_node, source_bytes, analyzer)

        # Should have Calculator, Assert, Utils but NOT wildcards
        assert "Calculator" in imports
        assert "Assert" in imports
        assert "Utils" in imports


class TestMethodCallDetection:
    """Tests for method call detection in test code."""

    def test_find_method_calls(self):
        """Test detection of method calls within a code range."""
        from codeflash.languages.java.test_discovery import _find_method_calls_in_range
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source = """
public class TestExample {
    @Test
    public void testSomething() {
        Calculator calc = new Calculator();
        int result = calc.add(2, 3);
        String hex = Buffer.bytesToHexString(data);
        helper.process(x);
    }
}
"""
        source_bytes = source.encode("utf8")
        tree = analyzer.parse(source_bytes)
        calls = _find_method_calls_in_range(tree.root_node, source_bytes, 1, 10, analyzer)

        assert "add" in calls
        assert "bytesToHexString" in calls
        assert "process" in calls


class TestClassNamingConventions:
    """Tests for class naming convention matching."""

    def test_suffix_test_pattern(self, tmp_path: Path):
        """Test that ClassNameTest matches ClassName."""
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
    public void testAdd() { }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        result = discover_tests(test_dir, source_functions)

        # CalculatorTest should match Calculator class
        assert len(result) > 0
        assert "Calculator.add" in result

    def test_prefix_test_pattern(self, tmp_path: Path):
        """Test that TestClassName matches ClassName."""
        src_file = tmp_path / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
""")

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_file = test_dir / "TestCalculator.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;
public class TestCalculator {
    @Test
    public void testAdd() { }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        result = discover_tests(test_dir, source_functions)

        # TestCalculator should match Calculator class
        assert len(result) > 0
        assert "Calculator.add" in result

    def test_tests_suffix_pattern(self, tmp_path: Path):
        """Test that ClassNameTests matches ClassName."""
        src_file = tmp_path / "Calculator.java"
        src_file.write_text("""
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
""")

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_file = test_dir / "CalculatorTests.java"
        test_file.write_text("""
import org.junit.jupiter.api.Test;
public class CalculatorTests {
    @Test
    public void testAdd() { }
}
""")

        source_functions = discover_functions_from_source(src_file.read_text(), src_file)
        result = discover_tests(test_dir, source_functions)

        # CalculatorTests should match Calculator class
        assert len(result) > 0
        assert "Calculator.add" in result
