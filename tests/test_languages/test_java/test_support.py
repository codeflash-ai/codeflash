"""Tests for the JavaSupport class."""

from pathlib import Path

import pytest

from codeflash.languages.base import Language, LanguageSupport
from codeflash.languages.java.support import JavaSupport, get_java_support


class TestJavaSupportProtocol:
    """Tests that JavaSupport implements the LanguageSupport protocol."""

    @pytest.fixture
    def support(self):
        """Get a JavaSupport instance."""
        return get_java_support()

    def test_implements_protocol(self, support):
        """Test that JavaSupport implements LanguageSupport."""
        assert isinstance(support, LanguageSupport)

    def test_language_property(self, support):
        """Test the language property."""
        assert support.language == Language.JAVA

    def test_file_extensions(self, support):
        """Test the file extensions property."""
        assert support.file_extensions == (".java",)

    def test_test_framework(self, support):
        """Test the test framework property."""
        assert support.test_framework == "junit5"

    def test_comment_prefix(self, support):
        """Test the comment prefix property."""
        assert support.comment_prefix == "//"


class TestJavaSupportFunctions:
    """Tests for JavaSupport methods."""

    @pytest.fixture
    def support(self):
        """Get a JavaSupport instance."""
        return get_java_support()

    def test_discover_functions(self, support, tmp_path: Path):
        """Test function discovery."""
        java_file = tmp_path / "Calculator.java"
        java_file.write_text("""
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
""")

        functions = support.discover_functions(java_file)
        assert len(functions) == 1
        assert functions[0].name == "add"
        assert functions[0].language == Language.JAVA

    def test_validate_syntax_valid(self, support):
        """Test syntax validation with valid code."""
        source = """
public class Test {
    public void method() {}
}
"""
        assert support.validate_syntax(source) is True

    def test_validate_syntax_invalid(self, support):
        """Test syntax validation with invalid code."""
        source = """
public class Test {
    public void method() {
"""
        assert support.validate_syntax(source) is False

    def test_normalize_code(self, support):
        """Test code normalization."""
        source = """
// Comment
public class Test {
    /* Block comment */
    public void method() {}
}
"""
        normalized = support.normalize_code(source)
        # Comments should be removed
        assert "//" not in normalized
        assert "/*" not in normalized

    def test_get_test_file_suffix(self, support):
        """Test getting test file suffix."""
        assert support.get_test_file_suffix() == "Test.java"

    def test_get_comment_prefix(self, support):
        """Test getting comment prefix."""
        assert support.get_comment_prefix() == "//"


class TestJavaSupportWithFixture:
    """Tests using the Java fixture project."""

    @pytest.fixture
    def java_fixture_path(self):
        """Get path to the Java fixture project."""
        fixture_path = Path(__file__).parent.parent.parent / "test_languages" / "fixtures" / "java_maven"
        if not fixture_path.exists():
            pytest.skip("Java fixture project not found")
        return fixture_path

    @pytest.fixture
    def support(self):
        """Get a JavaSupport instance."""
        return get_java_support()

    def test_find_test_root(self, support, java_fixture_path: Path):
        """Test finding test root."""
        test_root = support.find_test_root(java_fixture_path)
        assert test_root is not None
        assert test_root.exists()
        assert "test" in str(test_root)

    def test_discover_functions_from_fixture(self, support, java_fixture_path: Path):
        """Test discovering functions from fixture."""
        calculator_file = java_fixture_path / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
        if not calculator_file.exists():
            pytest.skip("Calculator.java not found")

        functions = support.discover_functions(calculator_file)
        assert len(functions) > 0
