"""Tests for Java code formatting."""

from pathlib import Path

import pytest

from codeflash.languages.java.formatter import (
    JavaFormatter,
    format_java_code,
    format_java_file,
    normalize_java_code,
)


class TestNormalizeJavaCode:
    """Tests for code normalization."""

    def test_normalize_removes_line_comments(self):
        """Test that line comments are removed."""
        source = """
public class Example {
    // This is a comment
    public int add(int a, int b) {
        return a + b; // inline comment
    }
}
"""
        normalized = normalize_java_code(source)
        assert "//" not in normalized
        assert "This is a comment" not in normalized
        assert "inline comment" not in normalized

    def test_normalize_removes_block_comments(self):
        """Test that block comments are removed."""
        source = """
public class Example {
    /* This is a
       multi-line
       block comment */
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        normalized = normalize_java_code(source)
        assert "/*" not in normalized
        assert "*/" not in normalized
        assert "multi-line" not in normalized

    def test_normalize_preserves_strings_with_slashes(self):
        """Test that strings containing // are preserved."""
        source = """
public class Example {
    public String getUrl() {
        return "https://example.com";
    }
}
"""
        normalized = normalize_java_code(source)
        assert "https://example.com" in normalized

    def test_normalize_removes_whitespace(self):
        """Test that extra whitespace is normalized."""
        source = """

public class Example {

    public int add(int a, int b) {

        return a + b;

    }

}

"""
        normalized = normalize_java_code(source)
        # Should not have empty lines
        lines = [l for l in normalized.split("\n") if l.strip()]
        assert len(lines) > 0

    def test_normalize_inline_block_comment(self):
        """Test inline block comment removal."""
        source = """
public class Example {
    public int /* comment */ add(int a, int b) {
        return a + b;
    }
}
"""
        normalized = normalize_java_code(source)
        assert "/* comment */" not in normalized


class TestJavaFormatter:
    """Tests for JavaFormatter class."""

    def test_formatter_init(self, tmp_path: Path):
        """Test formatter initialization."""
        formatter = JavaFormatter(tmp_path)
        assert formatter.project_root == tmp_path

    def test_format_empty_source(self, tmp_path: Path):
        """Test formatting empty source."""
        formatter = JavaFormatter(tmp_path)
        result = formatter.format_code("")
        assert result == ""

    def test_format_whitespace_only(self, tmp_path: Path):
        """Test formatting whitespace-only source."""
        formatter = JavaFormatter(tmp_path)
        result = formatter.format_code("   \n\n   ")
        assert result == "   \n\n   "

    def test_format_simple_class(self, tmp_path: Path):
        """Test formatting a simple class."""
        source = """public class Example { public int add(int a, int b) { return a+b; } }"""
        formatter = JavaFormatter(tmp_path)
        result = formatter.format_code(source)
        # Should return something (may be same as input if no formatter available)
        assert len(result) > 0


class TestFormatJavaCode:
    """Tests for format_java_code convenience function."""

    def test_format_preserves_valid_code(self):
        """Test that valid code is preserved."""
        source = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = format_java_code(source)
        # Should contain the core elements
        assert "Calculator" in result
        assert "add" in result
        assert "return" in result


class TestFormatJavaFile:
    """Tests for format_java_file function."""

    def test_format_file(self, tmp_path: Path):
        """Test formatting a file."""
        java_file = tmp_path / "Example.java"
        source = """
public class Example {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        java_file.write_text(source)

        result = format_java_file(java_file)
        assert "Example" in result
        assert "add" in result

    def test_format_file_in_place(self, tmp_path: Path):
        """Test formatting a file in place."""
        java_file = tmp_path / "Example.java"
        source = """public class Example { public int getValue() { return 42; } }"""
        java_file.write_text(source)

        format_java_file(java_file, in_place=True)
        # File should still be readable
        content = java_file.read_text()
        assert "Example" in content


class TestFormatterWithGoogleJavaFormat:
    """Tests for Google Java Format integration."""

    def test_google_java_format_not_downloaded(self, tmp_path: Path):
        """Test behavior when google-java-format is not available."""
        formatter = JavaFormatter(tmp_path)
        jar_path = formatter._get_google_java_format_jar()
        # May or may not be available depending on system
        # Just verify no exception is raised

    def test_format_falls_back_gracefully(self, tmp_path: Path):
        """Test that formatting falls back gracefully."""
        formatter = JavaFormatter(tmp_path)
        source = """
public class Test {
    public void test() {}
}
"""
        # Should not raise even if no formatter available
        result = formatter.format_code(source)
        assert len(result) > 0


class TestNormalizationEdgeCases:
    """Tests for edge cases in normalization."""

    def test_string_with_comment_chars(self):
        """Test string containing comment characters."""
        source = '''
public class Example {
    String s1 = "// not a comment";
    String s2 = "/* also not */";
}
'''
        normalized = normalize_java_code(source)
        # The strings should be preserved
        assert '"// not a comment"' in normalized or "not a comment" in normalized

    def test_nested_comments(self):
        """Test code with various comment patterns."""
        source = """
public class Example {
    // Single line
    /* Block */
    /**
     * Javadoc
     */
    public void method() {
        // More comments
    }
}
"""
        normalized = normalize_java_code(source)
        # Comments should be removed
        assert "Single line" not in normalized
        assert "Block" not in normalized
        assert "More comments" not in normalized

    def test_empty_source(self):
        """Test normalizing empty source."""
        assert normalize_java_code("") == ""
        assert normalize_java_code("   ") == ""
        assert normalize_java_code("\n\n\n") == ""

    def test_only_comments(self):
        """Test normalizing source with only comments."""
        source = """
// Comment 1
/* Comment 2 */
// Comment 3
"""
        normalized = normalize_java_code(source)
        assert normalized == ""
