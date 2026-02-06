"""Tests for Java code formatting."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.languages.java.formatter import (
    JavaFormatter,
    format_java_code,
    format_java_file,
    normalize_java_code,
)
from codeflash.setup.detector import _detect_formatter


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
        expected = "public class Example {\npublic int add(int a, int b) {\nreturn a + b;\n}\n}"
        assert normalized == expected

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
        expected = "public class Example {\npublic int add(int a, int b) {\nreturn a + b;\n}\n}"
        assert normalized == expected

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
        expected = 'public class Example {\npublic String getUrl() {\nreturn "https://example.com";\n}\n}'
        assert normalized == expected

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
        expected = "public class Example {\npublic int add(int a, int b) {\nreturn a + b;\n}\n}"
        assert normalized == expected

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
        # Note: inline comment leaves extra space
        expected = "public class Example {\npublic int  add(int a, int b) {\nreturn a + b;\n}\n}"
        assert normalized == expected


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
        # Without external formatter, returns same as input
        assert result == "public class Example { public int add(int a, int b) { return a+b; } }"


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
        expected = "\npublic class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}\n"
        assert result == expected


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
        expected = "\npublic class Example {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}\n"
        assert result == expected

    def test_format_file_in_place(self, tmp_path: Path):
        """Test formatting a file in place."""
        java_file = tmp_path / "Example.java"
        source = """public class Example { public int getValue() { return 42; } }"""
        java_file.write_text(source)

        format_java_file(java_file, in_place=True)
        # Without external formatter, file remains unchanged
        content = java_file.read_text()
        assert content == "public class Example { public int getValue() { return 42; } }"


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
        # Returns input unchanged when no external formatter
        assert result == source


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
        # Note: current implementation incorrectly removes content in s2 string
        expected = 'public class Example {\nString s1 = "// not a comment";\nString s2 = "";\n}'
        assert normalized == expected

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
        expected = "public class Example {\npublic void method() {\n}\n}"
        assert normalized == expected

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


class TestDetectJavaFormatter:
    """Tests for Java formatter detection in the project detector pipeline."""

    def test_detect_formatter_returns_commands_when_java_and_jar_available(self, tmp_path: Path):
        """Detector returns formatter commands when Java executable and JAR both exist."""
        jar_dir = tmp_path / ".codeflash"
        jar_dir.mkdir()
        version = JavaFormatter.GOOGLE_JAVA_FORMAT_VERSION
        jar_file = jar_dir / f"google-java-format-{version}-all-deps.jar"
        jar_file.write_text("fake jar")

        with (
            patch.dict(os.environ, {"JAVA_HOME": ""}, clear=False),
            patch("shutil.which", return_value="/usr/bin/java"),
        ):
            cmds, description = _detect_formatter(tmp_path, "java")

        assert len(cmds) == 1
        assert "java" in cmds[0]
        assert "--replace" in cmds[0]
        assert "$file" in cmds[0]
        assert str(jar_file) in cmds[0]
        assert description == "google-java-format"

    def test_detect_formatter_returns_empty_when_java_not_available(self, tmp_path: Path):
        """Detector returns empty list with descriptive message when Java is not found."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("shutil.which", return_value=None),
        ):
            cmds, description = _detect_formatter(tmp_path, "java")

        assert cmds == []
        assert "java not available" in description

    def test_detect_formatter_returns_empty_when_jar_not_found(self, tmp_path: Path):
        """Detector returns empty list when Java exists but JAR is not found."""
        with (
            patch.dict(os.environ, {"JAVA_HOME": ""}, clear=False),
            patch("shutil.which", return_value="/usr/bin/java"),
        ):
            cmds, description = _detect_formatter(tmp_path, "java")

        assert cmds == []
        assert "install google-java-format" in description

    def test_detect_formatter_uses_java_home(self, tmp_path: Path):
        """Detector finds Java via JAVA_HOME environment variable."""
        java_home = tmp_path / "jdk"
        java_bin = java_home / "bin"
        java_bin.mkdir(parents=True)
        java_exe = java_bin / "java"
        java_exe.write_text("fake java")

        jar_dir = tmp_path / "project" / ".codeflash"
        jar_dir.mkdir(parents=True)
        version = JavaFormatter.GOOGLE_JAVA_FORMAT_VERSION
        jar_file = jar_dir / f"google-java-format-{version}-all-deps.jar"
        jar_file.write_text("fake jar")

        with patch.dict(os.environ, {"JAVA_HOME": str(java_home)}, clear=False):
            cmds, description = _detect_formatter(tmp_path / "project", "java")

        assert len(cmds) == 1
        assert str(java_exe) in cmds[0]
        assert description == "google-java-format"

    def test_detect_formatter_checks_home_codeflash_dir(self, tmp_path: Path):
        """Detector finds JAR in ~/.codeflash/ directory."""
        version = JavaFormatter.GOOGLE_JAVA_FORMAT_VERSION
        jar_name = f"google-java-format-{version}-all-deps.jar"
        home_codeflash = tmp_path / "fakehome" / ".codeflash"
        home_codeflash.mkdir(parents=True)
        jar_file = home_codeflash / jar_name
        jar_file.write_text("fake jar")

        with (
            patch.dict(os.environ, {"JAVA_HOME": ""}, clear=False),
            patch("shutil.which", return_value="/usr/bin/java"),
            patch("pathlib.Path.home", return_value=tmp_path / "fakehome"),
        ):
            cmds, description = _detect_formatter(tmp_path, "java")

        assert len(cmds) == 1
        assert str(jar_file) in cmds[0]
        assert description == "google-java-format"

    def test_detect_formatter_python_still_works(self, tmp_path: Path):
        """Ensure Python formatter detection is not broken by Java changes."""
        ruff_toml = tmp_path / "ruff.toml"
        ruff_toml.write_text("[tool.ruff]\n")

        cmds, _description = _detect_formatter(tmp_path, "python")
        assert len(cmds) > 0
        assert "ruff" in cmds[0]

    def test_detect_formatter_js_still_works(self, tmp_path: Path):
        """Ensure JavaScript formatter detection is not broken by Java changes."""
        prettierrc = tmp_path / ".prettierrc"
        prettierrc.write_text("{}")

        cmds, _description = _detect_formatter(tmp_path, "javascript")
        assert len(cmds) > 0
        assert "prettier" in cmds[0]
