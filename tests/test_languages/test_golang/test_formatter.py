from __future__ import annotations

from unittest.mock import patch

from codeflash.languages.golang.formatter import format_go_code, normalize_go_code


class TestNormalizeGoCode:
    def test_strips_line_comments(self) -> None:
        source = "package calc\n\n// Add returns the sum.\nfunc Add(a, b int) int {\n\treturn a + b // fast path\n}\n"
        result = normalize_go_code(source)
        expected = "package calc\nfunc Add(a, b int) int {\nreturn a + b\n}"
        assert result == expected

    def test_strips_single_line_block_comment(self) -> None:
        source = "package calc\n\n/* block comment */\nfunc Subtract(a, b int) int {\n\treturn a - b\n}\n"
        result = normalize_go_code(source)
        expected = "package calc\nfunc Subtract(a, b int) int {\nreturn a - b\n}"
        assert result == expected

    def test_strips_multi_line_block_comment(self) -> None:
        source = "package calc\n\n/*\nThis is a\nmulti-line comment.\n*/\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        result = normalize_go_code(source)
        expected = "package calc\nfunc Add(a, b int) int {\nreturn a + b\n}"
        assert result == expected

    def test_preserves_comment_in_string(self) -> None:
        source = 'func Greet() string {\n\treturn "hello // world"\n}\n'
        result = normalize_go_code(source)
        expected = 'func Greet() string {\nreturn "hello // world"\n}'
        assert result == expected

    def test_preserves_comment_in_raw_string(self) -> None:
        source = "func Greet() string {\n\treturn `hello // world`\n}\n"
        result = normalize_go_code(source)
        expected = "func Greet() string {\nreturn `hello // world`\n}"
        assert result == expected

    def test_strips_whitespace_and_empty_lines(self) -> None:
        source = "package calc\n\n\n\nfunc Add(a, b int) int {\n\t\treturn a + b\n\t}\n"
        result = normalize_go_code(source)
        expected = "package calc\nfunc Add(a, b int) int {\nreturn a + b\n}"
        assert result == expected

    def test_mixed_comments(self) -> None:
        source = (
            "package calc\n\n"
            "// Add returns the sum.\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b // fast path\n"
            "}\n\n"
            "/* block comment */\n"
            "func Subtract(a, b int) int {\n"
            "\treturn a - b\n"
            "}\n"
        )
        result = normalize_go_code(source)
        expected = "package calc\nfunc Add(a, b int) int {\nreturn a + b\n}\nfunc Subtract(a, b int) int {\nreturn a - b\n}"
        assert result == expected

    def test_inline_block_comment(self) -> None:
        source = "func Add(a /* first */, b int) int {\n\treturn a + b\n}\n"
        result = normalize_go_code(source)
        expected = "func Add(a , b int) int {\nreturn a + b\n}"
        assert result == expected

    def test_empty_input(self) -> None:
        assert normalize_go_code("") == ""

    def test_only_comments(self) -> None:
        source = "// just a comment\n// another line\n"
        result = normalize_go_code(source)
        assert result == ""


class TestFormatGoCode:
    def test_no_formatter_returns_source(self) -> None:
        source = "package calc\n\nfunc Add(a, b int) int {\nreturn a+b\n}\n"
        with patch("codeflash.languages.golang.formatter.shutil.which", return_value=None):
            result = format_go_code(source)
        assert result == source

    def test_format_with_gofmt(self) -> None:
        import shutil

        if shutil.which("gofmt") is None:
            return
        source = "package calc\n\nfunc  Add(a,b int)int{\nreturn a+b\n}\n"
        result = format_go_code(source)
        assert result != source
        assert "func Add" in result

    def test_format_failure_returns_source(self) -> None:
        source = "this is not valid go"
        with patch("codeflash.languages.golang.formatter.shutil.which", return_value="/usr/bin/gofmt"):
            with patch("codeflash.languages.golang.formatter.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 2
                mock_run.return_value.stderr = "syntax error"
                result = format_go_code(source)
        assert result == source
