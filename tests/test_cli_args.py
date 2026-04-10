"""Tests for CLI argument parsing, specifically the optimize subparser flag isolation."""

import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _clear_parser_cache():
    """Clear the lru_cache on _build_parser between tests."""
    from codeflash.cli_cmds.cli import _build_parser

    _build_parser.cache_clear()
    yield
    _build_parser.cache_clear()


class TestOptimizeSubparserFlags:
    """Test that flags defined on the main parser are also recognized by the optimize subparser."""

    def _parse(self, argv: list[str]) -> tuple:
        from codeflash.cli_cmds.cli import _build_parser

        with mock.patch.object(sys, "argv", argv):
            parser = _build_parser()
            args, unknown = parser.parse_known_args()
        return args, unknown

    def test_verbose_flag_recognized_by_optimize_subparser(self) -> None:
        args, unknown = self._parse(["codeflash", "optimize", "--verbose", "mvn", "test"])
        assert args.verbose is True, f"--verbose should be True, got {args.verbose}"
        assert "--verbose" not in unknown, f"--verbose leaked into unknown_args: {unknown}"

    def test_no_pr_flag_recognized_by_optimize_subparser(self) -> None:
        args, unknown = self._parse(["codeflash", "optimize", "--no-pr", "mvn", "test"])
        assert args.no_pr is True, f"--no-pr should be True, got {args.no_pr}"
        assert "--no-pr" not in unknown, f"--no-pr leaked into unknown_args: {unknown}"

    def test_file_flag_recognized_by_optimize_subparser(self) -> None:
        args, unknown = self._parse(["codeflash", "optimize", "--file", "Foo.java", "mvn", "test"])
        assert args.file == "Foo.java", f"--file should be 'Foo.java', got {args.file}"
        assert "--file" not in unknown, f"--file leaked into unknown_args: {unknown}"
        assert "Foo.java" not in unknown, f"file value leaked into unknown_args: {unknown}"

    def test_function_flag_recognized_by_optimize_subparser(self) -> None:
        args, unknown = self._parse(["codeflash", "optimize", "--function", "bar", "mvn", "test"])
        assert args.function == "bar", f"--function should be 'bar', got {args.function}"
        assert "--function" not in unknown, f"--function leaked into unknown_args: {unknown}"

    def test_multiple_flags_recognized_by_optimize_subparser(self) -> None:
        args, unknown = self._parse(
            ["codeflash", "optimize", "--verbose", "--no-pr", "--file", "X.java", "--function", "foo", "mvn", "test"]
        )
        assert args.verbose is True
        assert args.no_pr is True
        assert args.file == "X.java"
        assert args.function == "foo"
        # Only the Java command should remain in unknown
        assert unknown == ["mvn", "test"], f"Expected ['mvn', 'test'], got {unknown}"

    def test_flags_after_java_command_not_leaked(self) -> None:
        args, unknown = self._parse(["codeflash", "optimize", "mvn", "test"])
        assert args.command == "optimize"
        assert unknown == ["mvn", "test"]

    def test_max_function_count_default_consistency(self) -> None:
        args, _ = self._parse(["codeflash", "optimize", "mvn", "test"])
        assert args.max_function_count == 256, (
            f"max_function_count default should be 256 (matching tracer), got {args.max_function_count}"
        )


class TestMainParserFlags:
    """Test that flags still work on the main parser (non-optimize path)."""

    def _parse(self, argv: list[str]) -> tuple:
        from codeflash.cli_cmds.cli import _build_parser

        with mock.patch.object(sys, "argv", argv):
            parser = _build_parser()
            args, unknown = parser.parse_known_args()
        return args, unknown

    def test_verbose_on_main_parser(self) -> None:
        args, _ = self._parse(["codeflash", "--verbose"])
        assert args.verbose is True

    def test_file_on_main_parser(self) -> None:
        args, _ = self._parse(["codeflash", "--file", "test.java"])
        assert args.file == "test.java"

    def test_no_pr_on_main_parser(self) -> None:
        args, _ = self._parse(["codeflash", "--no-pr"])
        assert args.no_pr is True
