from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.support import GoSupport
from codeflash.languages.language_enum import Language
from codeflash.languages.registry import clear_cache, clear_registry, get_language_support


class TestGoSupportProperties:
    def test_language(self) -> None:
        support = GoSupport()
        assert support.language == Language.GO

    def test_file_extensions(self) -> None:
        support = GoSupport()
        assert support.file_extensions == (".go",)

    def test_default_file_extension(self) -> None:
        support = GoSupport()
        assert support.default_file_extension == ".go"

    def test_test_framework(self) -> None:
        support = GoSupport()
        assert support.test_framework == "go-test"

    def test_comment_prefix(self) -> None:
        support = GoSupport()
        assert support.comment_prefix == "//"

    def test_valid_test_frameworks(self) -> None:
        support = GoSupport()
        assert support.valid_test_frameworks == ("go-test",)

    def test_serialization_format(self) -> None:
        support = GoSupport()
        assert support.test_result_serialization_format == "json"

    def test_get_test_file_suffix(self) -> None:
        support = GoSupport()
        assert support.get_test_file_suffix() == "_test.go"

    def test_dir_excludes(self) -> None:
        support = GoSupport()
        assert "vendor" in support.dir_excludes
        assert "testdata" in support.dir_excludes


class TestGoSupportRegistration:
    def test_lookup_by_language_enum(self) -> None:
        support = get_language_support(Language.GO)
        assert support.language == Language.GO

    def test_lookup_by_extension(self) -> None:
        support = get_language_support(Path("main.go"))
        assert support.language == Language.GO

    def test_lookup_by_string(self) -> None:
        support = get_language_support("go")
        assert support.language == Language.GO

    def test_lookup_by_dot_extension(self) -> None:
        support = get_language_support(".go")
        assert support.language == Language.GO


class TestGoSupportDiscoverFunctions:
    def test_discovers_functions(self) -> None:
        support = GoSupport()
        source = """\
package calc

func Add(a, b int) int {
	return a + b
}

func subtract(a, b int) int {
	return a - b
}
"""
        results = support.discover_functions(source, Path("/project/calc.go"))
        names = [f.function_name for f in results]
        assert "Add" in names
        assert "subtract" in names

    def test_validate_syntax_valid(self) -> None:
        support = GoSupport()
        assert support.validate_syntax("package main\n\nfunc main() {}") is True

    def test_validate_syntax_invalid(self) -> None:
        support = GoSupport()
        assert support.validate_syntax("func {{{ invalid") is False


class TestGoSupportHelpers:
    def test_find_test_root(self) -> None:
        support = GoSupport()
        root = Path("/project")
        assert support.find_test_root(root) == root

    def test_get_runtime_files(self) -> None:
        support = GoSupport()
        assert support.get_runtime_files() == []

    def test_instrument_for_behavior_passthrough(self) -> None:
        support = GoSupport()
        source = "package main\n\nfunc main() {}\n"
        assert support.instrument_for_behavior(source, []) == source

    def test_instrument_for_benchmarking_passthrough(self) -> None:
        support = GoSupport()
        source = "package main\n\nfunc Test() {}\n"
        result = support.instrument_for_benchmarking(source, None)  # type: ignore[arg-type]
        assert result == source

    def test_get_test_dir_for_source(self) -> None:
        support = GoSupport()
        source_file = Path("/project/pkg/calc.go")
        result = support.get_test_dir_for_source(Path("/project"), source_file)
        assert result == Path("/project/pkg")
