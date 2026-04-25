from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.support import GoSupport
from codeflash.languages.language_enum import Language
from codeflash.languages.registry import get_language_support


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

    def test_get_module_path(self) -> None:
        support = GoSupport()
        source_file = Path("/project/pkg/calc.go")
        result = support.get_module_path(source_file, Path("/project"))
        assert result == str(source_file)

    def test_setup_test_config_returns_true(self) -> None:
        support = GoSupport()

        class FakeTestCfg:
            project_root_path = Path("/nonexistent")

        assert support.setup_test_config(FakeTestCfg(), Path("/file.go")) is True

    def test_prepare_module_valid(self, tmp_path: Path) -> None:
        from codeflash.models.models import ValidCode

        support = GoSupport()
        code = "package main\n\nfunc main() {}\n"
        module_path = (tmp_path / "main.go").resolve()
        result = support.prepare_module(code, module_path, tmp_path)
        assert result is not None
        validated, ast_node = result
        assert ast_node is None
        assert module_path in validated
        assert isinstance(validated[module_path], ValidCode)
        assert validated[module_path].source_code == code

    def test_prepare_module_invalid(self, tmp_path: Path) -> None:
        support = GoSupport()
        result = support.prepare_module("func {{{ invalid", (tmp_path / "bad.go").resolve(), tmp_path)
        assert result is None

    def test_instrument_existing_test_reads_file(self, tmp_path: Path) -> None:
        support = GoSupport()
        test_file = (tmp_path / "calc_test.go").resolve()
        test_file.write_text("package calc\n\nfunc TestAdd(t *testing.T) {}\n", encoding="utf-8")
        success, content = support.instrument_existing_test(
            test_path=test_file, call_positions=[], function_to_optimize=None, tests_project_root=tmp_path, mode="behavior"
        )
        assert success is True
        assert content is not None
        assert "TestAdd" in content

    def test_instrument_existing_test_missing_file(self, tmp_path: Path) -> None:
        support = GoSupport()
        success, content = support.instrument_existing_test(
            test_path=(tmp_path / "missing.go").resolve(),
            call_positions=[],
            function_to_optimize=None,
            tests_project_root=tmp_path,
            mode="behavior",
        )
        assert success is False
        assert content is None

    def test_postprocess_generated_tests_passthrough(self) -> None:
        support = GoSupport()
        sentinel = object()
        result = support.postprocess_generated_tests(sentinel, "go-test", Path("/project"), Path("/project/calc.go"))  # type: ignore[arg-type]
        assert result is sentinel

    def test_process_generated_test_strings_passthrough(self) -> None:
        support = GoSupport()
        gen, beh, perf = support.process_generated_test_strings(
            "gen_code", "beh_code", "perf_code", None, Path("/test.go"), None, None
        )
        assert gen == "gen_code"
        assert beh == "beh_code"
        assert perf == "perf_code"

    def test_add_runtime_comments_to_generated_tests_passthrough(self) -> None:
        support = GoSupport()
        sentinel = object()
        result = support.add_runtime_comments_to_generated_tests(sentinel, {}, {})  # type: ignore[arg-type]
        assert result is sentinel

    def test_remove_test_functions_from_generated_tests(self) -> None:
        from codeflash.models.models import GeneratedTests, GeneratedTestsList

        support = GoSupport()
        source = """\
package calc

import "testing"

func TestAdd(t *testing.T) {
\tif Add(1, 2) != 3 {
\t\tt.Fatal("bad")
\t}
}

func TestSub(t *testing.T) {
\tif Sub(3, 1) != 2 {
\t\tt.Fatal("bad")
\t}
}
"""
        gt = GeneratedTests(
            generated_original_test_source=source,
            instrumented_behavior_test_source=source,
            instrumented_perf_test_source=source,
            behavior_file_path=Path("/test_beh.go"),
            perf_file_path=Path("/test_perf.go"),
        )
        tests_list = GeneratedTestsList(generated_tests=[gt])
        result = support.remove_test_functions_from_generated_tests(tests_list, ["TestSub"])
        assert "TestAdd" in result.generated_tests[0].generated_original_test_source
        assert "TestSub" not in result.generated_tests[0].generated_original_test_source
