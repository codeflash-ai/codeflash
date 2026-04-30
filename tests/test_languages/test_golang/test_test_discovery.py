from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.test_discovery import (
    _extract_target_name,
    _extract_test_body,
    _test_calls_function,
    discover_tests,
)
from codeflash.models.function_types import FunctionToOptimize

GO_TEST_SOURCE = """\
package calc

import "testing"

func TestAdd(t *testing.T) {
\tresult := Add(2, 3)
\tif result != 5 {
\t\tt.Fail()
\t}
}

func TestSubtract(t *testing.T) {
\tresult := Subtract(5, 3)
\tif result != 2 {
\t\tt.Fail()
\t}
}

func TestHelper(t *testing.T) {
\tx := 1 + 2
\t_ = x
}
"""


class TestExtractTargetName:
    def test_simple(self) -> None:
        assert _extract_target_name("TestAdd") == "Add"

    def test_with_underscore_suffix(self) -> None:
        assert _extract_target_name("TestAdd_negative") == "Add"

    def test_long_name(self) -> None:
        assert _extract_target_name("TestFibonacci") == "Fibonacci"

    def test_bare_test(self) -> None:
        assert _extract_target_name("Test") is None

    def test_not_a_test(self) -> None:
        assert _extract_target_name("NotATest") is None


class TestExtractTestBody:
    def test_extracts_body(self) -> None:
        body = _extract_test_body(GO_TEST_SOURCE, "TestAdd")
        assert body == "\n\tresult := Add(2, 3)\n\tif result != 5 {\n\t\tt.Fail()\n\t}\n"

    def test_extracts_second_body(self) -> None:
        body = _extract_test_body(GO_TEST_SOURCE, "TestSubtract")
        assert body == "\n\tresult := Subtract(5, 3)\n\tif result != 2 {\n\t\tt.Fail()\n\t}\n"

    def test_missing_function(self) -> None:
        assert _extract_test_body(GO_TEST_SOURCE, "TestMissing") is None


class TestTestCallsFunction:
    def test_calls_add(self) -> None:
        assert _test_calls_function(GO_TEST_SOURCE, "TestAdd", "Add") is True

    def test_does_not_call_subtract(self) -> None:
        assert _test_calls_function(GO_TEST_SOURCE, "TestAdd", "Subtract") is False

    def test_helper_does_not_call_add(self) -> None:
        assert _test_calls_function(GO_TEST_SOURCE, "TestHelper", "Add") is False


class TestDiscoverTests:
    def test_matches_by_name_convention(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "calc.go").write_text(
            "package calc\n\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8"
        )
        (root / "calc_test.go").write_text(GO_TEST_SOURCE, encoding="utf-8")

        funcs = [FunctionToOptimize(function_name="Add", file_path=root / "calc.go", language="go")]
        result = discover_tests(root, funcs)
        assert "Add" in result
        assert len(result["Add"]) == 1
        assert result["Add"][0].test_name == "TestAdd"

    def test_matches_multiple_functions(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "calc.go").write_text(
            "package calc\n\nfunc Add(a, b int) int { return a + b }\n\nfunc Subtract(a, b int) int { return a - b }\n",
            encoding="utf-8",
        )
        (root / "calc_test.go").write_text(GO_TEST_SOURCE, encoding="utf-8")

        funcs = [
            FunctionToOptimize(function_name="Add", file_path=root / "calc.go", language="go"),
            FunctionToOptimize(function_name="Subtract", file_path=root / "calc.go", language="go"),
        ]
        result = discover_tests(root, funcs)
        assert "Add" in result
        assert "Subtract" in result
        assert result["Add"][0].test_name == "TestAdd"
        assert result["Subtract"][0].test_name == "TestSubtract"

    def test_no_match_returns_empty(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "calc.go").write_text(
            "package calc\n\nfunc Multiply(a, b int) int { return a * b }\n", encoding="utf-8"
        )
        (root / "calc_test.go").write_text(GO_TEST_SOURCE, encoding="utf-8")

        funcs = [FunctionToOptimize(function_name="Multiply", file_path=root / "calc.go", language="go")]
        result = discover_tests(root, funcs)
        assert "Multiply" not in result

    def test_no_test_files(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "calc.go").write_text("package calc\n\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8")

        funcs = [FunctionToOptimize(function_name="Add", file_path=root / "calc.go", language="go")]
        result = discover_tests(root, funcs)
        assert result == {}

    def test_subdirectory_test_files(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "calc.go").write_text(
            "package calc\n\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8"
        )
        (pkg / "calc_test.go").write_text(GO_TEST_SOURCE, encoding="utf-8")

        funcs = [FunctionToOptimize(function_name="Add", file_path=pkg / "calc.go", language="go")]
        result = discover_tests(root, funcs)
        assert "Add" in result
        assert result["Add"][0].test_file == pkg / "calc_test.go"

    def test_fallback_content_match(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        (root / "calc.go").write_text(
            "package calc\n\nfunc DoMath(a, b int) int { return a + b }\n", encoding="utf-8"
        )
        (root / "calc_test.go").write_text(
            'package calc\n\nimport "testing"\n\nfunc TestComputation(t *testing.T) {\n'
            "\tresult := DoMath(2, 3)\n\tif result != 5 {\n\t\tt.Fail()\n\t}\n}\n",
            encoding="utf-8",
        )

        funcs = [FunctionToOptimize(function_name="DoMath", file_path=root / "calc.go", language="go")]
        result = discover_tests(root, funcs)
        assert "DoMath" in result
        assert result["DoMath"][0].test_name == "TestComputation"
