from __future__ import annotations

from pathlib import Path

from codeflash.languages.base import Language
from codeflash.languages.golang.context import extract_code_context, find_helper_functions
from codeflash.models.function_types import FunctionParent, FunctionToOptimize

GO_SOURCE_WITH_METHOD = """\
package calc

import "math"

type Calculator struct {
\tResult float64
}

// Add returns the sum.
func Add(a, b int) int {
\treturn a + b
}

func subtract(a, b int) int {
\treturn a - b
}

func (c *Calculator) AddFloat(val float64) float64 {
\tc.Result += val
\treturn c.Result
}
"""


class TestExtractCodeContextFunction:
    def test_target_code_with_doc_comment(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_code == "// Add returns the sum.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"

    def test_target_code_no_doc(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="subtract", file_path=source_file, language="go", starting_line=14, ending_line=16
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_code == "func subtract(a, b int) int {\n\treturn a - b\n}"

    def test_imports_extracted(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.imports == ['"math"']

    def test_no_read_only_context_for_function(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.read_only_context == ""

    def test_helpers_only_includes_called_functions(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.helper_functions == []

    def test_helpers_includes_called_function(self, tmp_path: Path) -> None:
        source = (
            "package calc\n\n"
            "func helper(x int) int { return x * 2 }\n\n"
            "func Target(a int) int { return helper(a) }\n"
        )
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(source, encoding="utf-8")
        func = FunctionToOptimize(function_name="Target", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        helper_names = [h.name for h in ctx.helper_functions]
        assert helper_names == ["helper"]

    def test_language_is_go(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.language == Language.GO

    def test_target_file_path(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Add", file_path=source_file, language="go", starting_line=10, ending_line=12
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_file == source_file


class TestExtractCodeContextMethod:
    def test_method_target_code(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="AddFloat",
            file_path=source_file,
            parents=[FunctionParent(name="Calculator", type="StructDef")],
            language="go",
            is_method=True,
            starting_line=18,
            ending_line=21,
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_code == (
            "func (c *Calculator) AddFloat(val float64) float64 {\n"
            "\tc.Result += val\n"
            "\treturn c.Result\n"
            "}"
        )

    def test_method_read_only_context_is_struct(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="AddFloat",
            file_path=source_file,
            parents=[FunctionParent(name="Calculator", type="StructDef")],
            language="go",
            is_method=True,
            starting_line=18,
            ending_line=21,
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.read_only_context == "type Calculator struct {\n\tResult float64\n}"

    def test_method_helpers_exclude_self(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="AddFloat",
            file_path=source_file,
            parents=[FunctionParent(name="Calculator", type="StructDef")],
            language="go",
            is_method=True,
            starting_line=18,
            ending_line=21,
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.helper_functions == []

    def test_method_helpers_with_calls(self, tmp_path: Path) -> None:
        source = (
            "package calc\n\n"
            "type Calc struct{ Val int }\n\n"
            "func double(x int) int { return x * 2 }\n\n"
            "func (c *Calc) Compute() int { return double(c.Val) }\n"
        )
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(source, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Compute",
            file_path=source_file,
            parents=[FunctionParent(name="Calc", type="StructDef")],
            language="go",
            is_method=True,
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        helper_names = [h.name for h in ctx.helper_functions]
        assert helper_names == ["double"]
        assert "Compute" not in helper_names


class TestExtractCodeContextEdgeCases:
    def test_missing_file(self, tmp_path: Path) -> None:
        missing = (tmp_path / "missing.go").resolve()
        func = FunctionToOptimize(function_name="Foo", file_path=missing, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_code == ""
        assert ctx.language == Language.GO

    def test_function_not_in_source(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text("package calc\n\nfunc Other() {}\n", encoding="utf-8")
        func = FunctionToOptimize(function_name="Missing", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.target_code == ""

    def test_multi_import(self, tmp_path: Path) -> None:
        source = 'package calc\n\nimport (\n\t"fmt"\n\t"os"\n\tstr "strings"\n)\n\nfunc Hello() string {\n\treturn "hi"\n}\n'
        source_file = (tmp_path / "hello.go").resolve()
        source_file.write_text(source, encoding="utf-8")
        func = FunctionToOptimize(function_name="Hello", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert ctx.imports == ['"fmt"', '"os"', 'str "strings"']


GO_SOURCE_WITH_INIT = """\
package server

import "sync"

var (
\tglobalCache map[string]int
\tmu          sync.Mutex
)

const MaxRetries = 5

type Config struct {
\tName string
\tMax  int
}

func init() {
\tglobalCache = make(map[string]int)
\tglobalCache["default"] = 0
\tmu.Lock()
\tmu.Unlock()
}

func Process() int {
\treturn MaxRetries
}
"""


class TestExtractCodeContextWithInit:
    def test_init_in_read_only_context(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "server.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_INIT, encoding="utf-8")
        func = FunctionToOptimize(function_name="Process", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert "func init()" in ctx.read_only_context

    def test_init_referenced_globals_in_read_only_context(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "server.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_INIT, encoding="utf-8")
        func = FunctionToOptimize(function_name="Process", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert "globalCache" in ctx.read_only_context
        assert "mu" in ctx.read_only_context

    def test_init_not_in_helpers(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "server.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_INIT, encoding="utf-8")
        func = FunctionToOptimize(function_name="Process", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        helper_names = [h.name for h in ctx.helper_functions]
        assert "init" not in helper_names

    def test_no_init_no_extra_context(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "calc.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_METHOD, encoding="utf-8")
        func = FunctionToOptimize(function_name="Add", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        assert "func init()" not in ctx.read_only_context

    def test_full_init_read_only_context(self, tmp_path: Path) -> None:
        source_file = (tmp_path / "server.go").resolve()
        source_file.write_text(GO_SOURCE_WITH_INIT, encoding="utf-8")
        func = FunctionToOptimize(function_name="Process", file_path=source_file, language="go")
        ctx = extract_code_context(func, tmp_path.resolve())
        expected = (
            "var (\n"
            "\tglobalCache map[string]int\n"
            "\tmu          sync.Mutex\n"
            ")\n"
            "\n"
            "func init() {\n"
            "\tglobalCache = make(map[string]int)\n"
            "\tglobalCache[\"default\"] = 0\n"
            "\tmu.Lock()\n"
            "\tmu.Unlock()\n"
            "}"
        )
        assert ctx.read_only_context == expected

    def test_method_with_init_combines_struct_and_init_context(self, tmp_path: Path) -> None:
        source = """\
package server

var globalOffset = 10

type Calc struct {
\tVal int
}

func init() {
\tglobalOffset = 42
}

func (c *Calc) Compute() int {
\treturn c.Val + globalOffset
}
"""
        source_file = (tmp_path / "server.go").resolve()
        source_file.write_text(source, encoding="utf-8")
        func = FunctionToOptimize(
            function_name="Compute",
            file_path=source_file,
            parents=[FunctionParent(name="Calc", type="StructDef")],
            language="go",
            is_method=True,
        )
        ctx = extract_code_context(func, tmp_path.resolve())
        assert "type Calc struct" in ctx.read_only_context
        assert "func init()" in ctx.read_only_context
        assert "var globalOffset = 10" in ctx.read_only_context


class TestFindHelperFunctions:
    def test_skips_init_and_main(self, tmp_path: Path) -> None:
        source = "package main\n\nfunc init() { println() }\n\nfunc main() { println() }\n\nfunc Target() int { return 1 }\n"
        source_file = (tmp_path / "main.go").resolve()
        func = FunctionToOptimize(function_name="Target", file_path=source_file, language="go")
        helpers = find_helper_functions(source, func)
        helper_names = [h.name for h in helpers]
        assert "init" not in helper_names
        assert "main" not in helper_names

    def test_method_helpers_have_qualified_names(self, tmp_path: Path) -> None:
        source = (
            "package calc\n\n"
            "type Calc struct{}\n\n"
            "func (c Calc) Target() int { return c.Helper() }\n\n"
            "func (c Calc) Helper() int { return 2 }\n"
        )
        source_file = (tmp_path / "calc.go").resolve()
        func = FunctionToOptimize(
            function_name="Target",
            file_path=source_file,
            parents=[FunctionParent(name="Calc", type="StructDef")],
            language="go",
            is_method=True,
        )
        helpers = find_helper_functions(source, func)
        assert len(helpers) == 1
        assert helpers[0].qualified_name == "Calc.Helper"

    def test_transitive_helpers(self, tmp_path: Path) -> None:
        source = (
            "package calc\n\n"
            "func innerHelper(x int) int { return x }\n\n"
            "func outerHelper(x int) int { return innerHelper(x) }\n\n"
            "func Target(a int) int { return outerHelper(a) }\n"
        )
        source_file = (tmp_path / "calc.go").resolve()
        func = FunctionToOptimize(function_name="Target", file_path=source_file, language="go")
        helpers = find_helper_functions(source, func)
        helper_names = sorted(h.name for h in helpers)
        assert helper_names == ["innerHelper", "outerHelper"]

    def test_uncalled_functions_excluded(self, tmp_path: Path) -> None:
        source = (
            "package calc\n\n"
            "func unrelated() int { return 99 }\n\n"
            "func Target(a int) int { return a + 1 }\n"
        )
        source_file = (tmp_path / "calc.go").resolve()
        func = FunctionToOptimize(function_name="Target", file_path=source_file, language="go")
        helpers = find_helper_functions(source, func)
        assert helpers == []
