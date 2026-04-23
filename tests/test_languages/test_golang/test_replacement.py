from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.replacement import add_global_declarations, remove_test_functions, replace_function
from codeflash.models.function_types import FunctionParent, FunctionToOptimize


class TestReplaceFunction:
    def test_replace_basic_function(self) -> None:
        source = "package calc\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\nfunc Subtract(a, b int) int {\n\treturn a - b\n}\n"
        func = FunctionToOptimize(function_name="Add", file_path=Path("/project/calc.go"), language="go")
        new_body = "func Add(a, b int) int {\n\tresult := a + b\n\treturn result\n}"
        result = replace_function(source, func, new_body)
        expected = "package calc\n\nfunc Add(a, b int) int {\n\tresult := a + b\n\treturn result\n}\n\nfunc Subtract(a, b int) int {\n\treturn a - b\n}\n"
        assert result == expected

    def test_replace_function_with_doc_comment(self) -> None:
        source = "package calc\n\n// Add returns the sum.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        func = FunctionToOptimize(function_name="Add", file_path=Path("/project/calc.go"), language="go")
        new_body = "// Add returns an optimized sum.\nfunc Add(a, b int) int {\n\treturn a + b\n}"
        result = replace_function(source, func, new_body)
        expected = "package calc\n\n// Add returns an optimized sum.\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        assert result == expected

    def test_replace_method(self) -> None:
        source = (
            "package calc\n\n"
            "type Calc struct {\n\tResult float64\n}\n\n"
            "// AddFloat adds a value.\n"
            "func (c *Calc) AddFloat(val float64) float64 {\n\tc.Result += val\n\treturn c.Result\n}\n\n"
            "func (c Calc) GetResult() float64 {\n\treturn c.Result\n}\n"
        )
        func = FunctionToOptimize(
            function_name="AddFloat",
            file_path=Path("/project/calc.go"),
            parents=[FunctionParent(name="Calc", type="StructDef")],
            language="go",
            is_method=True,
        )
        new_body = "// AddFloat adds a value (optimized).\nfunc (c *Calc) AddFloat(val float64) float64 {\n\tc.Result = c.Result + val\n\treturn c.Result\n}"
        result = replace_function(source, func, new_body)
        expected = (
            "package calc\n\n"
            "type Calc struct {\n\tResult float64\n}\n\n"
            "// AddFloat adds a value (optimized).\n"
            "func (c *Calc) AddFloat(val float64) float64 {\n\tc.Result = c.Result + val\n\treturn c.Result\n}\n\n"
            "func (c Calc) GetResult() float64 {\n\treturn c.Result\n}\n"
        )
        assert result == expected

    def test_replace_nonexistent_returns_original(self) -> None:
        source = "package calc\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        func = FunctionToOptimize(function_name="Missing", file_path=Path("/project/calc.go"), language="go")
        result = replace_function(source, func, "func Missing() {}")
        assert result == source

    def test_replace_preserves_surrounding_code(self) -> None:
        source = (
            "package calc\n\n"
            "var version = 1\n\n"
            "func Add(a, b int) int {\n\treturn a + b\n}\n\n"
            "func Subtract(a, b int) int {\n\treturn a - b\n}\n"
        )
        func = FunctionToOptimize(function_name="Add", file_path=Path("/project/calc.go"), language="go")
        new_body = "func Add(a, b int) int {\n\treturn b + a\n}"
        result = replace_function(source, func, new_body)
        expected = (
            "package calc\n\n"
            "var version = 1\n\n"
            "func Add(a, b int) int {\n\treturn b + a\n}\n\n"
            "func Subtract(a, b int) int {\n\treturn a - b\n}\n"
        )
        assert result == expected


class TestAddGlobalDeclarations:
    def test_add_import_to_existing_block(self) -> None:
        original = 'package calc\n\nimport (\n\t"fmt"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        optimized = 'package calc\n\nimport (\n\t"fmt"\n\t"math"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        result = add_global_declarations(optimized, original)
        expected = 'package calc\n\nimport (\n\t"fmt"\n\t"math"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        assert result == expected

    def test_add_aliased_import(self) -> None:
        original = 'package calc\n\nimport (\n\t"fmt"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        optimized = 'package calc\n\nimport (\n\t"fmt"\n\tstr "strings"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        result = add_global_declarations(optimized, original)
        expected = 'package calc\n\nimport (\n\t"fmt"\n\tstr "strings"\n)\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        assert result == expected

    def test_add_import_when_no_existing_imports(self) -> None:
        original = "package calc\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        optimized = 'package calc\n\nimport "math"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        result = add_global_declarations(optimized, original)
        expected = 'package calc\nimport (\n\t"math"\n)\n\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n'
        assert result == expected

    def test_no_new_imports_returns_unchanged(self) -> None:
        source = "package calc\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
        result = add_global_declarations(source, source)
        assert result == source


class TestRemoveTestFunctions:
    def test_remove_single_function(self) -> None:
        test_source = (
            "package calc\n\n"
            'import "testing"\n\n'
            "func TestAdd(t *testing.T) {\n"
            "\tresult := Add(2, 3)\n"
            "\tif result != 5 {\n"
            '\t\tt.Errorf("want 5, got %d", result)\n'
            "\t}\n"
            "}\n\n"
            "func TestSubtract(t *testing.T) {\n"
            "\tresult := Subtract(5, 3)\n"
            "\tif result != 2 {\n"
            '\t\tt.Errorf("want 2, got %d", result)\n'
            "\t}\n"
            "}\n"
        )
        result = remove_test_functions(test_source, ["TestAdd"])
        expected = (
            "package calc\n\n"
            'import "testing"\n\n\n'
            "func TestSubtract(t *testing.T) {\n"
            "\tresult := Subtract(5, 3)\n"
            "\tif result != 2 {\n"
            '\t\tt.Errorf("want 2, got %d", result)\n'
            "\t}\n"
            "}\n"
        )
        assert result == expected

    def test_remove_multiple_functions(self) -> None:
        test_source = (
            "package calc\n\n"
            'import "testing"\n\n'
            "// TestAdd tests addition.\n"
            "func TestAdd(t *testing.T) {\n"
            "\tif Add(1, 2) != 3 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n\n"
            "func TestSubtract(t *testing.T) {\n"
            "\tif Subtract(5, 3) != 2 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n\n"
            "func TestMultiply(t *testing.T) {\n"
            "\tif Multiply(2, 3) != 6 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n"
        )
        result = remove_test_functions(test_source, ["TestAdd", "TestMultiply"])
        expected = (
            "package calc\n\n"
            'import "testing"\n\n\n'
            "func TestSubtract(t *testing.T) {\n"
            "\tif Subtract(5, 3) != 2 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n\n"
        )
        assert result == expected

    def test_remove_function_with_doc_comment(self) -> None:
        test_source = (
            "package calc\n\n"
            'import "testing"\n\n'
            "// TestAdd tests addition.\n"
            "func TestAdd(t *testing.T) {\n"
            "\tif Add(1, 2) != 3 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n\n"
            "func TestSubtract(t *testing.T) {\n"
            "\tif Subtract(5, 3) != 2 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n"
        )
        result = remove_test_functions(test_source, ["TestAdd"])
        expected = (
            "package calc\n\n"
            'import "testing"\n\n\n'
            "func TestSubtract(t *testing.T) {\n"
            "\tif Subtract(5, 3) != 2 {\n"
            "\t\tt.Fail()\n"
            "\t}\n"
            "}\n"
        )
        assert result == expected

    def test_remove_none_returns_unchanged(self) -> None:
        test_source = "package calc\n\nfunc TestAdd(t *testing.T) {\n\tt.Log(\"ok\")\n}\n"
        result = remove_test_functions(test_source, [])
        assert result == test_source
