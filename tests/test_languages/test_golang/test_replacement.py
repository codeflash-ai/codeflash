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


class TestAddGlobalDeclarationsImports:
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


class TestAddGlobalDeclarationsNewVar:
    def test_add_single_new_var(self) -> None:
        original = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "var cache = make(map[int]int)\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n"
            "var cache = make(map[int]int)\n\n"
            "\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        assert result == expected

    def test_add_grouped_var_block(self) -> None:
        original = (
            "package server\n\n"
            'import "fmt"\n\n'
            "func Process() {\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        optimized = (
            "package server\n\n"
            'import "fmt"\n\n'
            "var (\n"
            "\tcache  map[string]int\n"
            "\tbuffer []byte\n"
            ")\n\n"
            "func Process() {\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package server\n\n"
            'import "fmt"\n'
            "var (\n"
            "\tcache  map[string]int\n"
            "\tbuffer []byte\n"
            ")\n\n"
            "\n"
            "func Process() {\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        assert result == expected

    def test_add_new_var_preserves_existing_var(self) -> None:
        original = (
            "package calc\n\n"
            "var version = 1\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "var version = 1\n\n"
            "var cache = make(map[int]int)\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n\n"
            "var version = 1\n"
            "var cache = make(map[int]int)\n\n"
            "\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        assert result == expected


class TestAddGlobalDeclarationsNewConst:
    def test_add_single_new_const(self) -> None:
        original = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "const maxSize = 1024\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n"
            "const maxSize = 1024\n\n"
            "\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        assert result == expected

    def test_add_grouped_const_block(self) -> None:
        original = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "const (\n"
            "\tMaxRetries = 5\n"
            "\tTimeout    = 30\n"
            ")\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n"
            "const (\n"
            "\tMaxRetries = 5\n"
            "\tTimeout    = 30\n"
            ")\n\n"
            "\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        assert result == expected

    def test_add_new_const_preserves_existing_const(self) -> None:
        original = (
            "package calc\n\n"
            "const Pi = 3.14\n\n"
            "func Area(r float64) float64 {\n"
            "\treturn Pi * r * r\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "const Pi = 3.14\n\n"
            "const TwoPi = 6.28\n\n"
            "func Area(r float64) float64 {\n"
            "\treturn Pi * r * r\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n\n"
            "const Pi = 3.14\n"
            "const TwoPi = 6.28\n\n"
            "\n"
            "func Area(r float64) float64 {\n"
            "\treturn Pi * r * r\n"
            "}\n"
        )
        assert result == expected


class TestAddGlobalDeclarationsModifyVar:
    def test_modify_single_var_value(self) -> None:
        original = (
            "package calc\n\n"
            "var bufferSize = 256\n\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "var bufferSize = 1024\n\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n\n"
            "var bufferSize = 1024\n"
            "\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        assert result == expected

    def test_modify_grouped_var_block(self) -> None:
        original = (
            "package server\n\n"
            "var (\n"
            '\thost = "localhost"\n'
            "\tport = 8080\n"
            ")\n\n"
            "func Addr() string {\n"
            "\treturn host\n"
            "}\n"
        )
        optimized = (
            "package server\n\n"
            "var (\n"
            '\thost = "0.0.0.0"\n'
            "\tport = 9090\n"
            ")\n\n"
            "func Addr() string {\n"
            "\treturn host\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package server\n\n"
            "var (\n"
            '\thost = "0.0.0.0"\n'
            "\tport = 9090\n"
            ")\n"
            "\n"
            "func Addr() string {\n"
            "\treturn host\n"
            "}\n"
        )
        assert result == expected

    def test_modify_var_type(self) -> None:
        original = (
            "package calc\n\n"
            "var counter int\n\n"
            "func Inc() {\n"
            "\tcounter++\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "var counter int64\n\n"
            "func Inc() {\n"
            "\tcounter++\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n\n"
            "var counter int64\n"
            "\n"
            "func Inc() {\n"
            "\tcounter++\n"
            "}\n"
        )
        assert result == expected


class TestAddGlobalDeclarationsModifyConst:
    def test_modify_single_const_value(self) -> None:
        original = (
            "package calc\n\n"
            "const MaxRetries = 3\n\n"
            "func Retries() int {\n"
            "\treturn MaxRetries\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "const MaxRetries = 10\n\n"
            "func Retries() int {\n"
            "\treturn MaxRetries\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n\n"
            "const MaxRetries = 10\n"
            "\n"
            "func Retries() int {\n"
            "\treturn MaxRetries\n"
            "}\n"
        )
        assert result == expected

    def test_modify_const_group(self) -> None:
        original = (
            "package server\n\n"
            "const (\n"
            "\tDefaultTimeout = 30\n"
            "\tMaxConnections = 100\n"
            ")\n\n"
            "func Config() int {\n"
            "\treturn DefaultTimeout\n"
            "}\n"
        )
        optimized = (
            "package server\n\n"
            "const (\n"
            "\tDefaultTimeout = 60\n"
            "\tMaxConnections = 500\n"
            ")\n\n"
            "func Config() int {\n"
            "\treturn DefaultTimeout\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package server\n\n"
            "const (\n"
            "\tDefaultTimeout = 60\n"
            "\tMaxConnections = 500\n"
            ")\n"
            "\n"
            "func Config() int {\n"
            "\treturn DefaultTimeout\n"
            "}\n"
        )
        assert result == expected


class TestAddGlobalDeclarationsMixed:
    def test_new_import_and_new_var(self) -> None:
        original = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            'import "sync"\n\n'
            "var mu sync.Mutex\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package calc\n"
            "import (\n"
            '\t"sync"\n'
            ")\n"
            "var mu sync.Mutex\n\n"
            "\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        assert result == expected

    def test_new_and_modified_globals_together(self) -> None:
        original = (
            "package server\n\n"
            "var bufferSize = 256\n\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        optimized = (
            "package server\n\n"
            "var bufferSize = 1024\n\n"
            "var cache = make(map[string]int)\n\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package server\n\n"
            "var bufferSize = 1024\n"
            "var cache = make(map[string]int)\n\n"
            "\n"
            "func Process() int {\n"
            "\treturn bufferSize\n"
            "}\n"
        )
        assert result == expected

    def test_no_globals_in_optimized_returns_unchanged(self) -> None:
        original = (
            "package calc\n\n"
            "var version = 1\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        optimized = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        assert result == original

    def test_identical_globals_returns_unchanged(self) -> None:
        source = (
            "package calc\n\n"
            "var version = 1\n\n"
            "const MaxSize = 100\n\n"
            "func Add(a, b int) int {\n"
            "\treturn a + b\n"
            "}\n"
        )
        result = add_global_declarations(source, source)
        assert result == source

    def test_full_round_trip_new_import_var_const(self) -> None:
        original = (
            "package server\n\n"
            "import (\n"
            '\t"fmt"\n'
            ")\n\n"
            "const Version = 1\n\n"
            "func Handle() {\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        optimized = (
            "package server\n\n"
            "import (\n"
            '\t"fmt"\n'
            '\t"sync"\n'
            ")\n\n"
            "const Version = 1\n\n"
            "var mu sync.Mutex\n\n"
            "const MaxConns = 100\n\n"
            "func Handle() {\n"
            "\tmu.Lock()\n"
            "\tdefer mu.Unlock()\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        result = add_global_declarations(optimized, original)
        expected = (
            "package server\n\n"
            "import (\n"
            '\t"fmt"\n'
            '\t"sync"\n'
            ")\n\n"
            "const Version = 1\n"
            "var mu sync.Mutex\n"
            "const MaxConns = 100\n\n"
            "\n"
            "func Handle() {\n"
            "\tfmt.Println()\n"
            "}\n"
        )
        assert result == expected


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
