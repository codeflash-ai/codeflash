from __future__ import annotations

from pathlib import Path

from codeflash.languages.base import FunctionFilterCriteria
from codeflash.languages.golang.discovery import discover_functions_from_source

GO_SOURCE = """\
package calculator

import "math"

// Add returns the sum of two integers.
func Add(a, b int) int {
	return a + b
}

func subtract(a, b int) int {
	return a - b
}

func init() {
	println("setup")
}

func main() {
	println("hello")
}

func noReturn() {
	println("hello")
}

type Calculator struct {
	Result float64
}

func (c *Calculator) AddFloat(val float64) float64 {
	c.Result += val
	return c.Result
}

func (c Calculator) GetResult() float64 {
	return c.Result
}

func Hypotenuse(a, b float64) float64 {
	return math.Sqrt(a*a + b*b)
}
"""

GO_TEST_SOURCE = """\
package calculator

import "testing"

func TestAdd(t *testing.T) {
	result := Add(2, 3)
	if result != 5 {
		t.Errorf("want 5, got %d", result)
	}
}
"""


class TestDiscoverFunctions:
    def test_discovers_exported_functions(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        names = [f.function_name for f in results]
        assert "Add" in names
        assert "Hypotenuse" in names

    def test_discovers_unexported_functions(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        names = [f.function_name for f in results]
        assert "subtract" in names
        assert "noReturn" in names

    def test_skips_init_and_main(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        names = [f.function_name for f in results]
        assert "init" not in names
        assert "main" not in names

    def test_skips_test_files(self) -> None:
        results = discover_functions_from_source(GO_TEST_SOURCE, Path("/project/calc_test.go"))
        assert len(results) == 0

    def test_discovers_methods(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        methods = [f for f in results if f.is_method]
        assert len(methods) == 2
        names = [m.function_name for m in methods]
        assert "AddFloat" in names
        assert "GetResult" in names

    def test_method_parents(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        method = next(f for f in results if f.function_name == "AddFloat")
        assert len(method.parents) == 1
        assert method.parents[0].name == "Calculator"
        assert method.parents[0].type == "StructDef"

    def test_language_is_go(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        for func in results:
            assert func.language == "go"

    def test_is_async_false(self) -> None:
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"))
        for func in results:
            assert func.is_async is False


class TestDiscoverWithFilters:
    def test_filter_export_only(self) -> None:
        criteria = FunctionFilterCriteria(require_export=True, require_return=False)
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        names = [f.function_name for f in results]
        assert "Add" in names
        assert "Hypotenuse" in names
        assert "subtract" not in names
        assert "noReturn" not in names

    def test_filter_require_return(self) -> None:
        criteria = FunctionFilterCriteria(require_export=False, require_return=True)
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        names = [f.function_name for f in results]
        assert "Add" in names
        assert "noReturn" not in names

    def test_filter_exclude_methods(self) -> None:
        criteria = FunctionFilterCriteria(require_export=False, require_return=False, include_methods=False)
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        methods = [f for f in results if f.is_method]
        assert len(methods) == 0

    def test_filter_exclude_pattern(self) -> None:
        criteria = FunctionFilterCriteria(
            require_export=False, require_return=False, exclude_patterns=["subtract"]
        )
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        names = [f.function_name for f in results]
        assert "subtract" not in names
        assert "Add" in names

    def test_filter_include_pattern(self) -> None:
        criteria = FunctionFilterCriteria(
            require_export=False, require_return=False, include_patterns=["Add*"]
        )
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        names = [f.function_name for f in results]
        assert "Add" in names
        assert "AddFloat" in names
        assert "subtract" not in names
        assert "Hypotenuse" not in names

    def test_filter_min_lines(self) -> None:
        criteria = FunctionFilterCriteria(require_export=False, require_return=False, min_lines=4)
        results = discover_functions_from_source(GO_SOURCE, Path("/project/calc.go"), criteria)
        for func in results:
            line_count = func.ending_line - func.starting_line + 1
            assert line_count >= 4
