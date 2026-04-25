from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from codeflash.languages.golang.context import extract_code_context
from codeflash.languages.golang.function_optimizer import _build_optimization_context
from codeflash.models.function_types import FunctionParent, FunctionToOptimize

if TYPE_CHECKING:
    from codeflash.models.models import CodeOptimizationContext

# ---------------------------------------------------------------------------
# Realistic Go sources used across test classes
# ---------------------------------------------------------------------------

CALCULATOR_SOURCE = dedent("""\
    package calc

    import (
    \t"fmt"
    \t"math"
    \tstr "strings"
    )

    // Calculator holds running computation state.
    type Calculator struct {
    \tResult  float64
    \tHistory []float64
    }

    // Formatter controls output rendering.
    type Formatter interface {
    \tFormat(val float64) string
    }

    // Add returns the sum of two integers.
    func Add(a, b int) int {
    \treturn a + b
    }

    func subtract(a, b int) int {
    \treturn a - b
    }

    func multiply(a, b int) int {
    \treturn a * b
    }

    // Greet builds a greeting message.
    func Greet(name string) string {
    \treturn fmt.Sprintf("Hello, %s", str.TrimSpace(name))
    }

    // AddFloat adds a float value and records history.
    func (c *Calculator) AddFloat(val float64) float64 {
    \tc.Result += val
    \tc.History = append(c.History, c.Result)
    \treturn c.Result
    }

    // Sqrt computes the square root of the current result.
    func (c *Calculator) Sqrt() float64 {
    \tc.Result = math.Sqrt(c.Result)
    \tc.History = append(c.History, c.Result)
    \treturn c.Result
    }

    // Reset zeroes out the calculator.
    func (c Calculator) Reset() Calculator {
    \tc.Result = 0
    \tc.History = nil
    \treturn c
    }
""")

SIMPLE_SOURCE = dedent("""\
    package simple

    func Double(x int) int {
    \treturn x * 2
    }
""")

INIT_SOURCE = dedent("""\
    package server

    import (
    \t"fmt"
    \t"sync"
    )

    var (
    \tglobalCache map[string]int
    \tmu          sync.Mutex
    )

    var singleVar = 42

    const MaxRetries = 5

    type Config struct {
    \tName string
    \tMax  int
    }

    func init() {
    \tglobalCache = make(map[string]int)
    \tglobalCache["default"] = 0
    \tdefaultCfg := Config{Name: "prod", Max: MaxRetries}
    \t_ = defaultCfg
    \tmu.Lock()
    \tmu.Unlock()
    }

    func Process() int {
    \tfmt.Println("processing")
    \treturn singleVar + MaxRetries
    }
""")


# ---------------------------------------------------------------------------
# Helpers to drive the full extract → build pipeline
# ---------------------------------------------------------------------------


def _build_context_for_function(
    source: str,
    filename: str,
    function_name: str,
    tmp_path: Path,
    parents: list[FunctionParent] | None = None,
    is_method: bool = False,
) -> CodeOptimizationContext:
    root = tmp_path.resolve()
    source_file = (root / filename).resolve()
    source_file.write_text(source, encoding="utf-8")

    func = FunctionToOptimize(
        function_name=function_name, file_path=source_file, parents=parents or [], language="go", is_method=is_method
    )
    code_context = extract_code_context(func, root)
    return _build_optimization_context(code_context, source_file, "go", root)


# ---------------------------------------------------------------------------
# Tests: targeting a plain exported function
# ---------------------------------------------------------------------------


class TestBuildContextExportedFunction:
    """Target: Add(a, b int) int — a plain exported function with a doc comment."""

    def test_full_assembled_code_string(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        code = result.read_writable_code.code_strings[0].code

        expected = dedent("""\
            import (
            \t"fmt"
            \t"math"
            \tstr "strings"
            )

            // Add returns the sum of two integers.
            func Add(a, b int) int {
            \treturn a + b
            }
        """)
        assert code == expected

    def test_code_excludes_package_clause(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "package calc" not in code

    def test_code_excludes_struct_definition(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "type Calculator struct" not in code

    def test_code_excludes_interface_definition(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "type Formatter interface" not in code

    def test_no_helpers_when_no_calls(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.helper_functions == []

    def test_no_read_only_context_for_plain_function(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.read_only_context_code == ""

    def test_relative_path(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.read_writable_code.code_strings[0].file_path == Path("calc.go")

    def test_language_tag(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.read_writable_code.code_strings[0].language == "go"

    def test_testgen_fqns_match_helpers(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        fqns = set(result.testgen_helper_fqns)
        helper_fqns = {h.fully_qualified_name for h in result.helper_functions}
        assert fqns == helper_fqns


# ---------------------------------------------------------------------------
# Tests: targeting a method with a pointer receiver
# ---------------------------------------------------------------------------


class TestBuildContextPointerReceiverMethod:
    """Target: (c *Calculator) AddFloat(val float64) — pointer receiver method."""

    def _build(self, tmp_path: Path) -> CodeOptimizationContext:
        return _build_context_for_function(
            CALCULATOR_SOURCE,
            "calc.go",
            "AddFloat",
            tmp_path,
            parents=[FunctionParent(name="Calculator", type="StructDef")],
            is_method=True,
        )

    def test_full_assembled_code_string(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        code = result.read_writable_code.code_strings[0].code

        expected = dedent("""\
            import (
            \t"fmt"
            \t"math"
            \tstr "strings"
            )

            // AddFloat adds a float value and records history.
            func (c *Calculator) AddFloat(val float64) float64 {
            \tc.Result += val
            \tc.History = append(c.History, c.Result)
            \treturn c.Result
            }
        """)
        assert code == expected

    def test_code_excludes_package_and_type_defs(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "package calc" not in code
        assert "type Calculator struct" not in code
        assert "type Formatter interface" not in code

    def test_read_only_context_is_struct_definition(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        assert result.read_only_context_code == dedent("""\
            type Calculator struct {
            \tResult  float64
            \tHistory []float64
            }""")

    def test_no_helpers_when_no_calls(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        assert result.helper_functions == []

    def test_target_not_duplicated_in_code_string(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert code.count("func (c *Calculator) AddFloat") == 1


# ---------------------------------------------------------------------------
# Tests: targeting a value receiver method
# ---------------------------------------------------------------------------


class TestBuildContextValueReceiverMethod:
    """Target: (c Calculator) Reset() — value receiver method."""

    def _build(self, tmp_path: Path) -> CodeOptimizationContext:
        return _build_context_for_function(
            CALCULATOR_SOURCE,
            "calc.go",
            "Reset",
            tmp_path,
            parents=[FunctionParent(name="Calculator", type="StructDef")],
            is_method=True,
        )

    def test_target_in_code_string(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        code = result.read_writable_code.code_strings[0].code

        expected_target = dedent("""\
            // Reset zeroes out the calculator.
            func (c Calculator) Reset() Calculator {
            \tc.Result = 0
            \tc.History = nil
            \treturn c
            }""")
        assert code.count("func (c Calculator) Reset()") == 1
        assert expected_target in code

    def test_no_helpers_when_no_calls(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        assert result.helper_functions == []

    def test_no_helper_code_in_assembled_string(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "func (c *Calculator) AddFloat" not in code
        assert "func Add(a, b int) int" not in code

    def test_struct_in_read_only_context(self, tmp_path: Path) -> None:
        result = self._build(tmp_path)
        assert result.read_only_context_code == dedent("""\
            type Calculator struct {
            \tResult  float64
            \tHistory []float64
            }""")


# ---------------------------------------------------------------------------
# Tests: simple source with no imports, no methods, one function
# ---------------------------------------------------------------------------


class TestBuildContextMinimalSource:
    """Target: Double(x int) — minimal file with no imports or structs."""

    def test_no_imports_no_prefix(self, tmp_path: Path) -> None:
        result = _build_context_for_function(SIMPLE_SOURCE, "simple.go", "Double", tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert code == dedent("""\
            func Double(x int) int {
            \treturn x * 2
            }""")

    def test_no_helpers(self, tmp_path: Path) -> None:
        result = _build_context_for_function(SIMPLE_SOURCE, "simple.go", "Double", tmp_path)
        assert result.helper_functions == []
        assert result.testgen_helper_fqns == []

    def test_empty_read_only_context(self, tmp_path: Path) -> None:
        result = _build_context_for_function(SIMPLE_SOURCE, "simple.go", "Double", tmp_path)
        assert result.read_only_context_code == ""

    def test_preexisting_objects_empty(self, tmp_path: Path) -> None:
        result = _build_context_for_function(SIMPLE_SOURCE, "simple.go", "Double", tmp_path)
        assert result.preexisting_objects == set()


# ---------------------------------------------------------------------------
# Tests: init function and globals in context
# ---------------------------------------------------------------------------


class TestBuildContextWithInit:
    """Target: Process() — source has init(), global vars, consts, struct."""

    def test_init_in_read_only_context(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        assert "func init()" in result.read_only_context_code

    def test_referenced_globals_in_read_only_context(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        assert "globalCache" in result.read_only_context_code
        assert "mu" in result.read_only_context_code

    def test_referenced_const_in_read_only_context(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        assert "MaxRetries" in result.read_only_context_code

    def test_referenced_struct_in_read_only_context(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        assert "type Config struct" in result.read_only_context_code

    def test_init_not_in_helpers(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        helper_names = [h.only_function_name for h in result.helper_functions]
        assert "init" not in helper_names

    def test_init_not_in_read_writable_code(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        code = result.read_writable_code.code_strings[0].code
        assert "func init()" not in code

    def test_full_read_only_context_string(self, tmp_path: Path) -> None:
        result = _build_context_for_function(INIT_SOURCE, "server.go", "Process", tmp_path)
        expected = dedent("""\
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
            \tdefaultCfg := Config{Name: "prod", Max: MaxRetries}
            \t_ = defaultCfg
            \tmu.Lock()
            \tmu.Unlock()
            }""")
        assert result.read_only_context_code == expected


class TestBuildContextNoInit:
    """Source without init — verify no init context is added."""

    def test_no_init_no_extra_read_only(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert "func init()" not in result.read_only_context_code

    def test_no_init_read_only_empty_for_function(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.read_only_context_code == ""


# ---------------------------------------------------------------------------
# Tests: subdirectory / relative path handling
# ---------------------------------------------------------------------------


class TestBuildContextSubdirectory:
    """Source file in a pkg/ subdirectory."""

    def test_relative_path_includes_subdir(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        pkg = root / "pkg"
        pkg.mkdir()
        source_file = (pkg / "calc.go").resolve()
        source_file.write_text(SIMPLE_SOURCE, encoding="utf-8")

        func = FunctionToOptimize(function_name="Double", file_path=source_file, language="go")
        ctx = extract_code_context(func, root)
        result = _build_optimization_context(ctx, source_file, "go", root)

        assert result.read_writable_code.code_strings[0].file_path == Path("pkg/calc.go")


# ---------------------------------------------------------------------------
# Tests: hashing
# ---------------------------------------------------------------------------


class TestBuildContextHashing:
    def test_hash_is_sha256_of_flat(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        expected_hash = hashlib.sha256(result.read_writable_code.flat.encode("utf-8")).hexdigest()
        assert result.hashing_code_context_hash == expected_hash

    def test_hashing_code_equals_flat(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.hashing_code_context == result.read_writable_code.flat

    def test_different_targets_different_hashes(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()

        r1 = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", dir_a)
        r2 = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Greet", dir_b)

        assert r1.hashing_code_context_hash != r2.hashing_code_context_hash


# ---------------------------------------------------------------------------
# Tests: testgen context
# ---------------------------------------------------------------------------


class TestBuildContextTestgen:
    def test_testgen_matches_read_writable(self, tmp_path: Path) -> None:
        result = _build_context_for_function(CALCULATOR_SOURCE, "calc.go", "Add", tmp_path)
        assert result.testgen_context.markdown == result.read_writable_code.markdown


# ---------------------------------------------------------------------------
# Tests: token limit enforcement
# ---------------------------------------------------------------------------


class TestBuildContextTokenLimits:
    def test_exceeds_optim_token_limit(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        source_file = (root / "big.go").resolve()
        huge_code = "package big\n\nfunc Big() string {\n\treturn " + '"x" + ' * 100000 + '"x"\n}\n'
        source_file.write_text(huge_code, encoding="utf-8")

        func = FunctionToOptimize(function_name="Big", file_path=source_file, language="go")
        ctx = extract_code_context(func, root)

        with pytest.raises(ValueError, match="Read-writable code has exceeded token limit"):
            _build_optimization_context(ctx, source_file, "go", root, optim_token_limit=10)

    def test_exceeds_testgen_token_limit(self, tmp_path: Path) -> None:
        root = tmp_path.resolve()
        source_file = (root / "big.go").resolve()
        huge_code = "package big\n\nfunc Big() string {\n\treturn " + '"x" + ' * 100000 + '"x"\n}\n'
        source_file.write_text(huge_code, encoding="utf-8")

        func = FunctionToOptimize(function_name="Big", file_path=source_file, language="go")
        ctx = extract_code_context(func, root)

        with pytest.raises(ValueError, match="Testgen code context has exceeded token limit"):
            _build_optimization_context(
                ctx, source_file, "go", root, optim_token_limit=1_000_000, testgen_token_limit=10
            )


# ---------------------------------------------------------------------------
# Tests: GoSupport wiring
# ---------------------------------------------------------------------------


class TestGoSupportFunctionOptimizerClass:
    def test_returns_go_function_optimizer(self) -> None:
        from codeflash.languages.golang.function_optimizer import GoFunctionOptimizer
        from codeflash.languages.golang.support import GoSupport

        support = GoSupport()
        assert support.function_optimizer_class is GoFunctionOptimizer
