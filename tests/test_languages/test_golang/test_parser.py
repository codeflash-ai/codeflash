from __future__ import annotations

from codeflash.languages.golang.parser import GoAnalyzer

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

func noReturn() {
	println("hello")
}

type Calculator struct {
	Result float64
}

// AddFloat adds a value.
func (c *Calculator) AddFloat(val float64) float64 {
	c.Result += val
	return c.Result
}

func (c Calculator) GetResult() float64 {
	return c.Result
}

// Reset zeroes the calculator.
func (c *Calculator) Reset() {
	c.Result = 0
}

type Adder interface {
	Add(a, b int) int
}
"""


class TestGoAnalyzerFunctions:
    def test_find_functions(self) -> None:
        analyzer = GoAnalyzer()
        functions = analyzer.find_functions(GO_SOURCE)
        names = [f.name for f in functions]
        assert "Add" in names
        assert "subtract" in names
        assert "noReturn" in names

    def test_exported_detection(self) -> None:
        analyzer = GoAnalyzer()
        functions = analyzer.find_functions(GO_SOURCE)
        by_name = {f.name: f for f in functions}
        assert by_name["Add"].is_exported is True
        assert by_name["subtract"].is_exported is False

    def test_return_type_detection(self) -> None:
        analyzer = GoAnalyzer()
        functions = analyzer.find_functions(GO_SOURCE)
        by_name = {f.name: f for f in functions}
        assert by_name["Add"].has_return_type is True
        assert by_name["noReturn"].has_return_type is False

    def test_doc_comment_detection(self) -> None:
        analyzer = GoAnalyzer()
        functions = analyzer.find_functions(GO_SOURCE)
        by_name = {f.name: f for f in functions}
        assert by_name["Add"].doc_start_line is not None
        assert by_name["subtract"].doc_start_line is None

    def test_line_numbers(self) -> None:
        analyzer = GoAnalyzer()
        functions = analyzer.find_functions(GO_SOURCE)
        by_name = {f.name: f for f in functions}
        add_func = by_name["Add"]
        assert add_func.starting_line == 6
        assert add_func.ending_line == 8


class TestGoAnalyzerMethods:
    def test_find_methods(self) -> None:
        analyzer = GoAnalyzer()
        methods = analyzer.find_methods(GO_SOURCE)
        names = [m.name for m in methods]
        assert "AddFloat" in names
        assert "GetResult" in names
        assert "Reset" in names

    def test_receiver_detection(self) -> None:
        analyzer = GoAnalyzer()
        methods = analyzer.find_methods(GO_SOURCE)
        by_name = {m.name: m for m in methods}
        assert by_name["AddFloat"].receiver_name == "Calculator"
        assert by_name["AddFloat"].receiver_is_pointer is True
        assert by_name["GetResult"].receiver_name == "Calculator"
        assert by_name["GetResult"].receiver_is_pointer is False

    def test_method_doc_comment(self) -> None:
        analyzer = GoAnalyzer()
        methods = analyzer.find_methods(GO_SOURCE)
        by_name = {m.name: m for m in methods}
        assert by_name["AddFloat"].doc_start_line is not None
        assert by_name["Reset"].doc_start_line is not None
        assert by_name["GetResult"].doc_start_line is None

    def test_method_exported(self) -> None:
        analyzer = GoAnalyzer()
        methods = analyzer.find_methods(GO_SOURCE)
        for m in methods:
            assert m.is_exported is True


class TestGoAnalyzerStructs:
    def test_find_structs(self) -> None:
        analyzer = GoAnalyzer()
        structs = analyzer.find_structs(GO_SOURCE)
        assert len(structs) == 1
        assert structs[0].name == "Calculator"
        assert len(structs[0].fields) > 0

    def test_struct_field_content(self) -> None:
        analyzer = GoAnalyzer()
        structs = analyzer.find_structs(GO_SOURCE)
        field_text = " ".join(structs[0].fields)
        assert "Result" in field_text
        assert "float64" in field_text


class TestGoAnalyzerInterfaces:
    def test_find_interfaces(self) -> None:
        analyzer = GoAnalyzer()
        interfaces = analyzer.find_interfaces(GO_SOURCE)
        assert len(interfaces) == 1
        assert interfaces[0].name == "Adder"
        assert len(interfaces[0].methods) > 0


class TestGoAnalyzerImports:
    def test_find_imports(self) -> None:
        analyzer = GoAnalyzer()
        imports = analyzer.find_imports(GO_SOURCE)
        assert len(imports) == 1
        assert imports[0].path == "math"
        assert imports[0].alias is None

    def test_multi_import(self) -> None:
        source = '''\
package main

import (
	"fmt"
	"os"
	str "strings"
)

func Main() string {
	return "hello"
}
'''
        analyzer = GoAnalyzer()
        imports = analyzer.find_imports(source)
        paths = {i.path for i in imports}
        assert paths == {"fmt", "os", "strings"}
        aliases = {i.path: i.alias for i in imports}
        assert aliases["strings"] == "str"
        assert aliases["fmt"] is None


class TestGoAnalyzerPackage:
    def test_find_package_name(self) -> None:
        analyzer = GoAnalyzer()
        assert analyzer.find_package_name(GO_SOURCE) == "calculator"

    def test_find_package_name_main(self) -> None:
        analyzer = GoAnalyzer()
        assert analyzer.find_package_name("package main\n\nfunc main() {}") == "main"


class TestGoAnalyzerSyntax:
    def test_valid_syntax(self) -> None:
        analyzer = GoAnalyzer()
        assert analyzer.validate_syntax(GO_SOURCE) is True

    def test_invalid_syntax(self) -> None:
        analyzer = GoAnalyzer()
        assert analyzer.validate_syntax("func {{{invalid") is False


class TestGoAnalyzerExtract:
    def test_extract_function_source(self) -> None:
        analyzer = GoAnalyzer()
        source = analyzer.extract_function_source(GO_SOURCE, "Add")
        assert source is not None
        assert "func Add" in source
        assert "return a + b" in source

    def test_extract_function_source_with_doc(self) -> None:
        analyzer = GoAnalyzer()
        source = analyzer.extract_function_source(GO_SOURCE, "Add")
        assert source is not None
        assert "// Add returns" in source

    def test_extract_method_source(self) -> None:
        analyzer = GoAnalyzer()
        source = analyzer.extract_function_source(GO_SOURCE, "AddFloat", receiver_type="Calculator")
        assert source is not None
        assert "func (c *Calculator) AddFloat" in source

    def test_extract_nonexistent(self) -> None:
        analyzer = GoAnalyzer()
        assert analyzer.extract_function_source(GO_SOURCE, "DoesNotExist") is None


GLOBALS_SOURCE = """\
package server

import "sync"

var (
\tglobalCache map[string]int
\tmu          sync.Mutex
)

var singleVar = 42

const MaxRetries = 5

const (
\tDefaultName = "prod"
\tTimeout     = 30
)

type Config struct {
\tName string
\tMax  int
}

func init() {
\tglobalCache = make(map[string]int)
\tglobalCache["default"] = 0
\tdefaultCfg := Config{Name: DefaultName, Max: MaxRetries}
\t_ = defaultCfg
\tmu.Lock()
\tmu.Unlock()
}

func Process() int {
\treturn singleVar + MaxRetries
}
"""


class TestGoAnalyzerGlobalDeclarations:
    def test_find_var_group(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations(GLOBALS_SOURCE)
        var_decls = [d for d in decls if d.kind == "var"]
        all_names = [name for d in var_decls for name in d.names]
        assert "globalCache" in all_names
        assert "mu" in all_names
        assert "singleVar" in all_names

    def test_find_const_group(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations(GLOBALS_SOURCE)
        const_decls = [d for d in decls if d.kind == "const"]
        all_names = [name for d in const_decls for name in d.names]
        assert "MaxRetries" in all_names
        assert "DefaultName" in all_names
        assert "Timeout" in all_names

    def test_grouped_var_names_together(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations(GLOBALS_SOURCE)
        var_group = next(d for d in decls if "globalCache" in d.names)
        assert var_group.names == ("globalCache", "mu")

    def test_single_var(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations(GLOBALS_SOURCE)
        single = next(d for d in decls if "singleVar" in d.names)
        assert single.kind == "var"
        assert single.source_code == "var singleVar = 42"

    def test_const_group_source_code(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations(GLOBALS_SOURCE)
        group = next(d for d in decls if "DefaultName" in d.names)
        assert "DefaultName" in group.source_code
        assert "Timeout" in group.source_code

    def test_no_globals_in_clean_source(self) -> None:
        analyzer = GoAnalyzer()
        decls = analyzer.find_global_declarations("package main\n\nfunc main() {}\n")
        assert decls == []


class TestGoAnalyzerCollectBodyIdentifiers:
    def test_init_body_identifiers(self) -> None:
        analyzer = GoAnalyzer()
        ids = analyzer.collect_body_identifiers(GLOBALS_SOURCE, "init")
        assert "globalCache" in ids
        assert "Config" in ids
        assert "DefaultName" in ids
        assert "MaxRetries" in ids
        assert "mu" in ids

    def test_process_body_identifiers(self) -> None:
        analyzer = GoAnalyzer()
        ids = analyzer.collect_body_identifiers(GLOBALS_SOURCE, "Process")
        assert "singleVar" in ids
        assert "MaxRetries" in ids

    def test_nonexistent_function_returns_empty(self) -> None:
        analyzer = GoAnalyzer()
        ids = analyzer.collect_body_identifiers(GLOBALS_SOURCE, "DoesNotExist")
        assert ids == set()

    def test_method_body_identifiers(self) -> None:
        source = """\
package calc

type Calc struct{ val int }

var offset = 10

func (c *Calc) Compute() int {
\treturn c.val + offset
}
"""
        analyzer = GoAnalyzer()
        ids = analyzer.collect_body_identifiers(source, "Compute", receiver_type="Calc")
        assert "offset" in ids
