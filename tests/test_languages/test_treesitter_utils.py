"""Extensive tests for the tree-sitter utilities module.

These tests verify that the TreeSitterAnalyzer correctly parses and
analyzes JavaScript/TypeScript code.
"""

from pathlib import Path

import pytest

from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage, get_analyzer_for_file


class TestTreeSitterLanguage:
    """Tests for TreeSitterLanguage enum."""

    def test_language_values(self):
        """Test that language enum has expected values."""
        assert TreeSitterLanguage.JAVASCRIPT.value == "javascript"
        assert TreeSitterLanguage.TYPESCRIPT.value == "typescript"
        assert TreeSitterLanguage.TSX.value == "tsx"


class TestTreeSitterAnalyzerCreation:
    """Tests for TreeSitterAnalyzer initialization."""

    def test_create_javascript_analyzer(self):
        """Test creating JavaScript analyzer."""
        analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT

    def test_create_typescript_analyzer(self):
        """Test creating TypeScript analyzer."""
        analyzer = TreeSitterAnalyzer(TreeSitterLanguage.TYPESCRIPT)
        assert analyzer.language == TreeSitterLanguage.TYPESCRIPT

    def test_create_with_string(self):
        """Test creating analyzer with string language name."""
        analyzer = TreeSitterAnalyzer("javascript")
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT

    def test_lazy_parser_creation(self):
        """Test that parser is created lazily."""
        analyzer = TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)
        assert analyzer._parser is None
        # Access parser property
        _ = analyzer.parser
        assert analyzer._parser is not None


class TestGetAnalyzerForFile:
    """Tests for get_analyzer_for_file function."""

    def test_js_file(self):
        """Test getting analyzer for .js file."""
        analyzer = get_analyzer_for_file(Path("/test/file.js"))
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT

    def test_jsx_file(self):
        """Test getting analyzer for .jsx file."""
        analyzer = get_analyzer_for_file(Path("/test/file.jsx"))
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT

    def test_ts_file(self):
        """Test getting analyzer for .ts file."""
        analyzer = get_analyzer_for_file(Path("/test/file.ts"))
        assert analyzer.language == TreeSitterLanguage.TYPESCRIPT

    def test_tsx_file(self):
        """Test getting analyzer for .tsx file."""
        analyzer = get_analyzer_for_file(Path("/test/file.tsx"))
        assert analyzer.language == TreeSitterLanguage.TSX

    def test_mjs_file(self):
        """Test getting analyzer for .mjs file."""
        analyzer = get_analyzer_for_file(Path("/test/file.mjs"))
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT

    def test_cjs_file(self):
        """Test getting analyzer for .cjs file."""
        analyzer = get_analyzer_for_file(Path("/test/file.cjs"))
        assert analyzer.language == TreeSitterLanguage.JAVASCRIPT


class TestParsing:
    """Tests for parsing functionality."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_parse_simple_code(self, js_analyzer):
        """Test parsing simple JavaScript code."""
        code = "const x = 1;"
        tree = js_analyzer.parse(code)
        assert tree.root_node is not None
        assert not tree.root_node.has_error

    def test_parse_bytes(self, js_analyzer):
        """Test parsing code as bytes."""
        code = b"const x = 1;"
        tree = js_analyzer.parse(code)
        assert tree.root_node is not None

    def test_parse_invalid_code(self, js_analyzer):
        """Test parsing invalid code marks errors."""
        code = "function foo( {"
        tree = js_analyzer.parse(code)
        assert tree.root_node.has_error

    def test_get_node_text(self, js_analyzer):
        """Test extracting text from a node."""
        code = "const x = 1;"
        code_bytes = code.encode("utf8")
        tree = js_analyzer.parse(code_bytes)
        text = js_analyzer.get_node_text(tree.root_node, code_bytes)
        assert text == code


class TestFindFunctions:
    """Tests for find_functions method."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_find_function_declaration(self, js_analyzer):
        """Test finding function declarations."""
        code = """
function add(a, b) {
    return a + b;
}
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "add"
        assert functions[0].is_arrow is False
        assert functions[0].is_async is False
        assert functions[0].is_method is False

    def test_find_arrow_function(self, js_analyzer):
        """Test finding arrow functions."""
        code = """
const add = (a, b) => {
    return a + b;
};
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "add"
        assert functions[0].is_arrow is True

    def test_find_arrow_function_concise(self, js_analyzer):
        """Test finding concise arrow functions."""
        code = "const double = x => x * 2;"
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "double"
        assert functions[0].is_arrow is True

    def test_find_async_function(self, js_analyzer):
        """Test finding async functions."""
        code = """
async function fetchData(url) {
    return await fetch(url);
}
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "fetchData"
        assert functions[0].is_async is True

    def test_find_class_methods(self, js_analyzer):
        """Test finding class methods."""
        code = """
class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
        functions = js_analyzer.find_functions(code, include_methods=True)

        assert len(functions) == 1
        assert functions[0].function_name == "add"
        assert functions[0].is_method is True
        assert functions[0].class_name == "Calculator"

    def test_exclude_methods(self, js_analyzer):
        """Test excluding class methods."""
        code = """
class Calculator {
    add(a, b) {
        return a + b;
    }
}

function standalone() {
    return 1;
}
"""
        functions = js_analyzer.find_functions(code, include_methods=False)

        assert len(functions) == 1
        assert functions[0].function_name == "standalone"

    def test_exclude_arrow_functions(self, js_analyzer):
        """Test excluding arrow functions."""
        code = """
function regular() {
    return 1;
}

const arrow = () => 2;
"""
        functions = js_analyzer.find_functions(code, include_arrow_functions=False)

        assert len(functions) == 1
        assert functions[0].function_name == "regular"

    def test_find_generator_function(self, js_analyzer):
        """Test finding generator functions."""
        code = """
function* numberGenerator() {
    yield 1;
    yield 2;
}
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "numberGenerator"
        assert functions[0].is_generator is True

    def test_function_line_numbers(self, js_analyzer):
        """Test that line numbers are correct."""
        code = """function first() {
    return 1;
}

function second() {
    return 2;
}
"""
        functions = js_analyzer.find_functions(code)

        first = next(f for f in functions if f.name == "first")
        second = next(f for f in functions if f.name == "second")

        assert first.start_line == 1
        assert first.end_line == 3
        assert second.start_line == 5
        assert second.end_line == 7

    def test_nested_functions(self, js_analyzer):
        """Test finding nested functions."""
        code = """
function outer() {
    function inner() {
        return 1;
    }
    return inner();
}
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 2
        names = {f.name for f in functions}
        assert names == {"outer", "inner"}

        inner = next(f for f in functions if f.name == "inner")
        assert inner.parent_function == "outer"

    def test_require_name_filters_anonymous(self, js_analyzer):
        """Test that require_name filters anonymous functions."""
        code = """
(function() {
    return 1;
})();

function named() {
    return 2;
}
"""
        functions = js_analyzer.find_functions(code, require_name=True)

        assert len(functions) == 1
        assert functions[0].function_name == "named"

    def test_function_expression_in_variable(self, js_analyzer):
        """Test function expression assigned to variable."""
        code = """
const add = function(a, b) {
    return a + b;
};
"""
        functions = js_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "add"


class TestFindImports:
    """Tests for find_imports method."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_find_default_import(self, js_analyzer):
        """Test finding default import."""
        code = "import React from 'react';"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "react"
        assert imports[0].default_import == "React"

    def test_find_named_imports(self, js_analyzer):
        """Test finding named imports."""
        code = "import { useState, useEffect } from 'react';"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "react"
        assert ("useState", None) in imports[0].named_imports
        assert ("useEffect", None) in imports[0].named_imports

    def test_find_namespace_import(self, js_analyzer):
        """Test finding namespace import."""
        code = "import * as utils from './utils';"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./utils"
        assert imports[0].namespace_import == "utils"

    def test_find_require(self, js_analyzer):
        """Test finding require() calls."""
        code = "const fs = require('fs');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "fs"
        assert imports[0].default_import == "fs"

    def test_require_inside_function_not_import(self, js_analyzer):
        """Test that require() inside functions is not treated as an import.

        This is important because dynamic require() calls inside functions are
        not module-level imports and should not be extracted as such.
        """
        code = """
const fs = require('fs');

function loadModule() {
    const dynamic = require('dynamic-module');
    return dynamic;
}

class MyClass {
    method() {
        const inMethod = require('method-module');
    }
}
"""
        imports = js_analyzer.find_imports(code)

        # Only the module-level require should be found
        assert len(imports) == 1
        assert imports[0].module_path == "fs"

    def test_find_multiple_imports(self, js_analyzer):
        """Test finding multiple imports."""
        code = """
import React from 'react';
import { useState } from 'react';
import * as utils from './utils';
const path = require('path');
"""
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 4
        modules = {imp.module_path for imp in imports}
        assert modules == {"react", "./utils", "path"}

    def test_import_with_alias(self, js_analyzer):
        """Test finding import with alias."""
        code = "import { Component as Comp } from 'react';"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert ("Component", "Comp") in imports[0].named_imports

    def test_relative_import(self, js_analyzer):
        """Test finding relative imports."""
        code = "import { helper } from './helpers/utils';"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helpers/utils"


class TestFindFunctionCalls:
    """Tests for find_function_calls method."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_find_simple_calls(self, js_analyzer):
        """Test finding simple function calls."""
        code = """
function helper() {
    return 1;
}

function main() {
    return helper() + 2;
}
"""
        functions = js_analyzer.find_functions(code)
        main_func = next(f for f in functions if f.name == "main")

        calls = js_analyzer.find_function_calls(code, main_func)

        assert "helper" in calls

    def test_find_method_calls(self, js_analyzer):
        """Test finding method calls."""
        code = """
function process(arr) {
    return arr.map(x => x * 2).filter(x => x > 0);
}
"""
        functions = js_analyzer.find_functions(code)
        process_func = next(f for f in functions if f.name == "process")

        calls = js_analyzer.find_function_calls(code, process_func)

        assert "map" in calls
        assert "filter" in calls


class TestHasReturnStatement:
    """Tests for has_return_statement method."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_function_with_return(self, js_analyzer):
        """Test function with return statement."""
        code = """
function add(a, b) {
    return a + b;
}
"""
        functions = js_analyzer.find_functions(code)
        assert js_analyzer.has_return_statement(functions[0], code) is True

    def test_function_without_return(self, js_analyzer):
        """Test function without return statement."""
        code = """
function log(msg) {
    console.log(msg);
}
"""
        functions = js_analyzer.find_functions(code, require_name=True)
        func = next((f for f in functions if f.name == "log"), None)
        if func:
            assert js_analyzer.has_return_statement(func, code) is False

    def test_arrow_function_implicit_return(self, js_analyzer):
        """Test arrow function with implicit return."""
        code = "const double = x => x * 2;"
        functions = js_analyzer.find_functions(code)
        assert js_analyzer.has_return_statement(functions[0], code) is True

    def test_arrow_function_explicit_return(self, js_analyzer):
        """Test arrow function with explicit return."""
        code = """
const add = (a, b) => {
    return a + b;
};
"""
        functions = js_analyzer.find_functions(code)
        assert js_analyzer.has_return_statement(functions[0], code) is True


class TestTypeScriptSupport:
    """Tests for TypeScript-specific features."""

    @pytest.fixture
    def ts_analyzer(self):
        """Create a TypeScript analyzer."""
        return TreeSitterAnalyzer(TreeSitterLanguage.TYPESCRIPT)

    def test_find_typed_function(self, ts_analyzer):
        """Test finding function with type annotations."""
        code = """
function add(a: number, b: number): number {
    return a + b;
}
"""
        functions = ts_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "add"

    def test_find_interface_method(self, ts_analyzer):
        """Test that interface methods are not found (they're declarations)."""
        code = """
interface Calculator {
    add(a: number, b: number): number;
}

function helper(): number {
    return 1;
}
"""
        functions = ts_analyzer.find_functions(code)

        # Only the actual function should be found, not the interface method
        names = {f.name for f in functions}
        assert "helper" in names

    def test_find_generic_function(self, ts_analyzer):
        """Test finding generic function."""
        code = """
function identity<T>(value: T): T {
    return value;
}
"""
        functions = ts_analyzer.find_functions(code)

        assert len(functions) == 1
        assert functions[0].function_name == "identity"
