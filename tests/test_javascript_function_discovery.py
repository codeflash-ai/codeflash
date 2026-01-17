"""
Tests for JavaScript function discovery in get_functions_to_optimize.

These tests verify that JavaScript functions are correctly discovered,
filtered, and returned from the function discovery pipeline.
"""

import tempfile
import unittest.mock
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import (
    FunctionToOptimize,
    filter_functions,
    find_all_functions_in_file,
    get_all_files_and_functions,
    get_functions_to_optimize,
)
from codeflash.languages.base import Language
from codeflash.verification.verification_utils import TestConfig


class TestJavaScriptFunctionDiscovery:
    """Tests for discovering functions in JavaScript files."""

    def test_simple_function_discovery(self, tmp_path):
        """Test discovering a simple JavaScript function with return statement."""
        js_file = tmp_path / "simple.js"
        js_file.write_text("""
function add(a, b) {
    return a + b;
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 1
        fn = functions[js_file][0]
        assert fn.function_name == "add"
        assert fn.language == "javascript"
        assert fn.file_path == js_file

    def test_multiple_functions_discovery(self, tmp_path):
        """Test discovering multiple JavaScript functions."""
        js_file = tmp_path / "multiple.js"
        js_file.write_text("""
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

function divide(a, b) {
    return a / b;
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 3
        names = {fn.function_name for fn in functions[js_file]}
        assert names == {"add", "multiply", "divide"}

    def test_function_without_return_excluded(self, tmp_path):
        """Test that functions without return statements are excluded."""
        js_file = tmp_path / "no_return.js"
        js_file.write_text("""
function withReturn() {
    return 42;
}

function withoutReturn() {
    console.log("hello");
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 1
        assert functions[js_file][0].function_name == "withReturn"

    def test_arrow_function_discovery(self, tmp_path):
        """Test discovering arrow functions with explicit return."""
        js_file = tmp_path / "arrow.js"
        js_file.write_text("""
const add = (a, b) => {
    return a + b;
};

const multiply = (a, b) => a * b;
""")
        functions = find_all_functions_in_file(js_file)

        # Arrow functions should be discovered
        assert len(functions.get(js_file, [])) >= 1
        names = {fn.function_name for fn in functions[js_file]}
        assert "add" in names

    def test_class_method_discovery(self, tmp_path):
        """Test discovering methods inside a JavaScript class."""
        js_file = tmp_path / "class.js"
        js_file.write_text("""
class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 2
        names = {fn.function_name for fn in functions[js_file]}
        assert names == {"add", "multiply"}

        # Check that methods have correct parent
        for fn in functions[js_file]:
            assert len(fn.parents) == 1
            assert fn.parents[0].name == "Calculator"

    def test_async_function_discovery(self, tmp_path):
        """Test discovering async JavaScript functions."""
        js_file = tmp_path / "async.js"
        js_file.write_text("""
async function fetchData(url) {
    return await fetch(url);
}

function syncFunc() {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 2
        async_fn = next(fn for fn in functions[js_file] if fn.function_name == "fetchData")
        sync_fn = next(fn for fn in functions[js_file] if fn.function_name == "syncFunc")

        assert async_fn.is_async is True
        assert sync_fn.is_async is False

    def test_nested_function_excluded(self, tmp_path):
        """Test that nested functions are handled correctly."""
        js_file = tmp_path / "nested.js"
        js_file.write_text("""
function outer() {
    function inner() {
        return 1;
    }
    return inner();
}
""")
        functions = find_all_functions_in_file(js_file)

        # Both outer and inner should be found (inner has a return)
        names = {fn.function_name for fn in functions.get(js_file, [])}
        assert "outer" in names

    def test_jsx_file_discovery(self, tmp_path):
        """Test discovering functions in JSX files."""
        jsx_file = tmp_path / "component.jsx"
        jsx_file.write_text("""
function Button({ onClick }) {
    return <button onClick={onClick}>Click me</button>;
}

function formatText(text) {
    return text.toUpperCase();
}
""")
        functions = find_all_functions_in_file(jsx_file)

        assert len(functions.get(jsx_file, [])) >= 1
        names = {fn.function_name for fn in functions[jsx_file]}
        assert "formatText" in names

    def test_invalid_javascript_returns_empty(self, tmp_path):
        """Test that invalid JavaScript code returns empty results."""
        js_file = tmp_path / "invalid.js"
        js_file.write_text("""
function broken( {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)

        # Should return empty dict or empty list for the file
        assert len(functions.get(js_file, [])) == 0

    def test_function_line_numbers(self, tmp_path):
        """Test that function line numbers are correctly detected."""
        js_file = tmp_path / "lines.js"
        js_file.write_text("""
function firstFunc() {
    return 1;
}

function secondFunc() {
    return 2;
}
""")
        functions = find_all_functions_in_file(js_file)

        assert len(functions.get(js_file, [])) == 2
        first_fn = next(fn for fn in functions[js_file] if fn.function_name == "firstFunc")
        second_fn = next(fn for fn in functions[js_file] if fn.function_name == "secondFunc")

        assert first_fn.starting_line is not None
        assert first_fn.ending_line is not None
        assert second_fn.starting_line is not None
        assert second_fn.ending_line is not None
        assert first_fn.starting_line < second_fn.starting_line


class TestJavaScriptFunctionFiltering:
    """Tests for filtering JavaScript functions."""

    def test_filter_functions_includes_javascript(self, tmp_path):
        """Test that filter_functions correctly includes JavaScript files."""
        js_file = tmp_path / "module.js"
        js_file.write_text("""
function add(a, b) {
    return a + b;
}
""")
        functions = find_all_functions_in_file(js_file)

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                functions,
                tests_root=tmp_path / "tests",
                ignore_paths=[],
                project_root=tmp_path,
                module_root=tmp_path,
            )

        assert js_file in filtered
        assert count == 1
        assert filtered[js_file][0].function_name == "add"

    def test_filter_excludes_test_directory(self, tmp_path):
        """Test that JavaScript files in test directories are excluded."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_module.test.js"
        test_file.write_text("""
function testHelper() {
    return 42;
}
""")
        functions = find_all_functions_in_file(test_file)
        modified_functions = {test_file: functions.get(test_file, [])}

        filtered, count = filter_functions(
            modified_functions,
            tests_root=tests_dir,
            ignore_paths=[],
            project_root=tmp_path,
            module_root=tmp_path,
        )

        assert test_file not in filtered
        assert count == 0

    def test_filter_excludes_ignored_paths(self, tmp_path):
        """Test that JavaScript files in ignored paths are excluded."""
        ignored_dir = tmp_path / "ignored"
        ignored_dir.mkdir()
        js_file = ignored_dir / "ignored_module.js"
        js_file.write_text("""
function ignoredFunc() {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)
        modified_functions = {js_file: functions.get(js_file, [])}

        filtered, count = filter_functions(
            modified_functions,
            tests_root=tmp_path / "tests",
            ignore_paths=[ignored_dir],
            project_root=tmp_path,
            module_root=tmp_path,
        )

        assert js_file not in filtered
        assert count == 0

    def test_filter_includes_files_with_dashes(self, tmp_path):
        """Test that JavaScript files with dashes in name are included (unlike Python)."""
        js_file = tmp_path / "my-module.js"
        js_file.write_text("""
function myFunc() {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)
        modified_functions = {js_file: functions.get(js_file, [])}

        with unittest.mock.patch(
            "codeflash.discovery.functions_to_optimize.get_blocklisted_functions", return_value={}
        ):
            filtered, count = filter_functions(
                modified_functions,
                tests_root=tmp_path / "tests",
                ignore_paths=[],
                project_root=tmp_path,
                module_root=tmp_path,
            )

        # JavaScript files with dashes should be allowed
        assert js_file in filtered
        assert count == 1


class TestGetFunctionsToOptimizeJavaScript:
    """Tests for get_functions_to_optimize with JavaScript files."""

    def test_get_functions_from_file(self, tmp_path):
        """Test getting functions to optimize from a JavaScript file."""
        js_file = tmp_path / "string_utils.js"
        js_file.write_text("""
function reverseString(str) {
    return str.split('').reverse().join('');
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
""")
        test_config = TestConfig(
            tests_root=str(tmp_path / "tests"),
            project_root_path=str(tmp_path),
            test_framework="jest",
            tests_project_rootdir=tmp_path / "tests",
        )

        functions, count, trace_file = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=js_file,
            only_get_this_function=None,
            test_cfg=test_config,
            ignore_paths=[],
            project_root=tmp_path,
            module_root=tmp_path,
        )

        assert count == 2
        assert js_file in functions
        names = {fn.function_name for fn in functions[js_file]}
        assert names == {"reverseString", "capitalize"}

    def test_get_specific_function(self, tmp_path):
        """Test getting a specific function by name."""
        js_file = tmp_path / "math_utils.js"
        js_file.write_text("""
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}
""")
        test_config = TestConfig(
            tests_root=str(tmp_path / "tests"),
            project_root_path=str(tmp_path),
            test_framework="jest",
            tests_project_rootdir=tmp_path / "tests",
        )

        functions, count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=js_file,
            only_get_this_function="add",
            test_cfg=test_config,
            ignore_paths=[],
            project_root=tmp_path,
            module_root=tmp_path,
        )

        assert count == 1
        assert functions[js_file][0].function_name == "add"

    def test_get_class_method(self, tmp_path):
        """Test getting a specific class method."""
        js_file = tmp_path / "calculator.js"
        js_file.write_text("""
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

function standaloneFunc() {
    return 42;
}
""")
        test_config = TestConfig(
            tests_root=str(tmp_path / "tests"),
            project_root_path=str(tmp_path),
            test_framework="jest",
            tests_project_rootdir=tmp_path / "tests",
        )

        functions, count, _ = get_functions_to_optimize(
            optimize_all=None,
            replay_test=None,
            file=js_file,
            only_get_this_function="Calculator.add",
            test_cfg=test_config,
            ignore_paths=[],
            project_root=tmp_path,
            module_root=tmp_path,
        )

        assert count == 1
        fn = functions[js_file][0]
        assert fn.function_name == "add"
        assert fn.qualified_name == "Calculator.add"


class TestGetAllFilesAndFunctionsJavaScript:
    """Tests for get_all_files_and_functions with JavaScript files."""

    def test_discover_all_js_functions(self, tmp_path):
        """Test discovering all JavaScript functions in a directory."""
        # Create multiple JS files
        (tmp_path / "math.js").write_text("""
function add(a, b) {
    return a + b;
}
""")
        (tmp_path / "string.js").write_text("""
function reverse(str) {
    return str.split('').reverse().join('');
}
""")
        # Create a non-JS file that should be ignored
        (tmp_path / "readme.txt").write_text("This is not code")

        functions = get_all_files_and_functions(tmp_path, language=Language.JAVASCRIPT)

        assert len(functions) == 2
        all_names = set()
        for funcs in functions.values():
            for fn in funcs:
                all_names.add(fn.function_name)

        assert all_names == {"add", "reverse"}

    def test_discover_both_python_and_javascript(self, tmp_path):
        """Test discovering functions from both Python and JavaScript."""
        (tmp_path / "py_module.py").write_text("""
def py_func():
    return 1
""")
        (tmp_path / "js_module.js").write_text("""
function jsFunc() {
    return 1;
}
""")

        functions = get_all_files_and_functions(tmp_path, language=None)

        assert len(functions) == 2

        all_funcs = []
        for funcs in functions.values():
            all_funcs.extend(funcs)

        languages = {fn.language for fn in all_funcs}
        assert "python" in languages
        assert "javascript" in languages


class TestFunctionToOptimizeJavaScript:
    """Tests for FunctionToOptimize dataclass with JavaScript functions."""

    def test_qualified_name_no_parents(self, tmp_path):
        """Test qualified name for top-level function."""
        js_file = tmp_path / "module.js"
        js_file.write_text("""
function topLevel() {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)
        fn = functions[js_file][0]

        assert fn.qualified_name == "topLevel"
        assert fn.top_level_parent_name == "topLevel"

    def test_qualified_name_with_class_parent(self, tmp_path):
        """Test qualified name for class method."""
        js_file = tmp_path / "module.js"
        js_file.write_text("""
class MyClass {
    myMethod() {
        return 42;
    }
}
""")
        functions = find_all_functions_in_file(js_file)
        fn = functions[js_file][0]

        assert fn.qualified_name == "MyClass.myMethod"
        assert fn.top_level_parent_name == "MyClass"

    def test_language_attribute(self, tmp_path):
        """Test that JavaScript functions have correct language attribute."""
        js_file = tmp_path / "module.js"
        js_file.write_text("""
function myFunc() {
    return 42;
}
""")
        functions = find_all_functions_in_file(js_file)
        fn = functions[js_file][0]

        assert fn.language == "javascript"
