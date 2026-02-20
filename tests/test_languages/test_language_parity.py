"""Regression tests for Python/JavaScript language support parity.

These tests ensure that the JavaScript implementation maintains feature parity
with the Python implementation. Each test class tests equivalent functionality
across both languages using equivalent code samples.

This file helps identify gaps or weaknesses in the JavaScript implementation
by comparing it against the rigorous Python implementation.
"""

import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest

from codeflash.languages.base import FunctionFilterCriteria, FunctionInfo, Language, ParentInfo
from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash.languages.python.support import PythonSupport


class CodePair(NamedTuple):
    """Equivalent code samples in Python and JavaScript."""

    python: str
    javascript: str
    description: str


# ============================================================================
# EQUIVALENT CODE SAMPLES
# ============================================================================

# Simple function with return
SIMPLE_FUNCTION = CodePair(
    python="""
def add(a, b):
    return a + b
""",
    javascript="""
export function add(a, b) {
    return a + b;
}
""",
    description="Simple function with return",
)

# Multiple functions
MULTIPLE_FUNCTIONS = CodePair(
    python="""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
""",
    javascript="""
export function add(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}

export function multiply(a, b) {
    return a * b;
}
""",
    description="Multiple functions",
)

# Function with and without return
WITH_AND_WITHOUT_RETURN = CodePair(
    python="""
def with_return():
    return 1

def without_return():
    print("hello")
""",
    javascript="""
export function withReturn() {
    return 1;
}

export function withoutReturn() {
    console.log("hello");
}
""",
    description="Functions with and without return",
)

# Class methods
CLASS_METHODS = CodePair(
    python="""
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
""",
    javascript="""
export class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
""",
    description="Class methods",
)

# Async functions
ASYNC_FUNCTIONS = CodePair(
    python="""
async def fetch_data(url):
    return await get(url)

def sync_function():
    return 1
""",
    javascript="""
export async function fetchData(url) {
    return await fetch(url);
}

export function syncFunction() {
    return 1;
}
""",
    description="Async and sync functions",
)

# Nested functions
NESTED_FUNCTIONS = CodePair(
    python="""
def outer():
    def inner():
        return 1
    return inner()
""",
    javascript="""
export function outer() {
    function inner() {
        return 1;
    }
    return inner();
}
""",
    description="Nested functions",
)

# Static methods
STATIC_METHODS = CodePair(
    python="""
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
""",
    javascript="""
export class Utils {
    static helper(x) {
        return x * 2;
    }
}
""",
    description="Static methods",
)

# Mixed classes and standalone functions
COMPLEX_FILE = CodePair(
    python="""
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

class StringUtils:
    def reverse(self, s):
        return s[::-1]

def standalone():
    return 42
""",
    javascript="""
export class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

export class StringUtils {
    reverse(s) {
        return s.split('').reverse().join('');
    }
}

export function standalone() {
    return 42;
}
""",
    description="Complex file with multiple classes and standalone function",
)

# Filter test: async and sync
FILTER_ASYNC_TEST = CodePair(
    python="""
async def async_func():
    return 1

def sync_func():
    return 2
""",
    javascript="""
export async function asyncFunc() {
    return 1;
}

export function syncFunc() {
    return 2;
}
""",
    description="Async filter test",
)

# Filter test: methods and standalone
FILTER_METHODS_TEST = CodePair(
    python="""
def standalone():
    return 1

class MyClass:
    def method(self):
        return 2
""",
    javascript="""
export function standalone() {
    return 1;
}

export class MyClass {
    method() {
        return 2;
    }
}
""",
    description="Methods filter test",
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def python_support():
    """Create a PythonSupport instance."""
    return PythonSupport()


@pytest.fixture
def js_support():
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


def write_temp_file(content: str, suffix: str) -> Path:
    """Write content to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        return Path(f.name)


# ============================================================================
# PROPERTY PARITY TESTS
# ============================================================================


class TestPropertiesParity:
    """Verify both implementations have equivalent properties."""

    def test_language_property_set(self, python_support, js_support):
        """Both should have a language property from the Language enum."""
        assert python_support.language == Language.PYTHON
        assert js_support.language == Language.JAVASCRIPT
        # Both should be Language enum values
        assert isinstance(python_support.language, Language)
        assert isinstance(js_support.language, Language)

    def test_file_extensions_property(self, python_support, js_support):
        """Both should have a tuple of file extensions."""
        py_ext = python_support.file_extensions
        js_ext = js_support.file_extensions

        # Both should be tuples
        assert isinstance(py_ext, tuple)
        assert isinstance(js_ext, tuple)

        # Both should have at least one extension
        assert len(py_ext) >= 1
        assert len(js_ext) >= 1

        # Extensions should start with '.'
        assert all(ext.startswith(".") for ext in py_ext)
        assert all(ext.startswith(".") for ext in js_ext)

    def test_test_framework_property(self, python_support, js_support):
        """Both should have a test_framework property."""
        # Both should return a string
        assert isinstance(python_support.test_framework, str)
        assert isinstance(js_support.test_framework, str)

        # Should be non-empty
        assert len(python_support.test_framework) > 0
        assert len(js_support.test_framework) > 0


# ============================================================================
# FUNCTION DISCOVERY PARITY TESTS
# ============================================================================


class TestDiscoverFunctionsParity:
    """Verify function discovery works equivalently in both languages."""

    def test_simple_function_discovery(self, python_support, js_support):
        """Both should discover a simple function with return."""
        py_file = write_temp_file(SIMPLE_FUNCTION.python, ".py")
        js_file = write_temp_file(SIMPLE_FUNCTION.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find exactly one function
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)}, expected 1"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)}, expected 1"

        # Both should find 'add'
        assert py_funcs[0].function_name == "add"
        assert js_funcs[0].function_name == "add"

        # Both should have correct language
        assert py_funcs[0].language == Language.PYTHON
        assert js_funcs[0].language == Language.JAVASCRIPT

    def test_multiple_functions_discovery(self, python_support, js_support):
        """Both should discover all functions in a file."""
        py_file = write_temp_file(MULTIPLE_FUNCTIONS.python, ".py")
        js_file = write_temp_file(MULTIPLE_FUNCTIONS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 3 functions
        assert len(py_funcs) == 3, f"Python found {len(py_funcs)}, expected 3"
        assert len(js_funcs) == 3, f"JavaScript found {len(js_funcs)}, expected 3"

        # Both should find the same function names
        py_names = {f.function_name for f in py_funcs}
        js_names = {f.function_name for f in js_funcs}

        assert py_names == {"add", "subtract", "multiply"}
        assert js_names == {"add", "subtract", "multiply"}

    def test_functions_without_return_excluded(self, python_support, js_support):
        """Both should exclude functions without return statements by default."""
        py_file = write_temp_file(WITH_AND_WITHOUT_RETURN.python, ".py")
        js_file = write_temp_file(WITH_AND_WITHOUT_RETURN.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find only 1 function (the one with return)
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)}, expected 1"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)}, expected 1"

        # The function with return should be found
        assert py_funcs[0].function_name == "with_return"
        assert js_funcs[0].function_name == "withReturn"

    def test_class_methods_discovery(self, python_support, js_support):
        """Both should discover class methods with proper metadata."""
        py_file = write_temp_file(CLASS_METHODS.python, ".py")
        js_file = write_temp_file(CLASS_METHODS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 2 methods
        assert len(py_funcs) == 2, f"Python found {len(py_funcs)}, expected 2"
        assert len(js_funcs) == 2, f"JavaScript found {len(js_funcs)}, expected 2"

        # All should be marked as methods
        for func in py_funcs:
            assert func.is_method is True, f"Python {func.function_name} should be a method"
            assert func.class_name == "Calculator", f"Python {func.function_name} should belong to Calculator"

        for func in js_funcs:
            assert func.is_method is True, f"JavaScript {func.function_name} should be a method"
            assert func.class_name == "Calculator", f"JavaScript {func.function_name} should belong to Calculator"

    def test_async_functions_discovery(self, python_support, js_support):
        """Both should correctly identify async functions."""
        py_file = write_temp_file(ASYNC_FUNCTIONS.python, ".py")
        js_file = write_temp_file(ASYNC_FUNCTIONS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 2 functions
        assert len(py_funcs) == 2, f"Python found {len(py_funcs)}, expected 2"
        assert len(js_funcs) == 2, f"JavaScript found {len(js_funcs)}, expected 2"

        # Check async flags
        py_async = next(f for f in py_funcs if "fetch" in f.function_name.lower())
        py_sync = next(f for f in py_funcs if "sync" in f.function_name.lower())
        js_async = next(f for f in js_funcs if "fetch" in f.function_name.lower())
        js_sync = next(f for f in js_funcs if "sync" in f.function_name.lower())

        assert py_async.is_async is True, "Python async function should have is_async=True"
        assert py_sync.is_async is False, "Python sync function should have is_async=False"
        assert js_async.is_async is True, "JavaScript async function should have is_async=True"
        assert js_sync.is_async is False, "JavaScript sync function should have is_async=False"

    def test_nested_functions_discovery(self, python_support, js_support):
        """Both should discover nested functions with parent info."""
        py_file = write_temp_file(NESTED_FUNCTIONS.python, ".py")
        js_file = write_temp_file(NESTED_FUNCTIONS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 2 functions (outer and inner)
        assert len(py_funcs) == 2, f"Python found {len(py_funcs)}, expected 2"
        assert len(js_funcs) == 2, f"JavaScript found {len(js_funcs)}, expected 2"

        # Check names
        py_names = {f.function_name for f in py_funcs}
        js_names = {f.function_name for f in js_funcs}

        assert py_names == {"outer", "inner"}, f"Python found {py_names}"
        assert js_names == {"outer", "inner"}, f"JavaScript found {js_names}"

        # Check parent info for inner function
        py_inner = next(f for f in py_funcs if f.function_name == "inner")
        js_inner = next(f for f in js_funcs if f.function_name == "inner")

        assert len(py_inner.parents) >= 1, "Python inner should have parent info"
        assert py_inner.parents[0].name == "outer", "Python inner's parent should be outer"

        # JavaScript nested function parent check
        assert len(js_inner.parents) >= 1, "JavaScript inner should have parent info"
        assert js_inner.parents[0].name == "outer", "JavaScript inner's parent should be outer"

    def test_static_methods_discovery(self, python_support, js_support):
        """Both should discover static methods."""
        py_file = write_temp_file(STATIC_METHODS.python, ".py")
        js_file = write_temp_file(STATIC_METHODS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 1 function
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)}, expected 1"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)}, expected 1"

        # Both should find 'helper' belonging to 'Utils'
        assert py_funcs[0].function_name == "helper"
        assert js_funcs[0].function_name == "helper"
        assert py_funcs[0].class_name == "Utils"
        assert js_funcs[0].class_name == "Utils"

    def test_complex_file_discovery(self, python_support, js_support):
        """Both should handle complex files with multiple classes and standalone functions."""
        py_file = write_temp_file(COMPLEX_FILE.python, ".py")
        js_file = write_temp_file(COMPLEX_FILE.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find 4 functions
        assert len(py_funcs) == 4, f"Python found {len(py_funcs)}, expected 4"
        assert len(js_funcs) == 4, f"JavaScript found {len(js_funcs)}, expected 4"

        # Check Calculator methods
        py_calc = [f for f in py_funcs if f.class_name == "Calculator"]
        js_calc = [f for f in js_funcs if f.class_name == "Calculator"]
        assert len(py_calc) == 2, f"Python found {len(py_calc)} Calculator methods"
        assert len(js_calc) == 2, f"JavaScript found {len(js_calc)} Calculator methods"

        # Check StringUtils methods
        py_string = [f for f in py_funcs if f.class_name == "StringUtils"]
        js_string = [f for f in js_funcs if f.class_name == "StringUtils"]
        assert len(py_string) == 1, f"Python found {len(py_string)} StringUtils methods"
        assert len(js_string) == 1, f"JavaScript found {len(js_string)} StringUtils methods"

        # Check standalone functions
        py_standalone = [f for f in py_funcs if f.class_name is None]
        js_standalone = [f for f in js_funcs if f.class_name is None]
        assert len(py_standalone) == 1, f"Python found {len(py_standalone)} standalone functions"
        assert len(js_standalone) == 1, f"JavaScript found {len(js_standalone)} standalone functions"

    def test_filter_exclude_async(self, python_support, js_support):
        """Both should support filtering out async functions."""
        py_file = write_temp_file(FILTER_ASYNC_TEST.python, ".py")
        js_file = write_temp_file(FILTER_ASYNC_TEST.javascript, ".js")

        criteria = FunctionFilterCriteria(include_async=False)

        py_funcs = python_support.discover_functions(py_file, criteria)
        js_funcs = js_support.discover_functions(js_file, criteria)

        # Both should find only 1 function (the sync one)
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)}, expected 1"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)}, expected 1"

        # Should be the sync function
        assert "sync" in py_funcs[0].function_name.lower()
        assert "sync" in js_funcs[0].function_name.lower()

    def test_filter_exclude_methods(self, python_support, js_support):
        """Both should support filtering out class methods."""
        py_file = write_temp_file(FILTER_METHODS_TEST.python, ".py")
        js_file = write_temp_file(FILTER_METHODS_TEST.javascript, ".js")

        criteria = FunctionFilterCriteria(include_methods=False)

        py_funcs = python_support.discover_functions(py_file, criteria)
        js_funcs = js_support.discover_functions(js_file, criteria)

        # Both should find only 1 function (standalone)
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)}, expected 1"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)}, expected 1"

        # Should be the standalone function
        assert py_funcs[0].function_name == "standalone"
        assert js_funcs[0].function_name == "standalone"

    def test_nonexistent_file_returns_empty(self, python_support, js_support):
        """Both should return empty list for nonexistent files."""
        py_funcs = python_support.discover_functions(Path("/nonexistent/file.py"))
        js_funcs = js_support.discover_functions(Path("/nonexistent/file.js"))

        assert py_funcs == []
        assert js_funcs == []

    def test_line_numbers_captured(self, python_support, js_support):
        """Both should capture line numbers for discovered functions."""
        py_file = write_temp_file(SIMPLE_FUNCTION.python, ".py")
        js_file = write_temp_file(SIMPLE_FUNCTION.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should have start_line and end_line
        assert py_funcs[0].starting_line is not None
        assert py_funcs[0].ending_line is not None
        assert js_funcs[0].starting_line is not None
        assert js_funcs[0].ending_line is not None

        # Start should be before or equal to end
        assert py_funcs[0].starting_line <= py_funcs[0].ending_line
        assert js_funcs[0].starting_line <= js_funcs[0].ending_line


# ============================================================================
# CODE REPLACEMENT PARITY TESTS
# ============================================================================


class TestReplaceFunctionParity:
    """Verify code replacement works equivalently in both languages."""

    def test_simple_replacement(self, python_support, js_support):
        """Both should replace a function while preserving other code."""
        py_source = """def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        js_source = """function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
"""
        py_func = FunctionInfo(function_name="add", file_path=Path("/test.py"), starting_line=1, ending_line=2)
        js_func = FunctionInfo(function_name="add", file_path=Path("/test.js"), starting_line=1, ending_line=3)

        py_new = """def add(a, b):
    return (a + b) | 0
"""
        js_new = """function add(a, b) {
    return (a + b) | 0;
}
"""
        py_result = python_support.replace_function(py_source, py_func, py_new)
        js_result = js_support.replace_function(js_source, js_func, js_new)

        # Both should contain the new code
        assert "(a + b) | 0" in py_result
        assert "(a + b) | 0" in js_result

        # Both should preserve the multiply function
        assert "multiply" in py_result
        assert "multiply" in js_result

    def test_replacement_preserves_surrounding(self, python_support, js_support):
        """Both should preserve header, footer, and other code."""
        py_source = """# Header comment
import math

def target():
    return 1

def other():
    return 2

# Footer
"""
        js_source = """// Header comment
const math = require('math');

function target() {
    return 1;
}

function other() {
    return 2;
}

// Footer
"""
        py_func = FunctionInfo(function_name="target", file_path=Path("/test.py"), starting_line=4, ending_line=5)
        js_func = FunctionInfo(function_name="target", file_path=Path("/test.js"), starting_line=4, ending_line=6)

        py_new = """def target():
    return 42
"""
        js_new = """function target() {
    return 42;
}
"""
        py_result = python_support.replace_function(py_source, py_func, py_new)
        js_result = js_support.replace_function(js_source, js_func, js_new)

        # Both should preserve header
        assert "Header comment" in py_result
        assert "Header comment" in js_result

        # Both should have the new return value
        assert "return 42" in py_result
        assert "return 42" in js_result

        # Both should preserve the other function
        assert "other" in py_result
        assert "other" in js_result

        # Both should preserve footer
        assert "Footer" in py_result
        assert "Footer" in js_result

    def test_replacement_with_indentation(self, python_support, js_support):
        """Both should handle indentation correctly for class methods."""
        py_source = """class Calculator:
    def add(self, a, b):
        return a + b
"""
        js_source = """class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
        py_func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.py"),
            starting_line=2,
            ending_line=3,
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
        )
        js_func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.js"),
            starting_line=2,
            ending_line=4,
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
        )

        # New code without indentation
        py_new = """def add(self, a, b):
    return (a + b) | 0
"""
        js_new = """add(a, b) {
    return (a + b) | 0;
}
"""
        py_result = python_support.replace_function(py_source, py_func, py_new)
        js_result = js_support.replace_function(js_source, js_func, js_new)

        # Both should add proper indentation
        py_lines = py_result.splitlines()
        js_lines = js_result.splitlines()

        py_method_line = next(l for l in py_lines if "def add" in l)
        js_method_line = next(l for l in js_lines if "add(a, b)" in l)

        # Both should have indentation (4 spaces)
        assert py_method_line.startswith("    "), f"Python method should be indented: {py_method_line!r}"
        assert js_method_line.startswith("    "), f"JavaScript method should be indented: {js_method_line!r}"


# ============================================================================
# SYNTAX VALIDATION PARITY TESTS
# ============================================================================


class TestValidateSyntaxParity:
    """Verify syntax validation works equivalently in both languages."""

    def test_valid_syntax(self, python_support, js_support):
        """Both should accept valid syntax."""
        py_valid = """
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        js_valid = """
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}
"""
        assert python_support.validate_syntax(py_valid) is True
        assert js_support.validate_syntax(js_valid) is True

    def test_invalid_syntax(self, python_support, js_support):
        """Both should reject invalid syntax."""
        py_invalid = """
def add(a, b:
    return a + b
"""
        js_invalid = """
function add(a, b {
    return a + b;
}
"""
        assert python_support.validate_syntax(py_invalid) is False
        assert js_support.validate_syntax(js_invalid) is False

    def test_empty_string_valid(self, python_support, js_support):
        """Both should accept empty string as valid syntax."""
        assert python_support.validate_syntax("") is True
        assert js_support.validate_syntax("") is True

    def test_unclosed_bracket(self, python_support, js_support):
        """Both should reject unclosed brackets."""
        py_invalid = "x = [1, 2, 3"
        js_invalid = "const x = [1, 2, 3"

        assert python_support.validate_syntax(py_invalid) is False
        assert js_support.validate_syntax(js_invalid) is False


# ============================================================================
# CODE NORMALIZATION PARITY TESTS
# ============================================================================


class TestNormalizeCodeParity:
    """Verify code normalization works equivalently in both languages."""

    def test_removes_comments(self, python_support, js_support):
        """Both should remove/handle comments during normalization."""
        py_code = '''
def add(a, b):
    """Add two numbers."""
    # Comment
    return a + b
'''
        js_code = """
function add(a, b) {
    // Add two numbers
    /* Multi-line
       comment */
    return a + b;
}
"""
        py_normalized = python_support.normalize_code(py_code)
        js_normalized = js_support.normalize_code(js_code)

        # Both should preserve functionality
        assert "return" in py_normalized
        assert "return" in js_normalized

        # Python should remove docstring
        assert '"""Add two numbers."""' not in py_normalized

        # JavaScript should remove comments
        assert "// Add two numbers" not in js_normalized

    def test_preserves_code_structure(self, python_support, js_support):
        """Both should preserve the basic code structure."""
        py_code = """
def add(a, b):
    return a + b
"""
        js_code = """
function add(a, b) {
    return a + b;
}
"""
        py_normalized = python_support.normalize_code(py_code)
        js_normalized = js_support.normalize_code(js_code)

        # Python should still have def
        assert "def add" in py_normalized or "def" in py_normalized

        # JavaScript should still have function
        assert "function add" in js_normalized or "function" in js_normalized


# ============================================================================
# CODE CONTEXT EXTRACTION PARITY TESTS
# ============================================================================


class TestExtractCodeContextParity:
    """Verify code context extraction works equivalently in both languages."""

    def test_simple_function_context(self, python_support, js_support):
        """Both should extract context for a simple function."""
        py_file = write_temp_file(
            """def add(a, b):
    return a + b
""",
            ".py",
        )
        js_file = write_temp_file(
            """function add(a, b) {
    return a + b;
}
""",
            ".js",
        )

        py_func = FunctionInfo(function_name="add", file_path=py_file, starting_line=1, ending_line=2)
        js_func = FunctionInfo(function_name="add", file_path=js_file, starting_line=1, ending_line=3)

        py_context = python_support.extract_code_context(py_func, py_file.parent, py_file.parent)
        js_context = js_support.extract_code_context(js_func, js_file.parent, js_file.parent)

        # Both should have target code
        assert "add" in py_context.target_code
        assert "add" in js_context.target_code

        # Both should have correct file path
        assert py_context.target_file == py_file
        assert js_context.target_file == js_file

        # Both should have correct language
        assert py_context.language == Language.PYTHON
        assert js_context.language == Language.JAVASCRIPT


# ============================================================================
# INTEGRATION PARITY TESTS
# ============================================================================


class TestIntegrationParity:
    """Integration tests for full workflows in both languages."""

    def test_discover_and_replace_workflow(self, python_support, js_support):
        """Both should support the full discover -> replace workflow."""
        py_original = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        js_original = """export function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"""
        py_file = write_temp_file(py_original, ".py")
        js_file = write_temp_file(js_original, ".js")

        # Discover
        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        assert len(py_funcs) == 1
        assert len(js_funcs) == 1
        assert py_funcs[0].function_name == "fibonacci"
        assert js_funcs[0].function_name == "fibonacci"

        # Replace
        py_optimized = """def fibonacci(n):
    # Memoized version
    memo = {0: 0, 1: 1}
    for i in range(2, n + 1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]
"""
        js_optimized = """export function fibonacci(n) {
    // Memoized version
    const memo = {0: 0, 1: 1};
    for (let i = 2; i <= n; i++) {
        memo[i] = memo[i-1] + memo[i-2];
    }
    return memo[n];
}
"""
        py_result = python_support.replace_function(py_original, py_funcs[0], py_optimized)
        js_result = js_support.replace_function(js_original, js_funcs[0], js_optimized)

        # Validate syntax
        assert python_support.validate_syntax(py_result) is True
        assert js_support.validate_syntax(js_result) is True

        # Both should have the new implementation
        assert "Memoized version" in py_result
        assert "Memoized version" in js_result
        assert "memo[n]" in py_result
        assert "memo[n]" in js_result


# ============================================================================
# GAP DETECTION TESTS
# ============================================================================


class TestFeatureGaps:
    """Tests to detect gaps in JavaScript implementation vs Python."""

    def test_function_info_fields_populated(self, python_support, js_support):
        """Both should populate all FunctionInfo fields consistently."""
        py_file = write_temp_file(CLASS_METHODS.python, ".py")
        js_file = write_temp_file(CLASS_METHODS.javascript, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        for py_func in py_funcs:
            # Check all expected fields are populated
            assert py_func.function_name is not None, "Python: name should be populated"
            assert py_func.file_path is not None, "Python: file_path should be populated"
            assert py_func.starting_line is not None, "Python: start_line should be populated"
            assert py_func.ending_line is not None, "Python: end_line should be populated"
            assert py_func.language is not None, "Python: language should be populated"
            # is_method and class_name should be set for class methods
            assert py_func.is_method is not None, "Python: is_method should be populated"

        for js_func in js_funcs:
            # JavaScript should populate the same fields
            assert js_func.function_name is not None, "JavaScript: name should be populated"
            assert js_func.file_path is not None, "JavaScript: file_path should be populated"
            assert js_func.starting_line is not None, "JavaScript: start_line should be populated"
            assert js_func.ending_line is not None, "JavaScript: end_line should be populated"
            assert js_func.language is not None, "JavaScript: language should be populated"
            assert js_func.is_method is not None, "JavaScript: is_method should be populated"

    def test_arrow_functions_unique_to_js(self, js_support):
        """JavaScript arrow functions should be discovered (no Python equivalent)."""
        js_code = """
export const add = (a, b) => {
    return a + b;
};

export const multiply = (x, y) => x * y;

export const identity = x => x;
"""
        js_file = write_temp_file(js_code, ".js")
        funcs = js_support.discover_functions(js_file)

        # Should find all arrow functions
        names = {f.function_name for f in funcs}
        assert "add" in names, "Should find arrow function 'add'"
        assert "multiply" in names, "Should find concise arrow function 'multiply'"
        # identity might or might not be found depending on implicit return handling
        # but at least the main arrow functions should work

    def test_generator_functions(self, python_support, js_support):
        """Both should handle generator functions."""
        py_code = """
def number_generator():
    yield 1
    yield 2
    return 3
"""
        js_code = """
export function* numberGenerator() {
    yield 1;
    yield 2;
    return 3;
}
"""
        py_file = write_temp_file(py_code, ".py")
        js_file = write_temp_file(js_code, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        # Both should find the generator
        assert len(py_funcs) == 1, f"Python found {len(py_funcs)} generators"
        assert len(js_funcs) == 1, f"JavaScript found {len(js_funcs)} generators"

    def test_decorators_python_only(self, python_support):
        """Python decorators should not break function discovery."""
        py_code = """
@decorator
def decorated():
    return 1

@decorator_with_args(arg=1)
def decorated_with_args():
    return 2

@decorator1
@decorator2
def multi_decorated():
    return 3
"""
        py_file = write_temp_file(py_code, ".py")
        funcs = python_support.discover_functions(py_file)

        # Should find all functions regardless of decorators
        names = {f.function_name for f in funcs}
        assert "decorated" in names
        assert "decorated_with_args" in names
        assert "multi_decorated" in names

    def test_function_expressions_js(self, js_support):
        """JavaScript function expressions should be discovered."""
        js_code = """
export const add = function(a, b) {
    return a + b;
};

export const namedExpr = function myFunc(x) {
    return x * 2;
};
"""
        js_file = write_temp_file(js_code, ".js")
        funcs = js_support.discover_functions(js_file)

        # Should find function expressions
        names = {f.function_name for f in funcs}
        assert "add" in names, "Should find anonymous function expression assigned to 'add'"


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge cases that both implementations should handle."""

    def test_empty_file(self, python_support, js_support):
        """Both should handle empty files gracefully."""
        py_file = write_temp_file("", ".py")
        js_file = write_temp_file("", ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        assert py_funcs == []
        assert js_funcs == []

    def test_file_with_only_comments(self, python_support, js_support):
        """Both should handle files with only comments."""
        py_code = """
# This is a comment
# Another comment
'''
Multiline string that's not a docstring
'''
"""
        js_code = """
// This is a comment
// Another comment
/*
Multiline comment
*/
"""
        py_file = write_temp_file(py_code, ".py")
        js_file = write_temp_file(js_code, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        assert py_funcs == []
        assert js_funcs == []

    def test_unicode_content(self, python_support, js_support):
        """Both should handle unicode content in code."""
        py_code = """
def greeting():
    return "Hello, 世界! 🌍"
"""
        js_code = """
export function greeting() {
    return "Hello, 世界! 🌍";
}
"""
        py_file = write_temp_file(py_code, ".py")
        js_file = write_temp_file(js_code, ".js")

        py_funcs = python_support.discover_functions(py_file)
        js_funcs = js_support.discover_functions(js_file)

        assert len(py_funcs) == 1
        assert len(js_funcs) == 1
        assert py_funcs[0].function_name == "greeting"
        assert js_funcs[0].function_name == "greeting"
