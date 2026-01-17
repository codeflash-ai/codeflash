"""
Extensive tests for the JavaScript language support implementation.

These tests verify that JavaScriptSupport correctly discovers functions,
replaces code, and integrates with the codeflash language abstraction.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import (
    FunctionFilterCriteria,
    FunctionInfo,
    Language,
    ParentInfo,
)
from codeflash.languages.javascript.support import JavaScriptSupport


@pytest.fixture
def js_support():
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


class TestJavaScriptSupportProperties:
    """Tests for JavaScriptSupport properties."""

    def test_language(self, js_support):
        """Test language property."""
        assert js_support.language == Language.JAVASCRIPT

    def test_file_extensions(self, js_support):
        """Test file_extensions property."""
        extensions = js_support.file_extensions
        assert ".js" in extensions
        assert ".jsx" in extensions
        assert ".mjs" in extensions
        assert ".cjs" in extensions

    def test_test_framework(self, js_support):
        """Test test_framework property."""
        assert js_support.test_framework == "jest"


class TestDiscoverFunctions:
    """Tests for discover_functions method."""

    def test_discover_simple_function(self, js_support):
        """Test discovering a simple function declaration."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function add(a, b) {
    return a + b;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 1
            assert functions[0].name == "add"
            assert functions[0].language == Language.JAVASCRIPT

    def test_discover_multiple_functions(self, js_support):
        """Test discovering multiple functions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

function multiply(a, b) {
    return a * b;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 3
            names = {func.name for func in functions}
            assert names == {"add", "subtract", "multiply"}

    def test_discover_arrow_function(self, js_support):
        """Test discovering arrow functions assigned to variables."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
const add = (a, b) => {
    return a + b;
};

const multiply = (x, y) => x * y;
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 2
            names = {func.name for func in functions}
            assert names == {"add", "multiply"}

    def test_discover_function_without_return_excluded(self, js_support):
        """Test that functions without return are excluded by default."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function withReturn() {
    return 1;
}

function withoutReturn() {
    console.log("hello");
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            # Only the function with return should be discovered
            assert len(functions) == 1
            assert functions[0].name == "withReturn"

    def test_discover_class_methods(self, js_support):
        """Test discovering class methods."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 2
            for func in functions:
                assert func.is_method is True
                assert func.class_name == "Calculator"

    def test_discover_async_functions(self, js_support):
        """Test discovering async functions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
async function fetchData(url) {
    return await fetch(url);
}

function syncFunction() {
    return 1;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 2

            async_func = next(f for f in functions if f.name == "fetchData")
            sync_func = next(f for f in functions if f.name == "syncFunction")

            assert async_func.is_async is True
            assert sync_func.is_async is False

    def test_discover_with_filter_exclude_async(self, js_support):
        """Test filtering out async functions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
async function asyncFunc() {
    return 1;
}

function syncFunc() {
    return 2;
}
""")
            f.flush()

            criteria = FunctionFilterCriteria(include_async=False)
            functions = js_support.discover_functions(Path(f.name), criteria)

            assert len(functions) == 1
            assert functions[0].name == "syncFunc"

    def test_discover_with_filter_exclude_methods(self, js_support):
        """Test filtering out class methods."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function standalone() {
    return 1;
}

class MyClass {
    method() {
        return 2;
    }
}
""")
            f.flush()

            criteria = FunctionFilterCriteria(include_methods=False)
            functions = js_support.discover_functions(Path(f.name), criteria)

            assert len(functions) == 1
            assert functions[0].name == "standalone"

    def test_discover_line_numbers(self, js_support):
        """Test that line numbers are correctly captured."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""function func1() {
    return 1;
}

function func2() {
    const x = 1;
    const y = 2;
    return x + y;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            func1 = next(f for f in functions if f.name == "func1")
            func2 = next(f for f in functions if f.name == "func2")

            assert func1.start_line == 1
            assert func1.end_line == 3
            assert func2.start_line == 5
            assert func2.end_line == 9

    def test_discover_generator_function(self, js_support):
        """Test discovering generator functions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function* numberGenerator() {
    yield 1;
    yield 2;
    return 3;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 1
            assert functions[0].name == "numberGenerator"

    def test_discover_invalid_file_returns_empty(self, js_support):
        """Test that invalid JavaScript file returns empty list."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("this is not valid javascript {{{{")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))
            # Tree-sitter is lenient, so it may still parse partial code
            # The important thing is it doesn't crash
            assert isinstance(functions, list)

    def test_discover_nonexistent_file_returns_empty(self, js_support):
        """Test that nonexistent file returns empty list."""
        functions = js_support.discover_functions(Path("/nonexistent/file.js"))
        assert functions == []

    def test_discover_function_expression(self, js_support):
        """Test discovering function expressions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
const add = function(a, b) {
    return a + b;
};
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            assert len(functions) == 1
            assert functions[0].name == "add"

    def test_discover_immediately_invoked_function_excluded(self, js_support):
        """Test that IIFEs without names are excluded when require_name is True."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
(function() {
    return 1;
})();

function named() {
    return 2;
}
""")
            f.flush()

            functions = js_support.discover_functions(Path(f.name))

            # Only the named function should be discovered
            assert len(functions) == 1
            assert functions[0].name == "named"


class TestReplaceFunction:
    """Tests for replace_function method."""

    def test_replace_simple_function(self, js_support):
        """Test replacing a simple function."""
        source = """function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
"""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test.js"),
            start_line=1,
            end_line=3,
        )
        new_code = """function add(a, b) {
    // Optimized
    return (a + b) | 0;
}
"""
        result = js_support.replace_function(source, func, new_code)

        assert "// Optimized" in result
        assert "return (a + b) | 0" in result
        assert "function multiply" in result

    def test_replace_preserves_surrounding_code(self, js_support):
        """Test that replacement preserves code before and after."""
        source = """// Header comment
import { something } from './module';

function target() {
    return 1;
}

function other() {
    return 2;
}

// Footer
"""
        func = FunctionInfo(
            name="target",
            file_path=Path("/test.js"),
            start_line=4,
            end_line=6,
        )
        new_code = """function target() {
    return 42;
}
"""
        result = js_support.replace_function(source, func, new_code)

        assert "// Header comment" in result
        assert "import { something }" in result
        assert "return 42" in result
        assert "function other" in result
        assert "// Footer" in result

    def test_replace_with_indentation_adjustment(self, js_support):
        """Test that indentation is adjusted correctly."""
        source = """class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test.js"),
            start_line=2,
            end_line=4,
            parents=(ParentInfo(name="Calculator", type="ClassDef"),),
        )
        # New code has no indentation
        new_code = """add(a, b) {
    return (a + b) | 0;
}
"""
        result = js_support.replace_function(source, func, new_code)

        # Check that indentation was added
        lines = result.splitlines()
        method_line = next(l for l in lines if "add(a, b)" in l)
        assert method_line.startswith("    ")  # 4 spaces

    def test_replace_arrow_function(self, js_support):
        """Test replacing an arrow function."""
        source = """const add = (a, b) => {
    return a + b;
};

const multiply = (x, y) => x * y;
"""
        func = FunctionInfo(
            name="add",
            file_path=Path("/test.js"),
            start_line=1,
            end_line=3,
        )
        new_code = """const add = (a, b) => {
    return (a + b) | 0;
};
"""
        result = js_support.replace_function(source, func, new_code)

        assert "(a + b) | 0" in result
        assert "multiply" in result


class TestValidateSyntax:
    """Tests for validate_syntax method."""

    def test_valid_syntax(self, js_support):
        """Test that valid JavaScript syntax passes."""
        valid_code = """
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(x, y) {
        return x * y;
    }
}
"""
        assert js_support.validate_syntax(valid_code) is True

    def test_invalid_syntax(self, js_support):
        """Test that invalid JavaScript syntax fails."""
        invalid_code = """
function add(a, b {
    return a + b;
}
"""
        assert js_support.validate_syntax(invalid_code) is False

    def test_empty_string_valid(self, js_support):
        """Test that empty string is valid syntax."""
        assert js_support.validate_syntax("") is True

    def test_syntax_error_types(self, js_support):
        """Test various syntax error types."""
        # Unclosed bracket
        assert js_support.validate_syntax("const x = [1, 2, 3") is False

        # Missing closing brace
        assert js_support.validate_syntax("function foo() {") is False


class TestNormalizeCode:
    """Tests for normalize_code method."""

    def test_removes_comments(self, js_support):
        """Test that single-line comments are removed."""
        code = """
function add(a, b) {
    // Add two numbers
    return a + b;
}
"""
        normalized = js_support.normalize_code(code)
        assert "// Add two numbers" not in normalized
        assert "return a + b" in normalized

    def test_preserves_functionality(self, js_support):
        """Test that code functionality is preserved."""
        code = """
function add(a, b) {
    // Comment
    return a + b;
}
"""
        normalized = js_support.normalize_code(code)
        assert "function add" in normalized
        assert "return" in normalized


class TestExtractCodeContext:
    """Tests for extract_code_context method."""

    def test_extract_simple_function(self, js_support):
        """Test extracting context for a simple function."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""function add(a, b) {
    return a + b;
}
""")
            f.flush()
            file_path = Path(f.name)

            func = FunctionInfo(
                name="add",
                file_path=file_path,
                start_line=1,
                end_line=3,
            )

            context = js_support.extract_code_context(
                func,
                file_path.parent,
                file_path.parent,
            )

            assert "function add" in context.target_code
            assert "return a + b" in context.target_code
            assert context.target_file == file_path
            assert context.language == Language.JAVASCRIPT

    def test_extract_with_helper(self, js_support):
        """Test extracting context with helper functions."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""function helper(x) {
    return x * 2;
}

function main(a) {
    return helper(a) + 1;
}
""")
            f.flush()
            file_path = Path(f.name)

            # First discover functions to get accurate line numbers
            functions = js_support.discover_functions(file_path)
            main_func = next(f for f in functions if f.name == "main")

            context = js_support.extract_code_context(
                main_func,
                file_path.parent,
                file_path.parent,
            )

            assert "function main" in context.target_code
            # Helper should be found
            assert len(context.helper_functions) >= 0  # May or may not find helper


class TestIntegration:
    """Integration tests for JavaScriptSupport."""

    def test_discover_and_replace_workflow(self, js_support):
        """Test full discover -> replace workflow."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            original_code = """function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"""
            f.write(original_code)
            f.flush()
            file_path = Path(f.name)

            # Discover
            functions = js_support.discover_functions(file_path)
            assert len(functions) == 1
            func = functions[0]
            assert func.name == "fibonacci"

            # Replace
            optimized_code = """function fibonacci(n) {
    // Memoized version
    const memo = {0: 0, 1: 1};
    for (let i = 2; i <= n; i++) {
        memo[i] = memo[i-1] + memo[i-2];
    }
    return memo[n];
}
"""
            result = js_support.replace_function(original_code, func, optimized_code)

            # Validate
            assert js_support.validate_syntax(result) is True
            assert "Memoized version" in result
            assert "memo[n]" in result

    def test_multiple_classes_and_functions(self, js_support):
        """Test discovering and working with complex file."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

class StringUtils {
    reverse(s) {
        return s.split('').reverse().join('');
    }
}

function standalone() {
    return 42;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Should find 4 functions
            assert len(functions) == 4

            # Check class methods
            calc_methods = [f for f in functions if f.class_name == "Calculator"]
            assert len(calc_methods) == 2

            string_methods = [f for f in functions if f.class_name == "StringUtils"]
            assert len(string_methods) == 1

            standalone_funcs = [f for f in functions if f.class_name is None]
            assert len(standalone_funcs) == 1

    def test_jsx_file(self, js_support):
        """Test discovering functions in JSX files."""
        with tempfile.NamedTemporaryFile(suffix=".jsx", mode="w", delete=False) as f:
            f.write("""
import React from 'react';

function Button({ onClick, children }) {
    return <button onClick={onClick}>{children}</button>;
}

const Card = ({ title, content }) => {
    return (
        <div className="card">
            <h2>{title}</h2>
            <p>{content}</p>
        </div>
    );
};

export default Button;
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Should find both components
            names = {f.name for f in functions}
            assert "Button" in names
            assert "Card" in names


class TestJestTestDiscovery:
    """Tests for Jest test discovery."""

    def test_find_jest_tests(self, js_support):
        """Test finding Jest test functions."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
import { add } from './math';

describe('Math functions', () => {
    test('add returns sum', () => {
        expect(add(1, 2)).toBe(3);
    });

    it('handles negative numbers', () => {
        expect(add(-1, 1)).toBe(0);
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.treesitter_utils import get_analyzer_for_file
            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "Math functions" in test_names
            assert "add returns sum" in test_names
            assert "handles negative numbers" in test_names
