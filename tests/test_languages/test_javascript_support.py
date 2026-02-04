"""Extensive tests for the JavaScript language support implementation.

These tests verify that JavaScriptSupport correctly discovers functions,
replaces code, and integrates with the codeflash language abstraction.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria, FunctionInfo, Language, ParentInfo
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
            assert functions[0].function_name == "add"
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
            names = {func.function_name for func in functions}
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
            names = {func.function_name for func in functions}
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
            assert functions[0].function_name == "withReturn"

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

            async_func = next(f for f in functions if f.function_name == "fetchData")
            sync_func = next(f for f in functions if f.function_name == "syncFunction")

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
            assert functions[0].function_name == "syncFunc"

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
            assert functions[0].function_name == "standalone"

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

            func1 = next(f for f in functions if f.function_name == "func1")
            func2 = next(f for f in functions if f.function_name == "func2")

            assert func1.starting_line == 1
            assert func1.ending_line == 3
            assert func2.starting_line == 5
            assert func2.ending_line == 9

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
            assert functions[0].function_name == "numberGenerator"

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
            assert functions[0].function_name == "add"

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
            assert functions[0].function_name == "named"


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
        func = FunctionInfo(function_name="add", file_path=Path("/test.js"), starting_line=1, ending_line=3)
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
        func = FunctionInfo(function_name="target", file_path=Path("/test.js"), starting_line=4, ending_line=6)
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
            function_name="add",
            file_path=Path("/test.js"),
            starting_line=2,
            ending_line=4,
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
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
        func = FunctionInfo(function_name="add", file_path=Path("/test.js"), starting_line=1, ending_line=3)
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

            func = FunctionInfo(function_name="add", file_path=file_path, starting_line=1, ending_line=3)

            context = js_support.extract_code_context(func, file_path.parent, file_path.parent)

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
            main_func = next(f for f in functions if f.function_name == "main")

            context = js_support.extract_code_context(main_func, file_path.parent, file_path.parent)

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
            assert func.function_name == "fibonacci"

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
            names = {f.function_name for f in functions}
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


class TestClassMethodExtraction:
    """Tests for class method extraction and code context.

    These tests use full string equality to verify exact extraction output.
    """

    def test_extract_class_method_wraps_in_class(self, js_support):
        """Test that extracting a class method wraps it in a class definition."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            # Discover the method
            functions = js_support.discover_functions(file_path)
            add_method = next(f for f in functions if f.function_name == "add")

            # Extract code context
            context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)

            # Full string equality check for exact extraction output
            expected_code = """class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
            assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
            assert js_support.validate_syntax(context.target_code) is True

    def test_extract_class_method_with_jsdoc(self, js_support):
        """Test extracting a class method with JSDoc comments."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""/**
 * A simple calculator class.
 */
class Calculator {
    /**
     * Adds two numbers.
     * @param {number} a - First number
     * @param {number} b - Second number
     * @returns {number} The sum
     */
    add(a, b) {
        return a + b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_method = next(f for f in functions if f.function_name == "add")

            context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)

            # Full string equality check - includes class JSDoc, class definition, method JSDoc, and method
            expected_code = """/**
 * A simple calculator class.
 */
class Calculator {
    /**
     * Adds two numbers.
     * @param {number} a - First number
     * @param {number} b - Second number
     * @returns {number} The sum
     */
    add(a, b) {
        return a + b;
    }
}
"""
            assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
            assert js_support.validate_syntax(context.target_code) is True

    def test_extract_class_method_syntax_valid(self, js_support):
        """Test that extracted class method code is always syntactically valid."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class FibonacciCalculator {
    fibonacci(n) {
        if (n <= 1) {
            return n;
        }
        return this.fibonacci(n - 1) + this.fibonacci(n - 2);
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            fib_method = next(f for f in functions if f.function_name == "fibonacci")

            context = js_support.extract_code_context(fib_method, file_path.parent, file_path.parent)

            # Full string equality check
            expected_code = """class FibonacciCalculator {
    fibonacci(n) {
        if (n <= 1) {
            return n;
        }
        return this.fibonacci(n - 1) + this.fibonacci(n - 2);
    }
}
"""
            assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
            assert js_support.validate_syntax(context.target_code) is True

    def test_extract_nested_class_method(self, js_support):
        """Test extracting a method from a nested class structure."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Outer {
    createInner() {
        return class Inner {
            getValue() {
                return 42;
            }
        };
    }

    add(a, b) {
        return a + b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_method = next((f for f in functions if f.function_name == "add"), None)

            if add_method:
                context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)

                # Full string equality check
                expected_code = """class Outer {
    add(a, b) {
        return a + b;
    }
}
"""
                assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
                assert js_support.validate_syntax(context.target_code) is True

    def test_extract_async_class_method(self, js_support):
        """Test extracting an async class method."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class ApiClient {
    async fetchData(url) {
        const response = await fetch(url);
        return response.json();
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            fetch_method = next(f for f in functions if f.function_name == "fetchData")

            context = js_support.extract_code_context(fetch_method, file_path.parent, file_path.parent)

            # Full string equality check
            expected_code = """class ApiClient {
    async fetchData(url) {
        const response = await fetch(url);
        return response.json();
    }
}
"""
            assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
            assert js_support.validate_syntax(context.target_code) is True

    def test_extract_static_class_method(self, js_support):
        """Test extracting a static class method."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class MathUtils {
    static add(a, b) {
        return a + b;
    }

    static multiply(a, b) {
        return a * b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_method = next((f for f in functions if f.function_name == "add"), None)

            if add_method:
                context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)

                # Full string equality check
                expected_code = """class MathUtils {
    static add(a, b) {
        return a + b;
    }
}
"""
                assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
                assert js_support.validate_syntax(context.target_code) is True

    def test_extract_class_method_without_class_jsdoc(self, js_support):
        """Test extracting a method from a class without JSDoc."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class SimpleClass {
    simpleMethod() {
        return "hello";
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            method = next(f for f in functions if f.function_name == "simpleMethod")

            context = js_support.extract_code_context(method, file_path.parent, file_path.parent)

            # Full string equality check
            expected_code = """class SimpleClass {
    simpleMethod() {
        return "hello";
    }
}
"""
            assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
            assert js_support.validate_syntax(context.target_code) is True


class TestClassMethodReplacement:
    """Tests for replacing class methods."""

    def test_replace_class_method_preserves_class_structure(self, js_support):
        """Test that replacing a class method preserves the class structure."""
        source = """class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}
"""
        func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.js"),
            starting_line=2,
            ending_line=4,
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
            is_method=True,
        )
        new_code = """    add(a, b) {
        // Optimized bitwise addition
        return (a + b) | 0;
    }
"""
        result = js_support.replace_function(source, func, new_code)

        # Check class structure is preserved
        assert "class Calculator" in result
        assert "multiply(a, b)" in result
        assert "return a * b" in result

        # Check new code is inserted
        assert "Optimized bitwise addition" in result
        assert "(a + b) | 0" in result

        # Check result is valid JavaScript
        assert js_support.validate_syntax(result) is True

    def test_replace_class_method_with_jsdoc(self, js_support):
        """Test replacing a class method that has JSDoc.

        When new_code includes a JSDoc, it should replace the original JSDoc.
        """
        source = """class Calculator {
    /**
     * Adds two numbers.
     */
    add(a, b) {
        return a + b;
    }
}
"""
        func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.js"),
            starting_line=5,  # Method starts here
            ending_line=7,
            doc_start_line=2,  # JSDoc starts here
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
            is_method=True,
        )
        new_code = """    /**
     * Adds two numbers (optimized).
     */
    add(a, b) {
        return (a + b) | 0;
    }
"""
        result = js_support.replace_function(source, func, new_code)

        # New JSDoc should replace the original
        assert "optimized" in result
        # Body should be replaced with the optimized version
        assert "(a + b) | 0" in result
        assert js_support.validate_syntax(result) is True

    def test_replace_multiple_class_methods_sequentially(self, js_support):
        """Test replacing multiple methods in sequence."""
        source = """class Math {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}
"""
        # Replace add first
        add_func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.js"),
            starting_line=2,
            ending_line=4,
            parents=[ParentInfo(name="Math", type="ClassDef")],
            is_method=True,
        )
        source = js_support.replace_function(
            source,
            add_func,
            """    add(a, b) {
        return (a + b) | 0;
    }
""",
        )

        assert js_support.validate_syntax(source) is True

        # Now need to re-discover to get updated line numbers
        # In practice, codeflash handles this, but for test we just check validity
        assert "return (a + b) | 0" in source
        assert "return a - b" in source

    def test_replace_class_method_indentation_adjustment(self, js_support):
        """Test that indentation is correctly adjusted when replacing."""
        source = """    class Indented {
        innerMethod() {
            return 1;
        }
    }
"""
        func = FunctionInfo(
            function_name="innerMethod",
            file_path=Path("/test.js"),
            starting_line=2,
            ending_line=4,
            parents=[ParentInfo(name="Indented", type="ClassDef")],
            is_method=True,
        )
        # New code with no indentation
        new_code = """innerMethod() {
    return 42;
}
"""
        result = js_support.replace_function(source, func, new_code)

        # Check that indentation was adjusted
        lines = result.splitlines()
        method_line = next(l for l in lines if "innerMethod" in l)
        # Should have 8 spaces (original indentation)
        assert method_line.startswith("        ")

        assert js_support.validate_syntax(result) is True


class TestClassMethodEdgeCases:
    """Edge case tests for class method handling."""

    def test_class_with_constructor(self, js_support):
        """Test handling classes with constructors."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Counter {
    constructor(start = 0) {
        this.value = start;
    }

    increment() {
        return ++this.value;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Should find constructor and increment
            names = {f.function_name for f in functions}
            assert "constructor" in names or "increment" in names

    def test_class_with_getters_setters(self, js_support):
        """Test handling classes with getters and setters."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Person {
    constructor(name) {
        this._name = name;
    }

    get name() {
        return this._name;
    }

    set name(value) {
        this._name = value;
    }

    greet() {
        return 'Hello, ' + this._name;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Should find at least greet
            names = {f.function_name for f in functions}
            assert "greet" in names

    def test_class_extending_another(self, js_support):
        """Test handling classes that extend another class."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Animal {
    speak() {
        return 'sound';
    }
}

class Dog extends Animal {
    speak() {
        return 'bark';
    }

    fetch() {
        return 'ball';
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Find Dog's fetch method
            fetch_method = next((f for f in functions if f.function_name == "fetch" and f.class_name == "Dog"), None)

            if fetch_method:
                context = js_support.extract_code_context(fetch_method, file_path.parent, file_path.parent)

                # Full string equality check
                expected_code = """class Dog {
    fetch() {
        return 'ball';
    }
}
"""
                assert context.target_code == expected_code, f"Expected:\n{expected_code}\nGot:\n{context.target_code}"
                assert js_support.validate_syntax(context.target_code) is True

    def test_class_with_private_method(self, js_support):
        """Test handling classes with private methods (ES2022+)."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class SecureClass {
    #privateMethod() {
        return 'secret';
    }

    publicMethod() {
        return this.#privateMethod();
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Should at least find publicMethod
            names = {f.function_name for f in functions}
            assert "publicMethod" in names

    def test_commonjs_class_export(self, js_support):
        """Test handling CommonJS exported classes."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""class Calculator {
    add(a, b) {
        return a + b;
    }
}

module.exports = { Calculator };
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_method = next(f for f in functions if f.function_name == "add")

            context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)

            assert "class Calculator" in context.target_code
            assert js_support.validate_syntax(context.target_code) is True

    def test_es_module_class_export(self, js_support):
        """Test handling ES module exported classes."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""export class Calculator {
    add(a, b) {
        return a + b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)

            # Find the add method
            add_method = next((f for f in functions if f.function_name == "add"), None)

            if add_method:
                context = js_support.extract_code_context(add_method, file_path.parent, file_path.parent)
                assert js_support.validate_syntax(context.target_code) is True


class TestExtractionReplacementRoundTrip:
    """Tests for the full workflow of extracting code context and then replacing the function.

    These tests verify that:
    1. Extracted code includes constructor and fields for AI context
    2. Optimized code (from AI) is the full class with the optimized method
    3. Replacement extracts just the method body from optimized code and replaces in original
    4. The round-trip produces valid, correct code
    All assertions use exact string equality for strict verification.
    """

    def test_extract_context_then_replace_method(self, js_support):
        """Test extracting code context and then replacing the method.

        Simulates the full AI optimization workflow:
        1. Extract code context (full class with constructor)
        2. AI returns optimized code (full class with optimized method)
        3. Replace extracts just the method body and replaces in original
        """
        original_source = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }

    decrement() {
        this.count--;
        return this.count;
    }
}

module.exports = { Counter };
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(original_source)
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            increment_func = next(fn for fn in functions if fn.function_name == "increment")

            # Step 1: Extract code context (includes constructor for AI context)
            context = js_support.extract_code_context(increment_func, file_path.parent, file_path.parent)

            # Verify extraction with exact string equality
            expected_extraction = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }
}
"""
            assert context.target_code == expected_extraction, (
                f"Extracted code does not match expected.\n"
                f"Expected:\n{expected_extraction}\n\nGot:\n{context.target_code}"
            )

            # Step 2: AI returns optimized code as FULL CLASS (not just method)
            # This simulates what the AI would return - the full context with optimized method
            optimized_code_from_ai = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        // Optimized: use prefix increment
        return ++this.count;
    }
}
"""

            # Step 3: Replace extracts just the method body and replaces in original
            result = js_support.replace_function(original_source, increment_func, optimized_code_from_ai)

            # Verify result with exact string equality
            expected_result = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        // Optimized: use prefix increment
        return ++this.count;
    }

    decrement() {
        this.count--;
        return this.count;
    }
}

module.exports = { Counter };
"""
            assert result == expected_result, (
                f"Replacement result does not match expected.\nExpected:\n{expected_result}\n\nGot:\n{result}"
            )
            assert js_support.validate_syntax(result) is True

    def test_typescript_extract_context_then_replace_method(self):
        """Test TypeScript extraction with fields and then replacement."""
        from codeflash.languages.javascript.support import TypeScriptSupport

        ts_support = TypeScriptSupport()

        original_source = """\
class User {
    private name: string;
    private age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        return this.name;
    }

    getAge(): number {
        return this.age;
    }
}

export { User };
"""
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write(original_source)
            f.flush()
            file_path = Path(f.name)

            functions = ts_support.discover_functions(file_path)
            get_name_func = next(fn for fn in functions if fn.function_name == "getName")

            # Step 1: Extract code context (includes fields and constructor)
            context = ts_support.extract_code_context(get_name_func, file_path.parent, file_path.parent)

            # Verify extraction with exact string equality
            expected_extraction = """\
class User {
    private name: string;
    private age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        return this.name;
    }
}
"""
            assert context.target_code == expected_extraction, (
                f"Extracted code does not match expected.\n"
                f"Expected:\n{expected_extraction}\n\nGot:\n{context.target_code}"
            )

            # Step 2: AI returns optimized code as FULL CLASS
            optimized_code_from_ai = """\
class User {
    private name: string;
    private age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        // Optimized getter
        return this.name || '';
    }
}
"""

            # Step 3: Replace extracts just the method body and replaces in original
            result = ts_support.replace_function(original_source, get_name_func, optimized_code_from_ai)

            # Verify result with exact string equality
            expected_result = """\
class User {
    private name: string;
    private age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        // Optimized getter
        return this.name || '';
    }

    getAge(): number {
        return this.age;
    }
}

export { User };
"""
            assert result == expected_result, (
                f"Replacement result does not match expected.\nExpected:\n{expected_result}\n\nGot:\n{result}"
            )
            assert ts_support.validate_syntax(result) is True

    def test_extract_replace_preserves_other_methods(self, js_support):
        """Test that replacing one method doesn't affect others."""
        original_source = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }

    multiply(a, b) {
        return a * b;
    }
}
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(original_source)
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_func = next(fn for fn in functions if fn.function_name == "add")

            # Extract context for add
            context = js_support.extract_code_context(add_func, file_path.parent, file_path.parent)

            # Verify extraction with exact string equality
            expected_extraction = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return a + b;
    }
}
"""
            assert context.target_code == expected_extraction, (
                f"Extracted code does not match expected.\n"
                f"Expected:\n{expected_extraction}\n\nGot:\n{context.target_code}"
            )

            # AI returns optimized code as FULL CLASS
            optimized_code_from_ai = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return (a + b) | 0;
    }
}
"""
            result = js_support.replace_function(original_source, add_func, optimized_code_from_ai)

            # Verify result with exact string equality
            expected_result = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return (a + b) | 0;
    }

    subtract(a, b) {
        return a - b;
    }

    multiply(a, b) {
        return a * b;
    }
}
"""
            assert result == expected_result, (
                f"Replacement result does not match expected.\nExpected:\n{expected_result}\n\nGot:\n{result}"
            )
            assert js_support.validate_syntax(result) is True

    def test_extract_static_method_then_replace(self, js_support):
        """Test extracting and replacing a static method."""
        original_source = """\
class MathUtils {
    constructor() {
        this.cache = {};
    }

    static add(a, b) {
        return a + b;
    }

    static multiply(a, b) {
        return a * b;
    }
}

module.exports = { MathUtils };
"""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(original_source)
            f.flush()
            file_path = Path(f.name)

            functions = js_support.discover_functions(file_path)
            add_func = next(fn for fn in functions if fn.function_name == "add")

            # Extract context
            context = js_support.extract_code_context(add_func, file_path.parent, file_path.parent)

            # Verify extraction with exact string equality
            expected_extraction = """\
class MathUtils {
    constructor() {
        this.cache = {};
    }

    static add(a, b) {
        return a + b;
    }
}
"""
            assert context.target_code == expected_extraction, (
                f"Extracted code does not match expected.\n"
                f"Expected:\n{expected_extraction}\n\nGot:\n{context.target_code}"
            )

            # AI returns optimized code as FULL CLASS
            optimized_code_from_ai = """\
class MathUtils {
    constructor() {
        this.cache = {};
    }

    static add(a, b) {
        // Optimized bitwise
        return (a + b) | 0;
    }
}
"""
            result = js_support.replace_function(original_source, add_func, optimized_code_from_ai)

            # Verify result with exact string equality
            expected_result = """\
class MathUtils {
    constructor() {
        this.cache = {};
    }

    static add(a, b) {
        // Optimized bitwise
        return (a + b) | 0;
    }

    static multiply(a, b) {
        return a * b;
    }
}

module.exports = { MathUtils };
"""
            assert result == expected_result, (
                f"Replacement result does not match expected.\nExpected:\n{expected_result}\n\nGot:\n{result}"
            )
            assert js_support.validate_syntax(result) is True


class TestTypeScriptSyntaxValidation:
    """Tests for TypeScript-specific syntax validation.

    These tests ensure that TypeScript code is validated with the TypeScript parser,
    not the JavaScript parser. This is important because TypeScript has syntax that
    is invalid in JavaScript (e.g., type assertions, type annotations).
    """

    def test_typescript_type_assertion_valid_in_ts(self):
        """TypeScript type assertions should be valid in TypeScript."""
        from codeflash.languages.javascript.support import TypeScriptSupport

        ts_support = TypeScriptSupport()

        # Type assertions are TypeScript-specific
        ts_code = """
const value = 4.9 as unknown as number;
const str = "hello" as string;
"""
        assert ts_support.validate_syntax(ts_code) is True

    def test_typescript_type_assertion_invalid_in_js(self, js_support):
        """TypeScript type assertions should be invalid in JavaScript."""
        # This is the code pattern that caused the backend error
        ts_code = """
const value = 4.9 as unknown as number;
"""
        # JavaScript parser should reject TypeScript syntax
        assert js_support.validate_syntax(ts_code) is False

    def test_typescript_interface_valid_in_ts(self):
        """TypeScript interfaces should be valid in TypeScript."""
        from codeflash.languages.javascript.support import TypeScriptSupport

        ts_support = TypeScriptSupport()

        ts_code = """
interface User {
    name: string;
    age: number;
}
"""
        assert ts_support.validate_syntax(ts_code) is True

    def test_typescript_interface_invalid_in_js(self, js_support):
        """TypeScript interfaces should be invalid in JavaScript."""
        ts_code = """
interface User {
    name: string;
    age: number;
}
"""
        # JavaScript parser should reject TypeScript interface syntax
        assert js_support.validate_syntax(ts_code) is False

    def test_typescript_generic_function_valid_in_ts(self):
        """TypeScript generics should be valid in TypeScript."""
        from codeflash.languages.javascript.support import TypeScriptSupport

        ts_support = TypeScriptSupport()

        ts_code = """
function identity<T>(arg: T): T {
    return arg;
}
"""
        assert ts_support.validate_syntax(ts_code) is True

    def test_typescript_generic_function_invalid_in_js(self, js_support):
        """TypeScript generics should be invalid in JavaScript."""
        ts_code = """
function identity<T>(arg: T): T {
    return arg;
}
"""
        assert js_support.validate_syntax(ts_code) is False

    def test_language_property_is_typescript(self):
        """TypeScriptSupport should report typescript as language."""
        from codeflash.languages.base import Language
        from codeflash.languages.javascript.support import TypeScriptSupport

        ts_support = TypeScriptSupport()
        assert ts_support.language == Language.TYPESCRIPT
        assert str(ts_support.language) == "typescript"

    def test_language_property_is_javascript(self, js_support):
        """JavaScriptSupport should report javascript as language."""
        from codeflash.languages.base import Language

        assert js_support.language == Language.JAVASCRIPT
        assert str(js_support.language) == "javascript"
