"""Extensive tests for the Python language support implementation.

These tests verify that PythonSupport correctly discovers functions,
replaces code, and integrates with existing codeflash functionality.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import FunctionFilterCriteria, FunctionInfo, Language, ParentInfo
from codeflash.languages.python.support import PythonSupport


@pytest.fixture
def python_support():
    """Create a PythonSupport instance."""
    return PythonSupport()


class TestPythonSupportProperties:
    """Tests for PythonSupport properties."""

    def test_language(self, python_support):
        """Test language property."""
        assert python_support.language == Language.PYTHON

    def test_file_extensions(self, python_support):
        """Test file_extensions property."""
        extensions = python_support.file_extensions
        assert ".py" in extensions
        assert ".pyw" in extensions

    def test_test_framework(self, python_support):
        """Test test_framework property."""
        assert python_support.test_framework == "pytest"


class TestDiscoverFunctions:
    """Tests for discover_functions method."""

    def test_discover_simple_function(self, python_support):
        """Test discovering a simple function."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def add(a, b):
    return a + b
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            assert len(functions) == 1
            assert functions[0].function_name == "add"
            assert functions[0].language == Language.PYTHON

    def test_discover_multiple_functions(self, python_support):
        """Test discovering multiple functions."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            assert len(functions) == 3
            names = {func.function_name for func in functions}
            assert names == {"add", "subtract", "multiply"}

    def test_discover_function_with_no_return_excluded(self, python_support):
        """Test that functions without return are excluded by default."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def with_return():
    return 1

def without_return():
    print("hello")
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            # Only the function with return should be discovered
            assert len(functions) == 1
            assert functions[0].function_name == "with_return"

    def test_discover_class_methods(self, python_support):
        """Test discovering class methods."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            assert len(functions) == 2
            for func in functions:
                assert func.is_method is True
                assert func.class_name == "Calculator"

    def test_discover_async_functions(self, python_support):
        """Test discovering async functions."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
async def fetch_data(url):
    return await get(url)

def sync_function():
    return 1
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            assert len(functions) == 2

            async_func = next(f for f in functions if f.function_name == "fetch_data")
            sync_func = next(f for f in functions if f.function_name == "sync_function")

            assert async_func.is_async is True
            assert sync_func.is_async is False

    def test_discover_nested_functions(self, python_support):
        """Test that nested functions are excluded — only top-level and class-level functions are discovered."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def outer():
    def inner():
        return 1
    return inner()
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            # Only outer should be discovered; inner is nested and skipped
            assert len(functions) == 1
            assert functions[0].function_name == "outer"

    def test_discover_static_method(self, python_support):
        """Test discovering static methods."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            assert len(functions) == 1
            assert functions[0].function_name == "helper"
            assert functions[0].class_name == "Utils"

    def test_discover_with_filter_exclude_async(self, python_support):
        """Test filtering out async functions."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
async def async_func():
    return 1

def sync_func():
    return 2
""")
            f.flush()

            criteria = FunctionFilterCriteria(include_async=False)
            functions = python_support.discover_functions(Path(f.name), criteria)

            assert len(functions) == 1
            assert functions[0].function_name == "sync_func"

    def test_discover_with_filter_exclude_methods(self, python_support):
        """Test filtering out class methods."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def standalone():
    return 1

class MyClass:
    def method(self):
        return 2
""")
            f.flush()

            criteria = FunctionFilterCriteria(include_methods=False)
            functions = python_support.discover_functions(Path(f.name), criteria)

            assert len(functions) == 1
            assert functions[0].function_name == "standalone"

    def test_discover_line_numbers(self, python_support):
        """Test that line numbers are correctly captured."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""def func1():
    return 1

def func2():
    x = 1
    y = 2
    return x + y
""")
            f.flush()

            functions = python_support.discover_functions(Path(f.name))

            func1 = next(f for f in functions if f.function_name == "func1")
            func2 = next(f for f in functions if f.function_name == "func2")

            assert func1.starting_line == 1
            assert func1.ending_line == 2
            assert func2.starting_line == 4
            assert func2.ending_line == 7

    def test_discover_invalid_file_raises(self, python_support):
        """Test that invalid Python file raises a parse error."""
        from libcst._exceptions import ParserSyntaxError

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("this is not valid python {{{{")
            f.flush()

            with pytest.raises(ParserSyntaxError):
                python_support.discover_functions(Path(f.name))

    def test_discover_nonexistent_file_raises(self, python_support):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            python_support.discover_functions(Path("/nonexistent/file.py"))


class TestReplaceFunction:
    """Tests for replace_function method."""

    def test_replace_simple_function(self, python_support):
        """Test replacing a simple function."""
        source = """def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        func = FunctionInfo(function_name="add", file_path=Path("/test.py"), starting_line=1, ending_line=2)
        new_code = """def add(a, b):
    # Optimized
    return (a + b) | 0
"""
        result = python_support.replace_function(source, func, new_code)

        assert "# Optimized" in result
        assert "return (a + b) | 0" in result
        assert "def multiply" in result

    def test_replace_preserves_surrounding_code(self, python_support):
        """Test that replacement preserves code before and after."""
        source = """# Header comment
import math

def target():
    return 1

def other():
    return 2

# Footer
"""
        func = FunctionInfo(function_name="target", file_path=Path("/test.py"), starting_line=4, ending_line=5)
        new_code = """def target():
    return 42
"""
        result = python_support.replace_function(source, func, new_code)

        assert "# Header comment" in result
        assert "import math" in result
        assert "return 42" in result
        assert "def other" in result
        assert "# Footer" in result

    def test_replace_with_indentation_adjustment(self, python_support):
        """Test that indentation is adjusted correctly."""
        source = """class Calculator:
    def add(self, a, b):
        return a + b
"""
        func = FunctionInfo(
            function_name="add",
            file_path=Path("/test.py"),
            starting_line=2,
            ending_line=3,
            parents=[ParentInfo(name="Calculator", type="ClassDef")],
        )
        # New code has no indentation
        new_code = """def add(self, a, b):
    return (a + b) | 0
"""
        result = python_support.replace_function(source, func, new_code)

        # Check that indentation was added
        lines = result.splitlines()
        method_line = next(l for l in lines if "def add" in l)
        assert method_line.startswith("    ")  # 4 spaces

    def test_replace_first_function(self, python_support):
        """Test replacing the first function in file."""
        source = """def first():
    return 1

def second():
    return 2
"""
        func = FunctionInfo(function_name="first", file_path=Path("/test.py"), starting_line=1, ending_line=2)
        new_code = """def first():
    return 100
"""
        result = python_support.replace_function(source, func, new_code)

        assert "return 100" in result
        assert "return 2" in result

    def test_replace_last_function(self, python_support):
        """Test replacing the last function in file."""
        source = """def first():
    return 1

def last():
    return 999
"""
        func = FunctionInfo(function_name="last", file_path=Path("/test.py"), starting_line=4, ending_line=5)
        new_code = """def last():
    return 1000
"""
        result = python_support.replace_function(source, func, new_code)

        assert "return 1" in result
        assert "return 1000" in result

    def test_replace_only_function(self, python_support):
        """Test replacing the only function in file."""
        source = """def only():
    return 42
"""
        func = FunctionInfo(function_name="only", file_path=Path("/test.py"), starting_line=1, ending_line=2)
        new_code = """def only():
    return 100
"""
        result = python_support.replace_function(source, func, new_code)

        assert "return 100" in result
        assert "return 42" not in result


class TestValidateSyntax:
    """Tests for validate_syntax method."""

    def test_valid_syntax(self, python_support):
        """Test that valid Python syntax passes."""
        valid_code = """
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        assert python_support.validate_syntax(valid_code) is True

    def test_invalid_syntax(self, python_support):
        """Test that invalid Python syntax fails."""
        invalid_code = """
def add(a, b:
    return a + b
"""
        assert python_support.validate_syntax(invalid_code) is False

    def test_empty_string_valid(self, python_support):
        """Test that empty string is valid syntax."""
        assert python_support.validate_syntax("") is True

    def test_syntax_error_types(self, python_support):
        """Test various syntax error types."""
        # Unclosed bracket
        assert python_support.validate_syntax("x = [1, 2, 3") is False

        # Invalid indentation
        assert python_support.validate_syntax("  x = 1") is False

        # Missing colon
        assert python_support.validate_syntax("def foo()\n    pass") is False


class TestNormalizeCode:
    """Tests for normalize_code method."""

    def test_removes_docstrings(self, python_support):
        """Test that docstrings are removed."""
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
        normalized = python_support.normalize_code(code)
        assert '"""Add two numbers."""' not in normalized
        assert "return a + b" in normalized

    def test_preserves_functionality(self, python_support):
        """Test that code functionality is preserved."""
        code = """
def add(a, b):
    # Comment
    return a + b
"""
        normalized = python_support.normalize_code(code)
        # Should still have the function
        assert "def add" in normalized
        assert "return" in normalized


class TestFormatCode:
    """Tests for format_code method."""

    def test_format_basic_code(self, python_support):
        """Test basic code formatting."""
        code = "def add(a,b): return a+b"

        try:
            formatted = python_support.format_code(code)
            # If black is available, should have proper spacing
            assert "def add" in formatted
        except Exception:
            # If black not available, should return original
            assert python_support.format_code(code) == code

    def test_format_already_formatted(self, python_support):
        """Test formatting already formatted code."""
        code = """def add(a, b):
    return a + b
"""
        formatted = python_support.format_code(code)
        assert "def add" in formatted


class TestExtractCodeContext:
    """Tests for extract_code_context method."""

    def test_extract_simple_function(self, python_support):
        """Test extracting context for a simple function."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""def add(a, b):
    return a + b
""")
            f.flush()
            file_path = Path(f.name)

            func = FunctionInfo(function_name="add", file_path=file_path, starting_line=1, ending_line=2)

            context = python_support.extract_code_context(func, file_path.parent, file_path.parent)

            assert "def add" in context.target_code
            assert "return a + b" in context.target_code
            assert context.target_file == file_path
            assert context.language == Language.PYTHON


class TestIntegration:
    """Integration tests for PythonSupport."""

    def test_discover_and_replace_workflow(self, python_support):
        """Test full discover -> replace workflow."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            original_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
            f.write(original_code)
            f.flush()
            file_path = Path(f.name)

            # Discover
            functions = python_support.discover_functions(file_path)
            assert len(functions) == 1
            func = functions[0]
            assert func.function_name == "fibonacci"

            # Replace
            optimized_code = """def fibonacci(n):
    # Memoized version
    memo = {0: 0, 1: 1}
    for i in range(2, n + 1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]
"""
            result = python_support.replace_function(original_code, func, optimized_code)

            # Validate
            assert python_support.validate_syntax(result) is True
            assert "Memoized version" in result
            assert "memo[n]" in result

    def test_multiple_classes_and_functions(self, python_support):
        """Test discovering and working with complex file."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
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
""")
            f.flush()
            file_path = Path(f.name)

            functions = python_support.discover_functions(file_path)

            # Should find 4 functions
            assert len(functions) == 4

            # Check class methods
            calc_methods = [f for f in functions if f.class_name == "Calculator"]
            assert len(calc_methods) == 2

            string_methods = [f for f in functions if f.class_name == "StringUtils"]
            assert len(string_methods) == 1

            standalone_funcs = [f for f in functions if f.class_name is None]
            assert len(standalone_funcs) == 1


# === Tests for find_references method ===
# These tests verify that PythonSupport correctly finds references to functions
# using jedi, including the fix for using function.function_name instead of function.name.


def test_find_references_simple_function(python_support, tmp_path):
    """Test finding references to a simple function.

    This test specifically exercises the code path that was fixed in the
    regression where function.name was used instead of function.function_name.
    """
    from codeflash.models.function_types import FunctionToOptimize

    # Create source file with function definition
    source_file = tmp_path / "utils.py"
    source_file.write_text("""def helper_function(x):
    return x * 2
""")

    # Create a file that imports and uses the function
    consumer_file = tmp_path / "consumer.py"
    consumer_file.write_text("""from utils import helper_function

def process(value):
    return helper_function(value) + 1
""")

    func = FunctionToOptimize(function_name="helper_function", file_path=source_file, starting_line=1, ending_line=2)

    refs = python_support.find_references(func, project_root=tmp_path)

    assert len(refs) >= 1
    ref_files = {str(r.file_path) for r in refs}
    assert any("consumer.py" in f for f in ref_files)


def test_find_references_class_method(python_support, tmp_path):
    """Test finding references to a class method.

    This verifies the class_name attribute is correctly used to disambiguate methods.
    """
    from codeflash.models.function_types import FunctionParent, FunctionToOptimize

    # Create source file with class and method
    source_file = tmp_path / "calculator.py"
    source_file.write_text("""class Calculator:
    def add(self, a, b):
        return a + b
""")

    # Create a file that uses the class method
    consumer_file = tmp_path / "main.py"
    consumer_file.write_text("""from calculator import Calculator

def compute():
    calc = Calculator()
    return calc.add(1, 2)
""")

    func = FunctionToOptimize(
        function_name="add",
        file_path=source_file,
        parents=[FunctionParent(name="Calculator", type="ClassDef")],
        starting_line=2,
        ending_line=3,
        is_method=True,
    )

    refs = python_support.find_references(func, project_root=tmp_path)

    assert len(refs) >= 1
    ref_files = {str(r.file_path) for r in refs}
    assert any("main.py" in f for f in ref_files)


def test_find_references_no_references(python_support, tmp_path):
    """Test that find_references returns empty list when no references exist."""
    from codeflash.models.function_types import FunctionToOptimize

    source_file = tmp_path / "isolated.py"
    source_file.write_text("""def isolated_function():
    return 42
""")

    func = FunctionToOptimize(function_name="isolated_function", file_path=source_file, starting_line=1, ending_line=2)

    refs = python_support.find_references(func, project_root=tmp_path)

    assert refs == []


def test_find_references_nonexistent_function(python_support, tmp_path):
    """Test that find_references handles nonexistent functions gracefully."""
    from codeflash.models.function_types import FunctionToOptimize

    source_file = tmp_path / "source.py"
    source_file.write_text("""def existing_function():
    return 1
""")

    func = FunctionToOptimize(
        function_name="nonexistent_function", file_path=source_file, starting_line=1, ending_line=2
    )

    refs = python_support.find_references(func, project_root=tmp_path)

    assert refs == []
