"""
Tests for the integrated multi-language function discovery.

These tests verify that the function discovery in functions_to_optimize.py
correctly routes to language-specific implementations.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import (
    FunctionToOptimize,
    find_all_functions_in_file,
    get_all_files_and_functions,
    get_files_for_language,
)
from codeflash.languages.base import Language


class TestGetFilesForLanguage:
    """Tests for get_files_for_language helper."""

    def test_get_python_files_only(self, tmp_path):
        """Test getting only Python files."""
        # Create test files
        (tmp_path / "test.py").write_text("x = 1")
        (tmp_path / "test.js").write_text("const x = 1;")
        (tmp_path / "test.txt").write_text("hello")

        files = get_files_for_language(tmp_path, Language.PYTHON)
        names = {f.name for f in files}

        assert "test.py" in names
        assert "test.js" not in names
        assert "test.txt" not in names

    def test_get_javascript_files_only(self, tmp_path):
        """Test getting only JavaScript files."""
        (tmp_path / "test.py").write_text("x = 1")
        (tmp_path / "test.js").write_text("const x = 1;")
        (tmp_path / "test.jsx").write_text("const App = () => <div/>;")

        files = get_files_for_language(tmp_path, Language.JAVASCRIPT)
        names = {f.name for f in files}

        assert "test.py" not in names
        assert "test.js" in names
        assert "test.jsx" in names

    def test_get_all_supported_files(self, tmp_path):
        """Test getting all supported language files."""
        (tmp_path / "test.py").write_text("x = 1")
        (tmp_path / "test.js").write_text("const x = 1;")
        (tmp_path / "test.txt").write_text("hello")

        files = get_files_for_language(tmp_path, language=None)
        names = {f.name for f in files}

        assert "test.py" in names
        assert "test.js" in names
        assert "test.txt" not in names


class TestFindAllFunctionsInFile:
    """Tests for find_all_functions_in_file routing."""

    def test_python_file_routes_to_python_handler(self):
        """Test that Python files use the Python handler."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions.get(file_path, [])) == 2
            names = {fn.function_name for fn in functions[file_path]}
            assert names == {"add", "multiply"}

            # All should have language="python"
            for fn in functions[file_path]:
                assert fn.language == "python"

    def test_javascript_file_routes_to_js_handler(self):
        """Test that JavaScript files use the JavaScript handler."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions.get(file_path, [])) == 2
            names = {fn.function_name for fn in functions[file_path]}
            assert names == {"add", "multiply"}

            # All should have language="javascript"
            for fn in functions[file_path]:
                assert fn.language == "javascript"

    def test_unsupported_file_returns_empty(self):
        """Test that unsupported file extensions return empty."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("this is not code")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)
            assert functions == {}

    def test_function_to_optimize_has_correct_fields(self):
        """Test that FunctionToOptimize has all required fields populated."""
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
class Calculator {
    add(a, b) {
        return a + b;
    }
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)
            assert len(functions.get(file_path, [])) == 1

            fn = functions[file_path][0]
            assert fn.function_name == "add"
            assert fn.file_path == file_path
            assert fn.starting_line is not None
            assert fn.ending_line is not None
            assert fn.language == "javascript"
            assert len(fn.parents) == 1
            assert fn.parents[0].name == "Calculator"


class TestGetAllFilesAndFunctions:
    """Tests for get_all_files_and_functions with multi-language support."""

    def test_discovers_python_files_by_default(self, tmp_path):
        """Test that Python files are discovered by default."""
        (tmp_path / "module.py").write_text("""
def add(a, b):
    return a + b
""")

        functions = get_all_files_and_functions(tmp_path)
        assert len(functions) == 1

    def test_discovers_javascript_files_when_specified(self, tmp_path):
        """Test that JavaScript files are discovered when language is specified."""
        (tmp_path / "module.js").write_text("""
function add(a, b) {
    return a + b;
}
""")

        functions = get_all_files_and_functions(tmp_path, language=Language.JAVASCRIPT)
        assert len(functions) == 1

    def test_discovers_both_languages_when_none_specified(self, tmp_path):
        """Test that both Python and JavaScript files are discovered when no language specified."""
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

        # Should find both files
        assert len(functions) == 2

        # Check we have both Python and JavaScript functions
        all_funcs = []
        for funcs in functions.values():
            all_funcs.extend(funcs)

        languages = {fn.language for fn in all_funcs}
        assert "python" in languages
        assert "javascript" in languages


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing Python code."""

    def test_python_functions_detected_correctly(self):
        """Test that Python functions are correctly detected."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""def first():
    return 1

def second():
    x = 1
    return x
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            # Should find both functions
            assert len(functions[file_path]) == 2
            names = {fn.function_name for fn in functions[file_path]}
            assert names == {"first", "second"}

            # All should have language="python"
            for fn in functions[file_path]:
                assert fn.language == "python"

    def test_python_class_methods_detected(self):
        """Test that Python class methods are correctly detected."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
class MyClass:
    def method(self):
        return 1
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions[file_path]) == 1
            fn = functions[file_path][0]
            assert fn.function_name == "method"
            assert len(fn.parents) == 1
            assert fn.parents[0].name == "MyClass"

    def test_python_async_functions_detected(self):
        """Test that Python async functions are correctly detected."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
async def async_func():
    return 1
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions[file_path]) == 1
            fn = functions[file_path][0]
            assert fn.function_name == "async_func"
            assert fn.is_async is True

    def test_functions_without_return_excluded(self):
        """Test that functions without return statements are excluded."""
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
def with_return():
    return 1

def without_return():
    print("hello")
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions[file_path]) == 1
            assert functions[file_path][0].function_name == "with_return"
