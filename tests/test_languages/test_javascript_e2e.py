"""End-to-end integration tests for JavaScript pipeline.

Tests the full optimization pipeline for JavaScript:
- Function discovery
- Code context extraction
- Test discovery
- Code replacement

Note: These tests require JS/TS language support to be registered.
They will be skipped in environments where only Python is supported.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import Language


def skip_if_js_not_supported():
    """Skip test if JavaScript/TypeScript languages are not supported."""
    try:
        from codeflash.languages import get_language_support

        get_language_support(Language.JAVASCRIPT)
    except Exception as e:
        pytest.skip(f"JavaScript/TypeScript language support not available: {e}")


class TestJavaScriptFunctionDiscovery:
    """Tests for JavaScript function discovery in the main pipeline."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_discover_functions_in_fibonacci(self, js_project_dir):
        """Test discovering functions in fibonacci.js."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        fib_file = js_project_dir / "fibonacci.js"
        if not fib_file.exists():
            pytest.skip("fibonacci.js not found")

        functions = find_all_functions_in_file(fib_file)

        assert fib_file in functions
        func_list = functions[fib_file]

        func_names = {f.function_name for f in func_list}
        assert func_names == {"fibonacci", "isFibonacci", "isPerfectSquare", "fibonacciSequence"}

        for func in func_list:
            assert func.language == "javascript"

    def test_discover_functions_in_bubble_sort(self, js_project_dir):
        """Test discovering functions in bubble_sort.js."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        sort_file = js_project_dir / "bubble_sort.js"
        if not sort_file.exists():
            pytest.skip("bubble_sort.js not found")

        functions = find_all_functions_in_file(sort_file)

        assert sort_file in functions
        func_list = functions[sort_file]

        func_names = {f.function_name for f in func_list}
        assert "bubbleSort" in func_names

    def test_get_javascript_files(self, js_project_dir):
        """Test getting JavaScript files from directory."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import get_files_for_language

        files = get_files_for_language(js_project_dir, Language.JAVASCRIPT)

        js_files = [f for f in files if f.suffix == ".js"]
        assert len(js_files) >= 3

        root_files = [f for f in js_files if f.parent == js_project_dir]
        assert len(root_files) >= 3


class TestJavaScriptCodeContext:
    """Tests for JavaScript code context extraction."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_extract_code_context_for_javascript(self, js_project_dir):
        """Test extracting code context for a JavaScript function."""
        skip_if_js_not_supported()
        from codeflash.context.code_context_extractor import get_code_optimization_context
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current

        lang_current._current_language = Language.JAVASCRIPT

        fib_file = js_project_dir / "fibonacci.js"
        if not fib_file.exists():
            pytest.skip("fibonacci.js not found")

        functions = find_all_functions_in_file(fib_file)
        func_list = functions[fib_file]

        fib_func = next((f for f in func_list if f.function_name == "fibonacci"), None)
        assert fib_func is not None

        context = get_code_optimization_context(fib_func, js_project_dir)

        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "javascript"
        assert len(context.read_writable_code.code_strings) > 0

        code = context.read_writable_code.code_strings[0].code
        expected_code = """/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} - The nth Fibonacci number
 */
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"""
        assert code == expected_code


class TestJavaScriptCodeReplacement:
    """Tests for JavaScript code replacement."""

    def test_replace_function_in_javascript_file(self):
        """Test replacing a function in a JavaScript file."""
        skip_if_js_not_supported()
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo

        original_source = """
export function add(a, b) {
    return a + b;
}

export function multiply(a, b) {
    return a * b;
}
"""

        new_function = """export function add(a, b) {
    // Optimized version
    return a + b;
}"""

        js_support = get_language_support(Language.JAVASCRIPT)

        func_info = FunctionInfo(
            function_name="add", file_path=Path("/tmp/test.js"), starting_line=2, ending_line=4, language="javascript"
        )

        result = js_support.replace_function(original_source, func_info, new_function)

        expected_result = """
export function add(a, b) {
    // Optimized version
    return a + b;
}

export function multiply(a, b) {
    return a * b;
}
"""
        assert result == expected_result


class TestJavaScriptTestDiscovery:
    """Tests for JavaScript test discovery."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_discover_jest_tests(self, js_project_dir):
        """Test discovering Jest tests for JavaScript functions."""
        skip_if_js_not_supported()
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo

        js_support = get_language_support(Language.JAVASCRIPT)
        test_root = js_project_dir / "tests"

        if not test_root.exists():
            pytest.skip("tests directory not found")

        fib_file = js_project_dir / "fibonacci.js"
        func_info = FunctionInfo(
            function_name="fibonacci", file_path=fib_file, starting_line=11, ending_line=16, language="javascript"
        )

        tests = js_support.discover_tests(test_root, [func_info])

        assert func_info.qualified_name in tests or len(tests) > 0


class TestJavaScriptPipelineIntegration:
    """Integration tests for the full JavaScript pipeline."""

    def test_function_to_optimize_has_correct_fields(self):
        """Test that FunctionToOptimize from JavaScript has all required fields."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write("""
export class Calculator {
    add(a, b) {
        return a + b;
    }

    subtract(a, b) {
        return a - b;
    }
}

export function standalone(x) {
    return x * 2;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions.get(file_path, [])) >= 3

            standalone_fn = next((fn for fn in functions[file_path] if fn.function_name == "standalone"), None)
            assert standalone_fn is not None
            assert standalone_fn.language == "javascript"
            assert len(standalone_fn.parents) == 0

            add_fn = next((fn for fn in functions[file_path] if fn.function_name == "add"), None)
            assert add_fn is not None
            assert add_fn.language == "javascript"
            assert len(add_fn.parents) == 1
            assert add_fn.parents[0].name == "Calculator"

    def test_code_strings_markdown_uses_javascript_tag(self):
        """Test that CodeStringsMarkdown uses javascript for code blocks."""
        from codeflash.models.models import CodeString, CodeStringsMarkdown

        code_strings = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code="function add(a, b) { return a + b; }", file_path=Path("test.js"), language="javascript"
                )
            ],
            language="javascript",
        )

        markdown = code_strings.markdown
        assert "```javascript" in markdown
