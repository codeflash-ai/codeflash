"""End-to-end integration tests for JavaScript pipeline.

Tests the full optimization pipeline for JavaScript:
- Function discovery
- Code context extraction
- Test discovery
- Code replacement
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.discovery.functions_to_optimize import find_all_functions_in_file, get_files_for_language
from codeflash.languages.base import Language


class TestJavaScriptFunctionDiscovery:
    """Tests for JavaScript function discovery in the main pipeline."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_discover_functions_in_fibonacci(self, js_project_dir):
        """Test discovering functions in fibonacci.js."""
        fib_file = js_project_dir / "fibonacci.js"
        if not fib_file.exists():
            pytest.skip("fibonacci.js not found")

        functions = find_all_functions_in_file(fib_file)

        assert fib_file in functions
        func_list = functions[fib_file]

        # Should find the main exported functions
        func_names = {f.function_name for f in func_list}
        assert "fibonacci" in func_names
        assert "isFibonacci" in func_names
        assert "isPerfectSquare" in func_names
        assert "fibonacciSequence" in func_names

        # All should be JavaScript functions
        for func in func_list:
            assert func.language == "javascript"

    def test_discover_functions_in_bubble_sort(self, js_project_dir):
        """Test discovering functions in bubble_sort.js."""
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
        files = get_files_for_language(js_project_dir, Language.JAVASCRIPT)

        # Should find .js files
        js_files = [f for f in files if f.suffix == ".js"]
        assert len(js_files) >= 3  # fibonacci.js, bubble_sort.js, string_utils.js

        # Should not include test files in root (they're in tests/)
        root_files = [f for f in js_files if f.parent == js_project_dir]
        assert len(root_files) >= 3


class TestJavaScriptCodeContext:
    """Tests for JavaScript code context extraction."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_extract_code_context_for_javascript(self, js_project_dir):
        """Test extracting code context for a JavaScript function."""
        from codeflash.context.code_context_extractor import get_code_optimization_context

        fib_file = js_project_dir / "fibonacci.js"
        if not fib_file.exists():
            pytest.skip("fibonacci.js not found")

        functions = find_all_functions_in_file(fib_file)
        func_list = functions[fib_file]

        # Find the fibonacci function
        fib_func = next((f for f in func_list if f.function_name == "fibonacci"), None)
        assert fib_func is not None

        # Extract code context
        context = get_code_optimization_context(fib_func, js_project_dir)

        # Verify context structure
        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "javascript"
        assert len(context.read_writable_code.code_strings) > 0

        # The code should contain the function
        code = context.read_writable_code.code_strings[0].code
        assert "fibonacci" in code


class TestJavaScriptCodeReplacement:
    """Tests for JavaScript code replacement."""

    def test_replace_function_in_javascript_file(self):
        """Test replacing a function in a JavaScript file."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo, Language

        original_source = """
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}
"""

        new_function = """function add(a, b) {
    // Optimized version
    return a + b;
}"""

        js_support = get_language_support(Language.JAVASCRIPT)

        # Create FunctionInfo for the add function
        func_info = FunctionInfo(
            name="add", file_path=Path("/tmp/test.js"), start_line=2, end_line=4, language=Language.JAVASCRIPT
        )

        result = js_support.replace_function(original_source, func_info, new_function)

        # Verify the function was replaced
        assert "// Optimized version" in result
        assert "multiply" in result  # Other function should still be there


class TestJavaScriptTestDiscovery:
    """Tests for JavaScript test discovery."""

    @pytest.fixture
    def js_project_dir(self):
        """Get the JavaScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        js_dir = project_root / "code_to_optimize_js"
        if not js_dir.exists():
            pytest.skip("code_to_optimize_js directory not found")
        return js_dir

    def test_discover_jest_tests(self, js_project_dir):
        """Test discovering Jest tests for JavaScript functions."""
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo, Language

        js_support = get_language_support(Language.JAVASCRIPT)
        test_root = js_project_dir / "tests"

        if not test_root.exists():
            pytest.skip("tests directory not found")

        # Create FunctionInfo for fibonacci function
        fib_file = js_project_dir / "fibonacci.js"
        func_info = FunctionInfo(
            name="fibonacci", file_path=fib_file, start_line=11, end_line=16, language=Language.JAVASCRIPT
        )

        # Discover tests
        tests = js_support.discover_tests(test_root, [func_info])

        # Should find tests for fibonacci
        assert func_info.qualified_name in tests or "fibonacci" in str(tests)


class TestJavaScriptPipelineIntegration:
    """Integration tests for the full JavaScript pipeline."""

    def test_function_to_optimize_has_correct_fields(self):
        """Test that FunctionToOptimize from JavaScript has all required fields."""
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

function standalone(x) {
    return x * 2;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            # Should find class methods and standalone function
            assert len(functions.get(file_path, [])) >= 3

            # Check standalone function
            standalone_fn = next((fn for fn in functions[file_path] if fn.function_name == "standalone"), None)
            assert standalone_fn is not None
            assert standalone_fn.language == "javascript"
            assert len(standalone_fn.parents) == 0

            # Check class method
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
        assert "```javascript" in markdown or "```js" in markdown.lower()
