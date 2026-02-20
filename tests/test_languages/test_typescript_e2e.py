"""End-to-end integration tests for TypeScript pipeline.

Tests the full optimization pipeline for TypeScript:
- Function discovery
- Code context extraction
- Test discovery
- Code replacement
- Syntax validation with TypeScript parser (not JavaScript)

This is the TypeScript equivalent of test_javascript_e2e.py.
Ensures parity between JavaScript and TypeScript support.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.base import Language


def skip_if_ts_not_supported():
    """Skip test if TypeScript language is not supported."""
    try:
        from codeflash.languages import get_language_support

        get_language_support(Language.TYPESCRIPT)
    except Exception as e:
        pytest.skip(f"TypeScript language support not available: {e}")


class TestTypeScriptFunctionDiscovery:
    """Tests for TypeScript function discovery in the main pipeline."""

    @pytest.fixture
    def ts_project_dir(self):
        """Get the TypeScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        ts_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not ts_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return ts_dir

    def test_discover_functions_in_typescript_file(self, ts_project_dir):
        """Test discovering functions in a TypeScript file."""
        skip_if_ts_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        fib_file = ts_project_dir / "fibonacci.ts"
        if not fib_file.exists():
            pytest.skip("fibonacci.ts not found")

        functions = find_all_functions_in_file(fib_file)

        assert fib_file in functions
        func_list = functions[fib_file]

        func_names = {f.function_name for f in func_list}
        assert "fibonacci" in func_names

        # Critical: Verify language is "typescript", not "javascript"
        for func in func_list:
            assert func.language == "typescript", \
                f"Function {func.function_name} should have language='typescript', got '{func.language}'"

    def test_discover_functions_with_type_annotations(self):
        """Test discovering TypeScript functions with type annotations."""
        skip_if_ts_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write(r"""
export function add(a: number, b: number): number {
    return a + b;
}

export function greet(name: string): string {
    return `Hello, \${name}!`;
}

interface User {
    name: string;
    age: number;
}

export function getUserAge(user: User): number {
    return user.age;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions.get(file_path, [])) == 3

            for func in functions[file_path]:
                assert func.language == "typescript"

    def test_get_typescript_files(self, ts_project_dir):
        """Test getting TypeScript files from directory."""
        skip_if_ts_not_supported()
        from codeflash.discovery.functions_to_optimize import get_files_for_language

        files = get_files_for_language(ts_project_dir, Language.TYPESCRIPT)

        ts_files = [f for f in files if f.suffix == ".ts" and "test" not in f.name]
        assert len(ts_files) >= 1


class TestTypeScriptCodeContext:
    """Tests for TypeScript code context extraction."""

    @pytest.fixture
    def ts_project_dir(self):
        """Get the TypeScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        ts_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not ts_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return ts_dir

    def test_extract_code_context_for_typescript(self, ts_project_dir):
        """Test extracting code context for a TypeScript function."""
        skip_if_ts_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current
        from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context

        lang_current._current_language = Language.TYPESCRIPT

        fib_file = ts_project_dir / "fibonacci.ts"
        if not fib_file.exists():
            pytest.skip("fibonacci.ts not found")

        functions = find_all_functions_in_file(fib_file)
        func_list = functions[fib_file]

        fib_func = next((f for f in func_list if f.function_name == "fibonacci"), None)
        assert fib_func is not None

        context = get_code_optimization_context(fib_func, ts_project_dir)

        assert context.read_writable_code is not None
        # Critical: language should be "typescript", not "javascript"
        assert context.read_writable_code.language == "typescript"
        assert len(context.read_writable_code.code_strings) > 0


class TestTypeScriptCodeReplacement:
    """Tests for TypeScript code replacement."""

    def test_replace_function_in_typescript_file(self):
        """Test replacing a function in a TypeScript file."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo

        original_source = """
function add(a: number, b: number): number {
    return a + b;
}

function multiply(a: number, b: number): number {
    return a * b;
}
"""

        new_function = """function add(a: number, b: number): number {
    // Optimized version
    return a + b;
}"""

        ts_support = get_language_support(Language.TYPESCRIPT)

        func_info = FunctionInfo(
            function_name="add",
            file_path=Path("/tmp/test.ts"),
            starting_line=2,
            ending_line=4,
            language="typescript"
        )

        result = ts_support.replace_function(original_source, func_info, new_function)

        expected_result = """
function add(a: number, b: number): number {
    // Optimized version
    return a + b;
}

function multiply(a: number, b: number): number {
    return a * b;
}
"""
        assert result == expected_result

    def test_replace_function_preserves_types(self):
        """Test that replacing a function preserves TypeScript type annotations."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo

        original_source = r"""
interface Config {
    timeout: number;
    retries: number;
}

function processConfig(config: Config): string {
    return `timeout=\${config.timeout}, retries=\${config.retries}`;
}
"""

        new_function = r"""function processConfig(config: Config): string {
    // Optimized with template caching
    const { timeout, retries } = config;
    return `timeout=\${timeout}, retries=\${retries}`;
}"""

        ts_support = get_language_support(Language.TYPESCRIPT)

        func_info = FunctionInfo(
            function_name="processConfig",
            file_path=Path("/tmp/test.ts"),
            starting_line=7,
            ending_line=9,
            language="typescript"
        )

        result = ts_support.replace_function(original_source, func_info, new_function)

        # Verify type annotations are preserved
        assert "config: Config" in result
        assert ": string" in result
        assert "interface Config" in result


class TestTypeScriptTestDiscovery:
    """Tests for TypeScript test discovery."""

    @pytest.fixture
    def ts_project_dir(self):
        """Get the TypeScript sample project directory."""
        project_root = Path(__file__).parent.parent.parent
        ts_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"
        if not ts_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")
        return ts_dir

    def test_discover_vitest_tests_for_typescript(self, ts_project_dir):
        """Test discovering Vitest tests for TypeScript functions."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support
        from codeflash.languages.base import FunctionInfo

        ts_support = get_language_support(Language.TYPESCRIPT)
        test_root = ts_project_dir / "tests"

        if not test_root.exists():
            pytest.skip("tests directory not found")

        fib_file = ts_project_dir / "fibonacci.ts"
        func_info = FunctionInfo(
            function_name="fibonacci",
            file_path=fib_file,
            starting_line=1,
            ending_line=7,
            language="typescript"
        )

        tests = ts_support.discover_tests(test_root, [func_info])

        # Should find tests for the fibonacci function
        assert func_info.qualified_name in tests or len(tests) > 0


class TestTypeScriptPipelineIntegration:
    """Integration tests for the full TypeScript pipeline."""

    def test_function_to_optimize_has_correct_fields(self):
        """Test that FunctionToOptimize from TypeScript has all required fields."""
        skip_if_ts_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False) as f:
            f.write("""
export class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }

    subtract(a: number, b: number): number {
        return a - b;
    }
}

export function standalone(x: number): number {
    return x * 2;
}
""")
            f.flush()
            file_path = Path(f.name)

            functions = find_all_functions_in_file(file_path)

            assert len(functions.get(file_path, [])) >= 3

            standalone_fn = next((fn for fn in functions[file_path] if fn.function_name == "standalone"), None)
            assert standalone_fn is not None
            assert standalone_fn.language == "typescript"
            assert len(standalone_fn.parents) == 0

            add_fn = next((fn for fn in functions[file_path] if fn.function_name == "add"), None)
            assert add_fn is not None
            assert add_fn.language == "typescript"
            assert len(add_fn.parents) == 1
            assert add_fn.parents[0].name == "Calculator"

    def test_code_strings_markdown_uses_typescript_tag(self):
        """Test that CodeStringsMarkdown uses typescript for code blocks."""
        from codeflash.models.models import CodeString, CodeStringsMarkdown

        code_strings = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code="function add(a: number, b: number): number { return a + b; }",
                    file_path=Path("test.ts"),
                    language="typescript"
                )
            ],
            language="typescript",
        )

        markdown = code_strings.markdown
        assert "```typescript" in markdown


class TestTypeScriptSyntaxValidation:
    """Tests for TypeScript-specific syntax validation.

    These tests ensure TypeScript code is validated with the TypeScript parser,
    not the JavaScript parser. This was the root cause of production issues.
    """

    def test_typescript_type_assertion_valid(self):
        """TypeScript type assertions should be valid."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        ts_support = get_language_support(Language.TYPESCRIPT)

        # This is TypeScript-specific syntax that should pass
        code = "const value = 4.9 as unknown as number;"
        assert ts_support.validate_syntax(code) is True

    def test_typescript_type_assertion_invalid_in_javascript(self):
        """TypeScript type assertions should be INVALID in JavaScript.

        This test would have caught the production bug where TypeScript code
        was being validated with the JavaScript parser.
        """
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        js_support = get_language_support(Language.JAVASCRIPT)

        # This TypeScript syntax should FAIL JavaScript validation
        code = "const value = 4.9 as unknown as number;"
        assert js_support.validate_syntax(code) is False

    def test_typescript_interface_valid(self):
        """TypeScript interfaces should be valid."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        ts_support = get_language_support(Language.TYPESCRIPT)

        code = """
interface User {
    name: string;
    age: number;
}
"""
        assert ts_support.validate_syntax(code) is True

    def test_typescript_interface_invalid_in_javascript(self):
        """TypeScript interfaces should be INVALID in JavaScript."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        js_support = get_language_support(Language.JAVASCRIPT)

        code = """
interface User {
    name: string;
    age: number;
}
"""
        assert js_support.validate_syntax(code) is False

    def test_typescript_generic_function_valid(self):
        """TypeScript generics should be valid."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        ts_support = get_language_support(Language.TYPESCRIPT)

        code = "function identity<T>(arg: T): T { return arg; }"
        assert ts_support.validate_syntax(code) is True

    def test_typescript_generic_function_invalid_in_javascript(self):
        """TypeScript generics should be INVALID in JavaScript."""
        skip_if_ts_not_supported()
        from codeflash.languages import get_language_support

        js_support = get_language_support(Language.JAVASCRIPT)

        code = "function identity<T>(arg: T): T { return arg; }"
        assert js_support.validate_syntax(code) is False


class TestTypeScriptCodeStringValidation:
    """Tests for CodeString validation with TypeScript."""

    def test_code_string_validates_typescript_with_typescript_parser(self):
        """CodeString with language='typescript' should use TypeScript parser."""
        skip_if_ts_not_supported()
        from codeflash.models.models import CodeString

        # TypeScript-specific syntax should pass when language='typescript'
        ts_code = "const value = 4.9 as unknown as number;"
        cs = CodeString(code=ts_code, language="typescript")
        assert cs.code == ts_code

    def test_code_string_rejects_typescript_with_javascript_parser(self):
        """CodeString with language='javascript' should reject TypeScript syntax."""
        skip_if_ts_not_supported()
        from pydantic import ValidationError

        from codeflash.models.models import CodeString

        # TypeScript-specific syntax should FAIL when language='javascript'
        ts_code = "const value = 4.9 as unknown as number;"
        with pytest.raises(ValidationError):
            CodeString(code=ts_code, language="javascript")
