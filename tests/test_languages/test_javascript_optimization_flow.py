"""End-to-end tests for JavaScript/TypeScript optimization flow.

These tests verify the full optimization pipeline including:
- Test generation (with mocked backend)
- Language parameter propagation
- Syntax validation with correct parser
- Running and parsing tests

This is the JavaScript equivalent of test_instrument_tests.py for Python.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.models.models import CodeString, FunctionParent
from codeflash.verification.verification_utils import TestConfig


def skip_if_js_not_supported():
    """Skip test if JavaScript/TypeScript languages are not supported."""
    try:
        from codeflash.languages import get_language_support

        get_language_support(Language.JAVASCRIPT)
    except Exception as e:
        pytest.skip(f"JavaScript/TypeScript language support not available: {e}")


class TestLanguageParameterPropagation:
    """Tests verifying language parameter is correctly passed through all layers."""

    def test_function_to_optimize_has_correct_language_for_typescript(self, tmp_path):
        """Verify FunctionToOptimize has language='typescript' for .ts files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        ts_file = tmp_path / "utils.ts"
        ts_file.write_text("""
export function add(a: number, b: number): number {
    return a + b;
}
""")

        functions = find_all_functions_in_file(ts_file)
        assert ts_file in functions
        assert len(functions[ts_file]) == 1
        assert functions[ts_file][0].language == "typescript"

    def test_function_to_optimize_has_correct_language_for_javascript(self, tmp_path):
        """Verify FunctionToOptimize has language='javascript' for .js files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        js_file = tmp_path / "utils.js"
        js_file.write_text("""
function add(a, b) {
    return a + b;
}
module.exports = { add };
""")

        functions = find_all_functions_in_file(js_file)
        assert js_file in functions
        assert len(functions[js_file]) == 1
        assert functions[js_file][0].language == "javascript"

    def test_code_context_preserves_language(self, tmp_path):
        """Verify language is preserved in code context extraction."""
        skip_if_js_not_supported()
        from codeflash.context.code_context_extractor import get_code_optimization_context
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current

        lang_current._current_language = Language.TYPESCRIPT

        ts_file = tmp_path / "utils.ts"
        ts_file.write_text("""
export function add(a: number, b: number): number {
    return a + b;
}
""")

        functions = find_all_functions_in_file(ts_file)
        func = functions[ts_file][0]

        context = get_code_optimization_context(func, tmp_path)

        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "typescript"


class TestCodeStringSyntaxValidation:
    """Tests verifying CodeString validates with correct parser based on language."""

    def test_typescript_code_valid_with_typescript_language(self):
        """TypeScript code should pass validation when language='typescript'."""
        skip_if_js_not_supported()

        ts_code = "const value = 4.9 as unknown as number;"
        code_string = CodeString(code=ts_code, language="typescript")
        assert code_string.code == ts_code

    def test_typescript_code_invalid_with_javascript_language(self):
        """TypeScript code should FAIL validation when language='javascript'.

        This is the exact bug that was in production - TypeScript code being
        validated with JavaScript parser.
        """
        skip_if_js_not_supported()
        from pydantic import ValidationError

        ts_code = "const value = 4.9 as unknown as number;"
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)

    def test_typescript_interface_valid_with_typescript_language(self):
        """TypeScript interface should pass validation when language='typescript'."""
        skip_if_js_not_supported()

        ts_code = "interface User { name: string; age: number; }"
        code_string = CodeString(code=ts_code, language="typescript")
        assert code_string.code == ts_code

    def test_typescript_interface_invalid_with_javascript_language(self):
        """TypeScript interface should FAIL validation when language='javascript'."""
        skip_if_js_not_supported()
        from pydantic import ValidationError

        ts_code = "interface User { name: string; age: number; }"
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)


class TestBackendAPIResponseValidation:
    """Tests verifying backend API responses are validated with correct parser."""

    def test_testgen_request_includes_correct_language(self, tmp_path):
        """Verify test generation request includes the correct language parameter."""
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current

        lang_current._current_language = Language.TYPESCRIPT

        ts_file = tmp_path / "utils.ts"
        ts_file.write_text("""
export function add(a: number, b: number): number {
    return a + b;
}
""")

        functions = find_all_functions_in_file(ts_file)
        func = functions[ts_file][0]

        # Verify function has correct language
        assert func.language == "typescript"

        # Mock the AI service request
        ai_client = AiServiceClient()
        with patch.object(ai_client, 'make_ai_service_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_tests": "// test code",
                "instrumented_behavior_tests": "// behavior code",
                "instrumented_perf_tests": "// perf code",
            }
            mock_request.return_value = mock_response

            # Call generate_regression_tests with correct parameters
            ai_client.generate_regression_tests(
                source_code_being_tested="export function add(a: number, b: number): number { return a + b; }",
                function_to_optimize=func,
                helper_function_names=[],
                module_path=ts_file,
                test_module_path=tmp_path / "tests" / "utils.test.ts",
                test_framework="vitest",
                test_timeout=30,
                trace_id="test-trace-id",
                test_index=0,
                language=func.language,  # This is the key - language should be "typescript"
            )

            # Verify the request was made with correct language
            assert mock_request.called, "API request should have been made"
            call_args = mock_request.call_args
            payload = call_args[1].get('payload', call_args[0][1] if len(call_args[0]) > 1 else {})
            assert payload.get('language') == 'typescript', \
                f"Expected language='typescript', got language='{payload.get('language')}'"


class TestFunctionOptimizerForJavaScript:
    """Tests for FunctionOptimizer with JavaScript/TypeScript functions.

    This is the JavaScript equivalent of test_instrument_tests.py tests.
    """

    @pytest.fixture
    def js_project(self, tmp_path):
        """Create a minimal JavaScript project for testing."""
        project = tmp_path / "js_project"
        project.mkdir()

        # Create source file
        src_file = project / "utils.js"
        src_file.write_text("""
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

module.exports = { fibonacci };
""")

        # Create test file
        tests_dir = project / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "utils.test.js"
        test_file.write_text("""
const { fibonacci } = require('../utils');

describe('fibonacci', () => {
    test('returns 0 for n=0', () => {
        expect(fibonacci(0)).toBe(0);
    });

    test('returns 1 for n=1', () => {
        expect(fibonacci(1)).toBe(1);
    });

    test('returns 5 for n=5', () => {
        expect(fibonacci(5)).toBe(5);
    });
});
""")

        # Create package.json
        package_json = project / "package.json"
        package_json.write_text("""
{
    "name": "test-project",
    "devDependencies": {
        "jest": "^29.0.0"
    }
}
""")

        return project

    @pytest.fixture
    def ts_project(self, tmp_path):
        """Create a minimal TypeScript project for testing."""
        project = tmp_path / "ts_project"
        project.mkdir()

        # Create source file
        src_file = project / "utils.ts"
        src_file.write_text("""
export function fibonacci(n: number): number {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
""")

        # Create test file
        tests_dir = project / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "utils.test.ts"
        test_file.write_text("""
import { fibonacci } from '../utils';

describe('fibonacci', () => {
    test('returns 0 for n=0', () => {
        expect(fibonacci(0)).toBe(0);
    });

    test('returns 1 for n=1', () => {
        expect(fibonacci(1)).toBe(1);
    });
});
""")

        # Create package.json
        package_json = project / "package.json"
        package_json.write_text("""
{
    "name": "test-project",
    "devDependencies": {
        "vitest": "^1.0.0"
    }
}
""")

        return project

    def test_function_optimizer_instantiation_javascript(self, js_project):
        """Test FunctionOptimizer can be instantiated for JavaScript."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        src_file = js_project / "utils.js"
        functions = find_all_functions_in_file(src_file)
        func = functions[src_file][0]

        func_to_optimize = FunctionToOptimize(
            function_name=func.function_name,
            file_path=func.file_path,
            parents=[FunctionParent(name=p.name, type=p.type) for p in func.parents],
            starting_line=func.starting_line,
            ending_line=func.ending_line,
            language=func.language,
        )

        test_config = TestConfig(
            tests_root=js_project / "tests",
            tests_project_rootdir=js_project,
            project_root_path=js_project,
            pytest_cmd="jest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        assert optimizer is not None
        assert optimizer.function_to_optimize.language == "javascript"

    def test_function_optimizer_instantiation_typescript(self, ts_project):
        """Test FunctionOptimizer can be instantiated for TypeScript."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        src_file = ts_project / "utils.ts"
        functions = find_all_functions_in_file(src_file)
        func = functions[src_file][0]

        func_to_optimize = FunctionToOptimize(
            function_name=func.function_name,
            file_path=func.file_path,
            parents=[FunctionParent(name=p.name, type=p.type) for p in func.parents],
            starting_line=func.starting_line,
            ending_line=func.ending_line,
            language=func.language,
        )

        test_config = TestConfig(
            tests_root=ts_project / "tests",
            tests_project_rootdir=ts_project,
            project_root_path=ts_project,
            pytest_cmd="vitest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        assert optimizer is not None
        assert optimizer.function_to_optimize.language == "typescript"

    def test_get_code_optimization_context_javascript(self, js_project):
        """Test get_code_optimization_context for JavaScript."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        lang_current._current_language = Language.JAVASCRIPT

        src_file = js_project / "utils.js"
        functions = find_all_functions_in_file(src_file)
        func = functions[src_file][0]

        func_to_optimize = FunctionToOptimize(
            function_name=func.function_name,
            file_path=func.file_path,
            parents=[FunctionParent(name=p.name, type=p.type) for p in func.parents],
            starting_line=func.starting_line,
            ending_line=func.ending_line,
            language=func.language,
        )

        test_config = TestConfig(
            tests_root=js_project / "tests",
            tests_project_rootdir=js_project,
            project_root_path=js_project,
            pytest_cmd="jest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        result = optimizer.get_code_optimization_context()
        context = result.unwrap()

        assert context is not None
        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "javascript"

    def test_get_code_optimization_context_typescript(self, ts_project):
        """Test get_code_optimization_context for TypeScript."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        lang_current._current_language = Language.TYPESCRIPT

        src_file = ts_project / "utils.ts"
        functions = find_all_functions_in_file(src_file)
        func = functions[src_file][0]

        func_to_optimize = FunctionToOptimize(
            function_name=func.function_name,
            file_path=func.file_path,
            parents=[FunctionParent(name=p.name, type=p.type) for p in func.parents],
            starting_line=func.starting_line,
            ending_line=func.ending_line,
            language=func.language,
        )

        test_config = TestConfig(
            tests_root=ts_project / "tests",
            tests_project_rootdir=ts_project,
            project_root_path=ts_project,
            pytest_cmd="vitest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        result = optimizer.get_code_optimization_context()
        context = result.unwrap()

        assert context is not None
        assert context.read_writable_code is not None
        assert context.read_writable_code.language == "typescript"


class TestHelperFunctionLanguageAttribute:
    """Tests for helper function language attribute (import_resolver.py fix)."""

    def test_helper_functions_have_correct_language_javascript(self, tmp_path):
        """Verify helper functions have language='javascript' for .js files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current, get_language_support
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        lang_current._current_language = Language.JAVASCRIPT

        # Create a file with helper functions
        src_file = tmp_path / "main.js"
        src_file.write_text("""
function helper() {
    return 42;
}

function main() {
    return helper() * 2;
}

module.exports = { main };
""")

        functions = find_all_functions_in_file(src_file)
        main_func = next(f for f in functions[src_file] if f.function_name == "main")

        func_to_optimize = FunctionToOptimize(
            function_name=main_func.function_name,
            file_path=main_func.file_path,
            parents=[],
            starting_line=main_func.starting_line,
            ending_line=main_func.ending_line,
            language=main_func.language,
        )

        test_config = TestConfig(
            tests_root=tmp_path,
            tests_project_rootdir=tmp_path,
            project_root_path=tmp_path,
            pytest_cmd="jest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        result = optimizer.get_code_optimization_context()
        context = result.unwrap()

        # Verify main function has correct language
        assert context.read_writable_code.language == "javascript"

    def test_helper_functions_have_correct_language_typescript(self, tmp_path):
        """Verify helper functions have language='typescript' for .ts files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        lang_current._current_language = Language.TYPESCRIPT

        # Create a file with helper functions
        src_file = tmp_path / "main.ts"
        src_file.write_text("""
function helper(): number {
    return 42;
}

export function main(): number {
    return helper() * 2;
}
""")

        functions = find_all_functions_in_file(src_file)
        main_func = next(f for f in functions[src_file] if f.function_name == "main")

        func_to_optimize = FunctionToOptimize(
            function_name=main_func.function_name,
            file_path=main_func.file_path,
            parents=[],
            starting_line=main_func.starting_line,
            ending_line=main_func.ending_line,
            language=main_func.language,
        )

        test_config = TestConfig(
            tests_root=tmp_path,
            tests_project_rootdir=tmp_path,
            project_root_path=tmp_path,
            pytest_cmd="vitest",
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=func_to_optimize,
            test_cfg=test_config,
            aiservice_client=MagicMock(),
        )

        result = optimizer.get_code_optimization_context()
        context = result.unwrap()

        # Verify main function has correct language
        assert context.read_writable_code.language == "typescript"
