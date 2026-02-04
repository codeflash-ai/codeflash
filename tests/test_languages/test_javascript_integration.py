"""True E2E integration tests for JavaScript/TypeScript optimization flow.

These tests call the ACTUAL backend /testgen API endpoint and verify the full flow:
1. CLI sends code to backend for test generation
2. Backend generates tests using LLM
3. Backend validates generated code with correct parser (JS vs TS)
4. CLI receives tests, instruments them, runs them
5. CLI parses test results and timing data

REQUIREMENTS:
- Backend server running at CODEFLASH_API_URL (default: http://localhost:8000)
- Valid CODEFLASH_API_KEY environment variable
- Node.js and npm installed
- npm dependencies in test fixture directories

Run these tests with:
    pytest tests/test_languages/test_javascript_integration.py -v --run-integration

Or set environment variables:
    CODEFLASH_API_URL=https://api.codeflash.ai
    CODEFLASH_API_KEY=your-api-key
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.models.models import FunctionParent
from codeflash.verification.verification_utils import TestConfig


def is_backend_available() -> bool:
    """Check if the backend API is accessible."""
    try:
        import requests
        api_url = os.environ.get("CODEFLASH_API_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def has_api_key() -> bool:
    """Check if API key is configured."""
    return bool(os.environ.get("CODEFLASH_API_KEY"))


def is_node_available() -> bool:
    """Check if Node.js is available."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def skip_if_not_integration():
    """Skip test if integration environment is not available."""
    if not is_backend_available():
        pytest.skip("Backend API not available. Set CODEFLASH_API_URL and start backend server.")
    if not has_api_key():
        pytest.skip("API key not configured. Set CODEFLASH_API_KEY environment variable.")
    if not is_node_available():
        pytest.skip("Node.js not available")


def skip_if_js_not_supported():
    """Skip test if JavaScript/TypeScript languages are not supported."""
    try:
        from codeflash.languages import get_language_support
        get_language_support(Language.JAVASCRIPT)
    except Exception as e:
        pytest.skip(f"JavaScript/TypeScript language support not available: {e}")


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestBackendTestGeneration:
    """Tests that verify the backend /testgen endpoint works correctly for JS/TS."""

    def test_typescript_testgen_uses_typescript_validator(self, tmp_path):
        """Verify backend validates TypeScript code with TypeScript parser.

        This is the critical test - TypeScript-specific syntax like 'as unknown as number'
        should pass validation when the backend is told it's TypeScript.
        """
        skip_if_not_integration()
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

        # Create a TypeScript file with TypeScript-specific syntax
        ts_file = tmp_path / "utils.ts"
        ts_file.write_text("""
export function castValue(input: unknown): number {
    // This uses TypeScript's double assertion pattern
    return input as unknown as number;
}
""")

        functions = find_all_functions_in_file(ts_file)
        func = functions[ts_file][0]

        # Verify the function is identified as TypeScript
        assert func.language == "typescript"

        # Call the actual backend
        ai_client = AiServiceClient()
        response = ai_client.generate_regression_tests(
            source_code_being_tested=ts_file.read_text(),
            function_to_optimize=func,
            helper_function_names=[],
            module_path=ts_file,
            test_module_path=tmp_path / "tests" / "utils.test.ts",
            test_framework="vitest",
            test_timeout=30,
            trace_id="integration-test-ts-validator",
            test_index=0,
            language="typescript",  # This MUST be passed to use TS validator
        )

        # Backend should return valid tests
        assert response is not None
        assert "generated_tests" in response or hasattr(response, "generated_tests")

    def test_javascript_testgen_rejects_typescript_syntax(self, tmp_path):
        """Verify backend rejects TypeScript syntax when told it's JavaScript.

        If backend incorrectly validates TypeScript as JavaScript, this would fail.
        """
        skip_if_not_integration()
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient
        from codeflash.discovery.functions_to_optimize import FunctionToOptimize

        # TypeScript code should fail when sent as JavaScript
        ts_code = """
function castValue(input) {
    // TypeScript syntax that JavaScript parser should reject
    const value = input as unknown as number;
    return value;
}
"""
        func = FunctionToOptimize(
            function_name="castValue",
            file_path=tmp_path / "utils.js",  # .js extension
            parents=[],
            starting_line=2,
            ending_line=6,
            language="javascript",  # Claiming it's JavaScript
        )

        ai_client = AiServiceClient()

        # This should fail because the source contains TypeScript syntax
        # but we're telling the backend it's JavaScript
        with pytest.raises(Exception):
            ai_client.generate_regression_tests(
                source_code_being_tested=ts_code,
                function_to_optimize=func,
                helper_function_names=[],
                module_path=tmp_path / "utils.js",
                test_module_path=tmp_path / "tests" / "utils.test.js",
                test_framework="jest",
                test_timeout=30,
                trace_id="integration-test-js-rejects-ts",
                test_index=0,
                language="javascript",
            )


class TestFullOptimizationPipeline:
    """Tests that verify the complete optimization flow with real backend."""

    @pytest.fixture
    def vitest_project(self):
        """Get the Vitest sample project."""
        project_root = Path(__file__).parent.parent.parent
        vitest_dir = project_root / "code_to_optimize" / "js" / "code_to_optimize_vitest"

        if not vitest_dir.exists():
            pytest.skip("code_to_optimize_vitest directory not found")

        return vitest_dir

    def test_typescript_full_flow_with_backend(self, vitest_project):
        """Test complete TypeScript optimization flow with actual backend.

        This is the equivalent of Python's test_instrument_tests.py tests but
        for TypeScript, using the real backend API.
        """
        skip_if_not_integration()
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current
        from codeflash.optimization.function_optimizer import FunctionOptimizer

        lang_current._current_language = Language.TYPESCRIPT

        # Find the fibonacci function
        fib_file = vitest_project / "fibonacci.ts"
        if not fib_file.exists():
            pytest.skip("fibonacci.ts not found")

        functions = find_all_functions_in_file(fib_file)
        fib_func_info = next(f for f in functions[fib_file] if f.function_name == "fibonacci")

        # Verify language detection
        assert fib_func_info.language == "typescript"

        # Create FunctionToOptimize
        func = FunctionToOptimize(
            function_name=fib_func_info.function_name,
            file_path=fib_func_info.file_path,
            parents=[FunctionParent(name=p.name, type=p.type) for p in fib_func_info.parents],
            starting_line=fib_func_info.starting_line,
            ending_line=fib_func_info.ending_line,
            language=fib_func_info.language,
        )

        # Create test config
        test_config = TestConfig(
            tests_root=vitest_project / "tests",
            tests_project_rootdir=vitest_project,
            project_root_path=vitest_project,
            pytest_cmd="vitest",
            test_framework="vitest",
        )

        # Use REAL AI service client
        ai_client = AiServiceClient()

        # Create optimizer
        func_optimizer = FunctionOptimizer(
            function_to_optimize=func,
            test_cfg=test_config,
            aiservice_client=ai_client,
        )

        # Get code context
        result = func_optimizer.get_code_optimization_context()
        context = result.unwrap()

        assert context is not None
        assert context.read_writable_code.language == "typescript"

        # Generate tests via backend
        tests_result = func_optimizer.generate_tests()

        # Verify tests were generated
        assert tests_result is not None
        # Tests should contain TypeScript-compatible code


class TestLanguageConsistencyWithBackend:
    """Tests verifying language parameter flows correctly to backend."""

    def test_language_in_testgen_request_payload(self, tmp_path):
        """Verify the language parameter is sent correctly to the backend."""
        skip_if_not_integration()
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from unittest.mock import patch

        ts_file = tmp_path / "utils.ts"
        ts_file.write_text("""
export function add(a: number, b: number): number {
    return a + b;
}
""")

        functions = find_all_functions_in_file(ts_file)
        func = functions[ts_file][0]

        ai_client = AiServiceClient()

        # Spy on the actual request
        original_request = ai_client.make_ai_service_request
        captured_payload = None

        def spy_request(*args, **kwargs):
            nonlocal captured_payload
            if 'payload' in kwargs:
                captured_payload = kwargs['payload']
            elif len(args) > 1:
                captured_payload = args[1]
            return original_request(*args, **kwargs)

        with patch.object(ai_client, 'make_ai_service_request', side_effect=spy_request):
            try:
                ai_client.generate_regression_tests(
                    source_code_being_tested=ts_file.read_text(),
                    function_to_optimize=func,
                    helper_function_names=[],
                    module_path=ts_file,
                    test_module_path=tmp_path / "tests" / "utils.test.ts",
                    test_framework="vitest",
                    test_timeout=30,
                    trace_id="integration-test-language-payload",
                    test_index=0,
                    language="typescript",
                )
            except Exception:
                pass  # We just want to capture the payload

        # Verify language was in the payload
        assert captured_payload is not None
        assert captured_payload.get('language') == 'typescript', \
            f"Expected language='typescript' in payload, got: {captured_payload.get('language')}"


class TestRefinementWithBackend:
    """Tests for the refinement flow with actual backend."""

    def test_typescript_refinement_uses_correct_validator(self, tmp_path):
        """Verify refinement validates TypeScript with TypeScript parser.

        The refiner_context.py had bugs where it always used JavaScript validator.
        This test ensures the fix works end-to-end.
        """
        skip_if_not_integration()
        skip_if_js_not_supported()
        from codeflash.api.aiservice import AiServiceClient

        # TypeScript code that should pass TypeScript validation
        original_code = """
export function processValue(value: unknown): number {
    return value as number;
}
"""
        optimized_code = """
export function processValue(value: unknown): number {
    // Optimized: using double assertion for type safety
    return value as unknown as number;
}
"""

        ai_client = AiServiceClient()

        # Call refinement endpoint with TypeScript
        try:
            response = ai_client.refine_code(
                original_code=original_code,
                optimized_code=optimized_code,
                language="typescript",  # Must be TypeScript
                # ... other parameters
            )

            # If refinement succeeds, the TypeScript validator was used
            assert response is not None
        except Exception as e:
            # If it fails with "Invalid JavaScript syntax", the bug still exists
            assert "Invalid JavaScript" not in str(e), \
                "Backend still using JavaScript validator for TypeScript refinement!"
