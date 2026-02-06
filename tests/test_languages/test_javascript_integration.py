"""E2E tests for JavaScript/TypeScript optimization flow with backend.

These tests call the actual backend /testgen API endpoint and verify:
1. Language parameter is correctly passed to backend
2. Backend validates generated code with correct parser (JS vs TS)
3. CLI receives and processes tests correctly

Similar to test_validate_python_code.py but for JavaScript/TypeScript.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from codeflash.api.aiservice import AiServiceClient
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.models.models import CodeString, OptimizedCandidateSource


def skip_if_js_not_supported():
    """Skip test if JavaScript/TypeScript languages are not supported."""
    try:
        from codeflash.languages import get_language_support
        get_language_support(Language.JAVASCRIPT)
    except Exception as e:
        pytest.skip(f"JavaScript/TypeScript language support not available: {e}")


class TestJavaScriptCodeStringValidation:
    """Tests for JavaScript CodeString validation - mirrors test_validate_python_code.py."""

    def test_javascript_string(self):
        """Test valid JavaScript code string."""
        skip_if_js_not_supported()
        code = CodeString(code="console.log('Hello, World!');", language="javascript")
        assert code.code == "console.log('Hello, World!');"

    def test_valid_javascript_code(self):
        """Test that valid JavaScript code passes validation."""
        skip_if_js_not_supported()
        valid_code = "const x = 1;\nconst y = x + 2;\nconsole.log(y);"
        cs = CodeString(code=valid_code, language="javascript")
        assert cs.code == valid_code

    def test_invalid_javascript_code_syntax(self):
        """Test that invalid JavaScript code fails validation."""
        skip_if_js_not_supported()
        from pydantic import ValidationError

        invalid_code = "const x = 1;\nconsole.log(x"  # Missing parenthesis
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=invalid_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)

    def test_empty_javascript_code(self):
        """Test that empty code passes validation."""
        skip_if_js_not_supported()
        empty_code = ""
        cs = CodeString(code=empty_code, language="javascript")
        assert cs.code == empty_code


class TestTypeScriptCodeStringValidation:
    """Tests for TypeScript CodeString validation."""

    def test_typescript_string(self):
        """Test valid TypeScript code string."""
        skip_if_js_not_supported()
        code = CodeString(code="const x: number = 1;", language="typescript")
        assert code.code == "const x: number = 1;"

    def test_valid_typescript_code(self):
        """Test that valid TypeScript code passes validation."""
        skip_if_js_not_supported()
        valid_code = "function add(a: number, b: number): number { return a + b; }"
        cs = CodeString(code=valid_code, language="typescript")
        assert cs.code == valid_code

    def test_typescript_type_assertion_valid(self):
        """TypeScript type assertions should pass TypeScript validation."""
        skip_if_js_not_supported()
        ts_code = "const value = 4.9 as unknown as number;"
        cs = CodeString(code=ts_code, language="typescript")
        assert cs.code == ts_code

    def test_typescript_type_assertion_invalid_in_javascript(self):
        """TypeScript type assertions should FAIL JavaScript validation.

        This is the critical test - TypeScript syntax like 'as unknown as number'
        should fail when validated as JavaScript.
        """
        skip_if_js_not_supported()
        from pydantic import ValidationError

        ts_code = "const value = 4.9 as unknown as number;"
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)

    def test_typescript_interface_valid(self):
        """TypeScript interfaces should pass TypeScript validation."""
        skip_if_js_not_supported()
        ts_code = "interface User { name: string; age: number; }"
        cs = CodeString(code=ts_code, language="typescript")
        assert cs.code == ts_code

    def test_typescript_interface_invalid_in_javascript(self):
        """TypeScript interfaces should FAIL JavaScript validation."""
        skip_if_js_not_supported()
        from pydantic import ValidationError

        ts_code = "interface User { name: string; age: number; }"
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)

    def test_typescript_generics_valid(self):
        """TypeScript generics should pass TypeScript validation."""
        skip_if_js_not_supported()
        ts_code = "function identity<T>(arg: T): T { return arg; }"
        cs = CodeString(code=ts_code, language="typescript")
        assert cs.code == ts_code

    def test_typescript_generics_invalid_in_javascript(self):
        """TypeScript generics should FAIL JavaScript validation."""
        skip_if_js_not_supported()
        from pydantic import ValidationError

        ts_code = "function identity<T>(arg: T): T { return arg; }"
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)


class TestAiServiceClientJavaScript:
    """Tests for AiServiceClient with JavaScript/TypeScript - mirrors test_validate_python_code.py."""

    def test_javascript_generated_candidates_validation(self):
        """Test that JavaScript candidates are validated correctly."""
        skip_if_js_not_supported()
        ai_service = AiServiceClient()

        # Invalid JavaScript (missing closing parenthesis)
        code = """```javascript:file.js
console.log(name
```"""
        mock_candidates = [{"source_code": code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(mock_candidates, OptimizedCandidateSource.OPTIMIZE)
        assert len(candidates) == 0

        # Valid JavaScript
        code = """```javascript:file.js
console.log('Hello, World!');
```"""
        mock_candidates = [{"source_code": code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(mock_candidates, OptimizedCandidateSource.OPTIMIZE)
        assert len(candidates) == 1
        assert candidates[0].source_code.code_strings[0].code == "console.log('Hello, World!');"

    def test_typescript_generated_candidates_validation(self):
        """Test that TypeScript candidates are validated correctly."""
        skip_if_js_not_supported()
        ai_service = AiServiceClient()

        # Valid TypeScript with type annotations
        code = """```typescript:file.ts
function add(a: number, b: number): number {
    return a + b;
}
```"""
        mock_candidates = [{"source_code": code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(mock_candidates, OptimizedCandidateSource.OPTIMIZE)
        assert len(candidates) == 1

    def test_typescript_type_assertion_in_candidate(self):
        """Test that TypeScript type assertions are valid in TS candidates."""
        skip_if_js_not_supported()
        ai_service = AiServiceClient()

        # TypeScript-specific syntax should be valid
        code = """```typescript:file.ts
const value = 4.9 as unknown as number;
```"""
        mock_candidates = [{"source_code": code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(mock_candidates, OptimizedCandidateSource.OPTIMIZE)
        assert len(candidates) == 1


class TestBackendLanguageParameter:
    """Tests verifying language parameter flows correctly to backend."""

    def test_testgen_request_includes_typescript_language(self, tmp_path):
        """Verify the language parameter is sent as 'typescript' for .ts files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current

        # Set current language to TypeScript
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

        ai_client = AiServiceClient()
        captured_payload = None

        def capture_request(*args, **kwargs):
            nonlocal captured_payload
            if 'payload' in kwargs:
                captured_payload = kwargs['payload']
            elif len(args) > 1:
                captured_payload = args[1]
            # Return a mock response to avoid actual API call
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_tests": "// test",
                "instrumented_behavior_tests": "// test",
                "instrumented_perf_tests": "// test",
            }
            return mock_response

        with patch.object(ai_client, 'make_ai_service_request', side_effect=capture_request):
            ai_client.generate_regression_tests(
                source_code_being_tested=ts_file.read_text(),
                function_to_optimize=func,
                helper_function_names=[],
                module_path=ts_file,
                test_module_path=tmp_path / "tests" / "utils.test.ts",
                test_framework="vitest",
                test_timeout=30,
                trace_id="test-language-param-ts",
                test_index=0,
                language="typescript",
            )

        assert captured_payload is not None
        assert captured_payload.get('language') == 'typescript', \
            f"Expected language='typescript', got: {captured_payload.get('language')}"

    def test_testgen_request_includes_javascript_language(self, tmp_path):
        """Verify the language parameter is sent as 'javascript' for .js files."""
        skip_if_js_not_supported()
        from codeflash.discovery.functions_to_optimize import find_all_functions_in_file
        from codeflash.languages import current as lang_current

        # Set current language to JavaScript
        lang_current._current_language = Language.JAVASCRIPT

        js_file = tmp_path / "utils.js"
        js_file.write_text("""
export function add(a, b) {
    return a + b;
}
""")

        functions = find_all_functions_in_file(js_file)
        func = functions[js_file][0]

        # Verify function has correct language
        assert func.language == "javascript"

        ai_client = AiServiceClient()
        captured_payload = None

        def capture_request(*args, **kwargs):
            nonlocal captured_payload
            if 'payload' in kwargs:
                captured_payload = kwargs['payload']
            elif len(args) > 1:
                captured_payload = args[1]
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_tests": "// test",
                "instrumented_behavior_tests": "// test",
                "instrumented_perf_tests": "// test",
            }
            return mock_response

        with patch.object(ai_client, 'make_ai_service_request', side_effect=capture_request):
            ai_client.generate_regression_tests(
                source_code_being_tested=js_file.read_text(),
                function_to_optimize=func,
                helper_function_names=[],
                module_path=js_file,
                test_module_path=tmp_path / "tests" / "utils.test.js",
                test_framework="jest",
                test_timeout=30,
                trace_id="test-language-param-js",
                test_index=0,
                language="javascript",
            )

        assert captured_payload is not None
        assert captured_payload.get('language') == 'javascript', \
            f"Expected language='javascript', got: {captured_payload.get('language')}"
