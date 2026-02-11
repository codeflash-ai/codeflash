"""Tests for JavaScript/TypeScript code validation in CodeString.

These tests ensure that JavaScript and TypeScript code is validated correctly
using the appropriate syntax parser for each language.
"""

import pytest
from pydantic import ValidationError

from codeflash.api.aiservice import AiServiceClient
from codeflash.models.models import CodeString, OptimizedCandidateSource


class TestJavaScriptCodeValidation:
    """Tests for JavaScript code validation."""

    def test_valid_javascript_code(self):
        """Valid JavaScript code should pass validation."""
        valid_code = "const x = 1;\nconst y = x + 2;\nconsole.log(y);"
        cs = CodeString(code=valid_code, language="javascript")
        assert cs.code == valid_code

    def test_invalid_javascript_code_syntax(self):
        """Invalid JavaScript syntax should raise ValidationError."""
        invalid_code = "const x = 1;\nconsole.log(x"  # Missing closing parenthesis
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=invalid_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)

    def test_javascript_empty_code(self):
        """Empty code is syntactically valid."""
        empty_code = ""
        cs = CodeString(code=empty_code, language="javascript")
        assert cs.code == empty_code

    def test_javascript_arrow_function(self):
        """Arrow functions should be valid JavaScript."""
        code = "const add = (a, b) => a + b;"
        cs = CodeString(code=code, language="javascript")
        assert cs.code == code


class TestTypeScriptCodeValidation:
    """Tests for TypeScript code validation."""

    def test_valid_typescript_code(self):
        """Valid TypeScript code should pass validation."""
        valid_code = "const x: number = 1;\nconst y: number = x + 2;\nconsole.log(y);"
        cs = CodeString(code=valid_code, language="typescript")
        assert cs.code == valid_code

    def test_typescript_type_assertion(self):
        """TypeScript type assertions should be valid."""
        code = "const value = 4.9 as unknown as number;"
        cs = CodeString(code=code, language="typescript")
        assert cs.code == code

    def test_typescript_interface(self):
        """TypeScript interfaces should be valid."""
        code = "interface User { name: string; age: number; }"
        cs = CodeString(code=code, language="typescript")
        assert cs.code == code

    def test_typescript_generic_function(self):
        """TypeScript generics should be valid."""
        code = "function identity<T>(arg: T): T { return arg; }"
        cs = CodeString(code=code, language="typescript")
        assert cs.code == code

    def test_invalid_typescript_code_syntax(self):
        """Invalid TypeScript syntax should raise ValidationError."""
        invalid_code = "const x: number = 1;\nconsole.log(x"  # Missing closing parenthesis
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=invalid_code, language="typescript")
        assert "Invalid Typescript code" in str(exc_info.value)

    def test_typescript_syntax_invalid_as_javascript(self):
        """TypeScript-specific syntax should fail when validated as JavaScript."""
        ts_code = "const value = 4.9 as unknown as number;"
        # Should pass as TypeScript
        cs_ts = CodeString(code=ts_code, language="typescript")
        assert cs_ts.code == ts_code

        # Should fail as JavaScript (type assertions are not valid JS)
        with pytest.raises(ValidationError) as exc_info:
            CodeString(code=ts_code, language="javascript")
        assert "Invalid Javascript code" in str(exc_info.value)


class TestGeneratedCandidatesValidation:
    """Tests for validation of generated optimization candidates."""

    def test_javascript_generated_candidates_validation(self):
        """JavaScript optimization candidates should be validated."""
        ai_service = AiServiceClient()

        # Invalid JavaScript code
        invalid_code = """```javascript:file.js
const x = 1
console.log(x
```"""
        mock_candidates = [{"source_code": invalid_code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(
            mock_candidates, OptimizedCandidateSource.OPTIMIZE, language="javascript"
        )
        assert len(candidates) == 0

        # Valid JavaScript code
        valid_code = """```javascript:file.js
const x = 1;
console.log(x);
```"""
        mock_candidates = [{"source_code": valid_code, "explanation": "", "optimization_id": ""}]
        candidates = ai_service._get_valid_candidates(
            mock_candidates, OptimizedCandidateSource.OPTIMIZE, language="javascript"
        )
        assert len(candidates) == 1

    def test_typescript_generated_candidates_validation(self):
        """TypeScript optimization candidates should be validated."""
        ai_service = AiServiceClient()

        # TypeScript code with type assertions (valid TS, invalid JS)
        ts_code = """```typescript:file.ts
const value = 4.9 as unknown as number;
console.log(value);
```"""
        mock_candidates = [{"source_code": ts_code, "explanation": "", "optimization_id": ""}]

        # Should pass when validated as TypeScript
        candidates = ai_service._get_valid_candidates(
            mock_candidates, OptimizedCandidateSource.OPTIMIZE, language="typescript"
        )
        assert len(candidates) == 1
