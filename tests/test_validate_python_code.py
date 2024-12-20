from codeflash.models.models import CodeString
import pytest
from pydantic import ValidationError

def test_python_string():
    code = CodeString(code="print('Hello, World!')")
    assert code.code == "print('Hello, World!')"


def test_valid_python_code():
    # This should pass without errors
    valid_code = "x = 1\ny = x + 2\nprint(y)"
    cs = CodeString(code=valid_code)
    assert cs.code == valid_code

def test_invalid_python_code_syntax():
    # Missing a parenthesis should cause a syntax error
    invalid_code = "x = 1\nprint(x"
    with pytest.raises(ValidationError) as exc_info:
        CodeString(code=invalid_code)
    # Check that the error message mentions "Invalid Python code"
    assert "Invalid Python code:" in str(exc_info.value)

def test_invalid_python_code_name_error():
    # Note that compile won't catch NameError because it's a runtime error, not a syntax error.
    # This code is syntactically valid but would fail at runtime. However, compile won't fail.
    invalid_runtime_code = "print(undefined_variable)"
    cs = CodeString(code=invalid_runtime_code)
    assert cs.code == invalid_runtime_code

def test_empty_code_string():
    # Empty code is still syntactically valid (no-op)
    empty_code = ""
    cs = CodeString(code=empty_code)
    assert cs.code == empty_code

def test_whitespace_only():
    # Whitespace is still syntactically valid (no-op)
    whitespace_code = "    "
    cs = CodeString(code=whitespace_code)
    assert cs.code == whitespace_code