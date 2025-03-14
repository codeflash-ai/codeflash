import ast

from codeflash.verification.hypothesis_testing import (
    make_hypothesis_tests_deterministic,
    remove_functions_with_only_any_type,
)


def normalize_code(code: str) -> str:
    return ast.unparse(ast.parse(code))


def test_remove_functions_with_only_any_type_no_functions() -> None:
    code = """

def func1(x: Any) -> Any:
    pass

def func2(y: Any) -> Any:
    pass
"""
    expected = """
"""
    result = remove_functions_with_only_any_type(code)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))

    code = """

def func1(x: Any) -> Any:
    pass

def func2(y: int) -> int:
    pass

def func3(z):
    pass
"""
    expected = """

def func2(y: int) -> int:
    pass

def func3(z):
    pass
"""
    result = remove_functions_with_only_any_type(code)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))

    code = """

def func1(x):
    pass

def func2(y):
    pass
"""
    expected = """

def func1(x):
    pass

def func2(y):
    pass
"""
    result = remove_functions_with_only_any_type(code)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))

    code = """

def func1(x: Any):
    pass

def func2(y: int):
    pass
"""
    expected = """

def func2(y: int):
    pass
"""
    result = remove_functions_with_only_any_type(code)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))

    code = """
import os
x = 5
def func1(a: Any):
  pass
def func2(b: int):
  pass
from typing import List
def func3(c:List[str]):
  pass
"""
    expected = """
import os
x = 5
def func2(b: int):
    pass
from typing import List
def func3(c:List[str]):
    pass
"""
    result = remove_functions_with_only_any_type(code)
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(expected))


def test_make_hypothesis_tests_deterministic() -> None:
    input_code = """
@given(st.integers())
def test_something(x):
    assert x == x
"""
    expected_code = """from hypothesis import settings

@given(st.integers())
@settings(derandomize=True)
def test_something(x):
    assert x == x
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert normalize_code(processed_code) == normalize_code(expected_code)

    input_code = """
from hypothesis import given, strategies as st

@given(st.integers())
def test_something(x):
    assert x == x
"""
    expected_code = """from hypothesis import settings
from hypothesis import given, strategies as st

@given(st.integers())
@settings(derandomize=True)
def test_something(x):
    assert x == x
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert normalize_code(processed_code) == normalize_code(expected_code)

    input_code = """
from hypothesis import given, strategies as st, settings

@settings(max_examples=10)
@given(st.integers())
def test_something(x):
    assert x == x
"""
    expected_code = """from hypothesis import given, strategies as st, settings

@settings(max_examples=10, derandomize=True)
@given(st.integers())
def test_something(x):
    assert x == x
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert normalize_code(processed_code) == normalize_code(expected_code)

    input_code = """
from hypothesis import given, strategies as st, settings

@settings(derandomize=True)
@given(st.integers())
def test_something(x):
    assert x == x
"""
    expected_code = """from hypothesis import given, strategies as st, settings

@settings(derandomize=True)
@given(st.integers())
def test_something(x):
    assert x == x
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert normalize_code(processed_code) == normalize_code(expected_code)

    input_code = """
@given(st.integers())
def test_something(x):
    assert x == x

@settings(max_examples=1)
@given(st.booleans())
def test_something_else(y):
    assert y == y
"""
    expected_code = """from hypothesis import settings

@given(st.integers())
@settings(derandomize=True)
def test_something(x):
    assert x == x

@settings(max_examples=1, derandomize=True)
@given(st.booleans())
def test_something_else(y):
    assert y == y
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert normalize_code(processed_code) == normalize_code(expected_code)

    # now let's trigger the SyntaxError exception, which should return the original code
    input_code = """
@given(st.integers())
def test_something(x)
    assert x == x
"""
    processed_code = make_hypothesis_tests_deterministic(input_code)
    assert processed_code == input_code
