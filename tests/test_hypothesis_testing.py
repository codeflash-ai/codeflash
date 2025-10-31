"""Tests for hypothesis_testing.py functions."""

from codeflash.verification.hypothesis_testing import make_hypothesis_tests_deterministic


def test_adds_derandomize_decorator():
    """Test that @settings(derandomize=True) is added when missing."""
    src = """
from hypothesis import given, strategies as st

@given(x=st.integers())
def test_x(x):
    assert isinstance(x, int)
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(x=st.integers(min_value=-10000, max_value=10000))\n@settings(derandomize=True)\ndef test_x(x):\n    assert isinstance(x, int)"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_integers_constrained_with_negatives():
    """Test that st.integers() gets bounded to [-10000, 10000]."""
    src = """from hypothesis import given, strategies as st
@given(x=st.integers())
def t(x):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(x=st.integers(min_value=-10000, max_value=10000))\n@settings(derandomize=True)\ndef t(x):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_floats_constrained_to_finite():
    """Test that st.floats() is constrained to finite values with bounds."""
    src = """from hypothesis import given, strategies as st
@given(x=st.floats())
def t(x):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(x=st.floats(min_value=-1000000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False))\n@settings(derandomize=True)\ndef t(x):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_existing_constraints_not_overridden():
    """Test that existing constraints on strategies are preserved."""
    src = """from hypothesis import given, strategies as st, settings

@settings(derandomize=True, max_examples=5)
@given(x=st.integers(min_value=-5, max_value=5))
def t(x):
    pass
"""
    expected = """from hypothesis import given, strategies as st, settings\n\n@settings(derandomize=True, max_examples=5)\n@given(x=st.integers(min_value=-5, max_value=5))\ndef t(x):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_existing_float_constraints_preserved():
    """Test that existing float constraints are not overridden."""
    src = """from hypothesis import given, strategies as st

@given(y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def t(y):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))\n@settings(derandomize=True)\ndef t(y):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_idempotency():
    """Test that running the function twice produces the same result."""
    src = """from hypothesis import given, strategies as st

@given(x=st.integers(), y=st.floats())
def test_func(x, y):
    pass
"""
    out1 = make_hypothesis_tests_deterministic(src)
    out2 = make_hypothesis_tests_deterministic(out1)
    assert out1 == out2


def test_multiple_strategies_handled():
    """Test that multiple strategies in one test are all constrained."""
    src = """from hypothesis import given, strategies as st

@given(a=st.integers(), b=st.integers(), c=st.floats())
def test_multi(a, b, c):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(a=st.integers(min_value=-10000, max_value=10000), b=st.integers(min_value=-10000, max_value=10000), c=st.floats(min_value=-1000000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False))\n@settings(derandomize=True)\ndef test_multi(a, b, c):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_settings_import_added_if_missing():
    """Test that 'from hypothesis import settings' is added when needed."""
    src = """from hypothesis import given, strategies as st

@given(x=st.integers())
def test_x(x):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(x=st.integers(min_value=-10000, max_value=10000))\n@settings(derandomize=True)\ndef test_x(x):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_partial_constraints_completed():
    """Test that partial constraints are completed."""
    src = """from hypothesis import given, strategies as st

@given(x=st.integers(min_value=100))
def test_x(x):
    pass
"""
    expected = """from hypothesis import settings\nfrom hypothesis import given, strategies as st\n\n@given(x=st.integers(min_value=100))\n@settings(derandomize=True)\ndef test_x(x):\n    pass"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected


def test_syntax_error_returns_original():
    """Test that invalid Python syntax returns original code unchanged."""
    invalid_src = "this is not valid python @#$%"
    out = make_hypothesis_tests_deterministic(invalid_src)
    assert out == invalid_src


def test_no_hypothesis_code_unchanged():
    """Test that code without hypothesis is returned mostly unchanged."""
    src = """def regular_function(x):
    return x * 2

def test_regular():
    assert regular_function(2) == 4
"""
    expected = """from hypothesis import settings\n\n@settings(derandomize=True)\ndef regular_function(x):\n    return x * 2\n\n@settings(derandomize=True)\ndef test_regular():\n    assert regular_function(2) == 4"""
    out = make_hypothesis_tests_deterministic(src)
    assert out == expected
