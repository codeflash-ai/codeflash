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
    out = make_hypothesis_tests_deterministic(src)
    assert "@settings(derandomize=True)" in out or "settings(derandomize=True)" in out


def test_integers_constrained_with_negatives():
    """Test that st.integers() gets bounded to [-10000, 10000]."""
    src = """from hypothesis import given, strategies as st
@given(x=st.integers())
def t(x):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    # Remove spaces for easier checking
    normalized = out.replace(" ", "").replace("\n", "")
    assert "min_value=-10000" in normalized
    assert "max_value=10000" in normalized


def test_floats_constrained_to_finite():
    """Test that st.floats() is constrained to finite values with bounds."""
    src = """from hypothesis import given, strategies as st
@given(x=st.floats())
def t(x):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    normalized = out.replace(" ", "").replace("\n", "")
    assert "allow_nan=False" in normalized
    assert "allow_infinity=False" in normalized
    assert "min_value=" in normalized and "max_value=" in normalized


def test_existing_constraints_not_overridden():
    """Test that existing constraints on strategies are preserved."""
    src = """from hypothesis import given, strategies as st, settings

@settings(derandomize=True, max_examples=5)
@given(x=st.integers(min_value=-5, max_value=5))
def t(x):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    # Should not add duplicate settings decorator
    assert out.count("@settings") == 1
    # Should preserve original constraints
    assert "min_value=-5" in out or "min_value= -5" in out
    assert "max_value=5" in out or "max_value= 5" in out
    # Should not add the default -10000/10000 bounds
    assert "-10000" not in out


def test_existing_float_constraints_preserved():
    """Test that existing float constraints are not overridden."""
    src = """from hypothesis import given, strategies as st

@given(y=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def t(y):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    assert "min_value=-1.0" in out or "min_value= -1.0" in out
    assert "max_value=1.0" in out or "max_value= 1.0" in out
    # Should not add the default 1e6 bounds
    assert "1e6" not in out and "1000000" not in out


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
    out = make_hypothesis_tests_deterministic(src)
    normalized = out.replace(" ", "").replace("\n", "")
    # All integers should be constrained
    assert normalized.count("min_value=-10000") >= 2
    assert normalized.count("max_value=10000") >= 2
    # Float should be constrained
    assert "allow_nan=False" in normalized
    assert "allow_infinity=False" in normalized


def test_settings_import_added_if_missing():
    """Test that 'from hypothesis import settings' is added when needed."""
    src = """from hypothesis import given, strategies as st

@given(x=st.integers())
def test_x(x):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    # Should have settings import or settings in existing import
    assert "settings" in out


def test_partial_constraints_completed():
    """Test that partial constraints are completed."""
    src = """from hypothesis import given, strategies as st

@given(x=st.integers(min_value=100))
def test_x(x):
    pass
"""
    out = make_hypothesis_tests_deterministic(src)
    # Should keep the min_value=100 and not override
    assert "min_value=100" in out or "min_value= 100" in out
    # Should not add default bounds since min_value exists
    assert "-10000" not in out


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
    out = make_hypothesis_tests_deterministic(src)
    # Should still parse and return valid code
    assert "def regular_function" in out
    assert "def test_regular" in out
