import pytest

from code_to_optimize.final_test_set.integration import integrate_f


def isclose(a, b, rel_tol=1e-5, abs_tol=0.0):
    """Helper function to compare two floating points for 'closeness'.
    Uses a combination of relative and absolute tolerances.
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def test_simple_range():
    a, b, N = 0, 1, 1000
    result = integrate_f(a, b, N)
    expected = -1 / 6  # Analytical result
    assert isclose(result, expected), f"Expected {expected}, got {result}"


def test_negative_to_positive_range():
    a, b, N = -1, 1, 500
    result = integrate_f(a, b, N)
    expected = 0.6706719  # Analytical result
    assert isclose(result, expected), f"Expected {expected}, got {result}"


# Optionally, you can add more detailed information to your pytest output
def test_with_pytest_approx():
    a, b, N = 0, 1, 1000
    result = integrate_f(a, b, N)
    expected = -1 / 6
    assert result == pytest.approx(expected, rel=1e-5), "Test failed with pytest's approx."
