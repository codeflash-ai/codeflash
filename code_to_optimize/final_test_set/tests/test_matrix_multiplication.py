from code_to_optimize.final_test_set.matrix_multiplication import matrix_multiply
import pytest


def test_matrix_multiplication_basic():
    A = [[1, 2], [3, 4]]
    B = [[2, 0], [1, 2]]
    expected = [[4, 4], [10, 8]]
    assert matrix_multiply(A, B) == expected


def test_matrix_multiplication_dimension_mismatch():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        matrix_multiply(A, B)


def test_zero_matrix_multiplication():
    A = [[1, 2], [3, 4]]
    B = [[0, 0], [0, 0]]
    expected = [[0, 0], [0, 0]]
    assert matrix_multiply(A, B) == expected


def test_identity_matrix_multiplication():
    A = [[1, 2], [3, 4]]
    I = [[1, 0], [0, 1]]
    assert matrix_multiply(A, I) == A


def test_large_matrix_multiplication():
    A = [[1] * 100 for _ in range(100)]
    B = [[2] * 100 for _ in range(100)]
    expected = [[200] * 100 for _ in range(100)]
    assert matrix_multiply(A, B) == expected
