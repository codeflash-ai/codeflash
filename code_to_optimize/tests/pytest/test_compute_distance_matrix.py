from code_to_optimize.compute_distance_matrix import compute_distance_matrix
import numpy as np


def test_full_distance_matrix():
    Xs = np.array([[0, 0], [3, 4]])
    expected_output = np.array([[0.0, 5.0], [5.0, 0.0]])
    result = compute_distance_matrix(Xs)
    np.testing.assert_array_almost_equal(result, expected_output)


def test_update_specific_entries():
    Xs = np.array([[0, 0], [3, 4], [6, 8]])
    D_current = np.zeros((3, 3))
    D_current[:2, :2] = np.array([[0.0, 5.0], [5.0, 0.0]])
    i = 2
    expected_output = np.array([[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]])
    result = compute_distance_matrix(Xs, D_current, i)
    np.testing.assert_array_almost_equal(result, expected_output)


def test_single_vector():
    Xs = np.array([[0, 0]])
    expected_output = np.array([[0.0]])
    result = compute_distance_matrix(Xs)
    np.testing.assert_array_almost_equal(result, expected_output)
