from code_to_optimize.final_test_set.compute_mds_error import compute_mds_error
import numpy as np


def test_total_error():
    D_goal = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])
    D_current = np.array([[0, 1, 3], [1, 0, 2], [3, 2, 0]])
    expected = 4
    result = compute_mds_error(D_goal, D_current)
    assert result == expected


def test_row_wise_errors():
    D_goal = np.array([[0, 2], [2, 0]])
    D_current = np.array([[0, 3], [3, 0]])
    expected = np.array([1, 1])
    result = compute_mds_error(D_goal, D_current, i=-1)
    np.testing.assert_array_equal(result, expected)


def test_specific_row_error():
    D_goal = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])
    D_current = np.array([[0, 1, 4], [1, 0, 1], [4, 1, 0]])
    expected = 1
    result = compute_mds_error(D_goal, D_current, i=2)
    assert result == expected
