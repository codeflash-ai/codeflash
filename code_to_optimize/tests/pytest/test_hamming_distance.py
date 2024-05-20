from code_to_optimize.hamming_distance import _hamming_distance
import numpy as np


def test_no_differences():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3, 4])
    assert _hamming_distance(a, b) == 0.0


def test_partial_differences():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 0, 4])
    assert _hamming_distance(a, b) == 0.25
