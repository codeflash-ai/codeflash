import numpy as np


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
    # Efficient Hamming distance calculation via count_nonzero
    diff_count = np.count_nonzero(a != b)
    return np.float64(diff_count) / a.size
