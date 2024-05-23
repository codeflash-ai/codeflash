import numpy as np


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> np.floating:
    return np.mean(a != b)
