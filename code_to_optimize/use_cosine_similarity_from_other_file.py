from typing import List, Optional, Tuple

from code_to_optimize.math_utils import Matrix, cosine_similarity_top_k


def use_cosine_similarity(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    return cosine_similarity_top_k(X, Y, top_k, score_threshold)


CACHED_TESTS = """import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple, Union
Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]
def cosine_similarity_top_k(X: Matrix, Y: Matrix, top_k: Optional[int]=5, score_threshold: Optional[float]=None) -> Tuple[List[Tuple[int, int]], List[float]]:
    \"\"\"Row-wise cosine similarity with optional top-k and score threshold filtering.
    Args:
        X: Matrix.
        Y: Matrix, same width as X.
        top_k: Max number of results to return.
        score_threshold: Minimum cosine similarity of results.
    Returns:
        Tuple of two lists. First contains two-tuples of indices (X_idx, Y_idx),
            second contains corresponding cosine similarities.
    \"\"\"
    if len(X) == 0 or len(Y) == 0:
        return ([], [])
    score_array = cosine_similarity(X, Y)
    sorted_idxs = score_array.flatten().argsort()[::-1]
    top_k = top_k or len(sorted_idxs)
    top_idxs = sorted_idxs[:top_k]
    score_threshold = score_threshold or -1.0
    top_idxs = top_idxs[score_array.flatten()[top_idxs] > score_threshold]
    ret_idxs = [(x // score_array.shape[1], x % score_array.shape[1]) for x in top_idxs]
    scores = score_array.flatten()[top_idxs].tolist()
    return (ret_idxs, scores)
def use_cosine_similarity(X: Matrix, Y: Matrix, top_k: Optional[int]=5, score_threshold: Optional[float]=None) -> Tuple[List[Tuple[int, int]], List[float]]:
    return cosine_similarity_top_k(X, Y, top_k, score_threshold)
class TestUseCosineSimilarity(unittest.TestCase):
    def test_normal_scenario(self):
        X = [[1, 2, 3], [4, 5, 6]]
        Y = [[7, 8, 9], [10, 11, 12]]
        result = use_cosine_similarity(X, Y, top_k=1, score_threshold=0.5)
        self.assertEqual(result, ([(0, 1)], [0.9746318461970762]))
    def test_edge_case_empty_matrices(self):
        X = []
        Y = []
        result = use_cosine_similarity(X, Y)
        self.assertEqual(result, ([], []))
    def test_edge_case_different_widths(self):
        X = [[1, 2, 3]]
        Y = [[4, 5]]
        with self.assertRaises(ValueError):
            use_cosine_similarity(X, Y)
    def test_edge_case_negative_top_k(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        with self.assertRaises(IndexError):
            use_cosine_similarity(X, Y, top_k=-1)
    def test_edge_case_zero_top_k(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        result = use_cosine_similarity(X, Y, top_k=0)
        self.assertEqual(result, ([], []))
    def test_edge_case_negative_score_threshold(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        result = use_cosine_similarity(X, Y, score_threshold=-1.0)
        self.assertEqual(result, ([(0, 0)], [0.9746318461970762]))
    def test_edge_case_large_score_threshold(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        result = use_cosine_similarity(X, Y, score_threshold=2.0)
        self.assertEqual(result, ([], []))
    def test_exceptional_case_non_matrix_X(self):
        X = [1, 2, 3]
        Y = [[4, 5, 6]]
        with self.assertRaises(ValueError):
            use_cosine_similarity(X, Y)
    def test_exceptional_case_non_integer_top_k(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        with self.assertRaises(TypeError):
            use_cosine_similarity(X, Y, top_k='5')
    def test_exceptional_case_non_float_score_threshold(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        with self.assertRaises(TypeError):
            use_cosine_similarity(X, Y, score_threshold='0.5')
    def test_special_values_nan_in_matrices(self):
        X = [[1, 2, np.nan]]
        Y = [[4, 5, 6]]
        with self.assertRaises(ValueError):
            use_cosine_similarity(X, Y)
    def test_special_values_none_top_k(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        result = use_cosine_similarity(X, Y, top_k=None)
        self.assertEqual(result, ([(0, 0)], [0.9746318461970762]))
    def test_special_values_none_score_threshold(self):
        X = [[1, 2, 3]]
        Y = [[4, 5, 6]]
        result = use_cosine_similarity(X, Y, score_threshold=None)
        self.assertEqual(result, ([(0, 0)], [0.9746318461970762]))
    def test_large_inputs(self):
        X = np.random.rand(1000, 1000)
        Y = np.random.rand(1000, 1000)
        result = use_cosine_similarity(X, Y, top_k=10, score_threshold=0.5)
        self.assertEqual(len(result[0]), 10)
        self.assertEqual(len(result[1]), 10)
        self.assertTrue(all((score > 0.5 for score in result[1])))
if __name__ == '__main__':
    unittest.main()"""
