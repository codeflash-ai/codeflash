from typing import List, Optional, Tuple

from code_to_optimize.math_utils import Matrix, cosine_similarity_top_k


def use_cosine_similarity(
    X: Matrix,
    Y: Matrix,
    top_k: Optional[int] = 5,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    return cosine_similarity_top_k(X, Y, top_k, score_threshold)