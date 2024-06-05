from code_to_optimize.final_test_set.unique_paths import uniquePaths


def test_minimal_grid():
    assert uniquePaths(1, 1) == 1, "Failed on the minimal grid 1x1"


def test_single_row():
    assert uniquePaths(1, 5) == 1, "Failed on a single row grid 1x5"


def test_single_column():
    assert uniquePaths(5, 1) == 1, "Failed on a single column grid 5x1"


def test_square_grid():
    assert uniquePaths(3, 3) == 6, "Failed on square grid 3x3"


def test_rectangular_grid():
    assert uniquePaths(2, 3) == 3, "Failed on rectangular grid 2x3"


def test_large_grid():
    assert uniquePaths(10, 10) == 48620, "Failed on large grid 10x10"
