from code_to_optimize.find_factors import find_factors


def test_small_number():
    assert find_factors(12) == [
        (1, 12),
        (2, 6),
        (3, 4),
        (4, 3),
        (6, 2),
        (12, 1),
    ], "Failed on small number with multiple factors"


def test_prime_number():
    assert find_factors(13) == [(1, 13), (13, 1)], "Failed on prime number"


def test_perfect_square():
    assert find_factors(16) == [
        (1, 16),
        (2, 8),
        (4, 4),
        (8, 2),
        (16, 1),
    ], "Failed on perfect square number"


def test_large_number():
    # 120 has factors: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120
    result = find_factors(120)
    expected_factors = 16  # There should be 16 pairs
    assert (
        len(result) == expected_factors
    ), "Failed on large number with multiple factors"


def test_one():
    assert find_factors(1) == [
        (1, 1)
    ], "Failed on one, which should only have one factor pair"


def test_zero():
    assert find_factors(0) == [], "Failed on zero, which should have no factors"


def test_negative_number():
    # Expecting an error, or modify the function to handle negative input gracefully
    assert find_factors(-1) == [], "Failed on negative number"
