from code_to_optimize.find_top_k_elements import find_top_k_elements


def test_negative_k():
    assert find_top_k_elements([3, 1, 2, 4], -1) == [], "Failed when k is negative"


def test_zero_k():
    assert find_top_k_elements([10, 8, 12, 5], 0) == [], "Failed when k is zero"


def test_k_greater_than_array_length():
    array = [4, 1, 5, 6, 2]
    k = 10
    expected = sorted(array, reverse=True)
    assert (
        find_top_k_elements(array, k) == expected
    ), "Failed when k is greater than array length"


def test_normal_case():
    array = [20, 1, 15, 3, 30, 10]
    k = 3
    expected = [30, 20, 15]
    assert find_top_k_elements(array, k) == expected, "Failed in normal scenario"


def test_array_with_duplicate_values():
    array = [5, 5, 5, 5]
    k = 2
    expected = [5, 5]
    assert (
        find_top_k_elements(array, k) == expected
    ), "Failed when array contains duplicates"


def test_empty_array():
    assert find_top_k_elements([], 3) == [], "Failed when array is empty"


def test_single_element_array():
    array = [42]
    k = 1
    expected = [42]
    assert (
        find_top_k_elements(array, k) == expected
    ), "Failed when array contains a single element"
