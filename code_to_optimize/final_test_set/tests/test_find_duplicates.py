from code_to_optimize.final_test_set.find_duplicates import find_duplicates


def test_basic_case():
    assert find_duplicates([1, 2, 3, 2, 1, 5, 6, 5]) == [
        1,
        2,
        5,
    ], "Failed on basic case"


def test_no_duplicates():
    assert find_duplicates([1, 2, 3, 4, 5]) == [], "Failed when no duplicates present"


def test_multiple_duplicates():
    assert find_duplicates([1, 2, 2, 3, 3, 3, 4]) == [
        2,
        3,
    ], "Failed on multiple duplicates of the same item"


def test_empty_list():
    assert find_duplicates([]) == [], "Failed on empty list"


def test_all_elements_same():
    assert find_duplicates([7, 7, 7, 7]) == [7], "Failed when all elements are the same"


def test_mixed_data_types():
    assert find_duplicates(["apple", "banana", "apple", 42, 42]) == [
        "apple",
        42,
    ], "Failed on mixed data types"
