from code_to_optimize.bubble_sort import sorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    if len(input) > 0:
        assert sorter(input) == [0, 1, 2, 3, 4, 5]
