from code_to_optimize.bubble_sort import sorter as bubble_sorter


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = bubble_sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = bubble_sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(5000)))
    output = bubble_sorter(input)
    assert output == list(range(5000))
