from code_to_optimize.bubble_sort_deps import sorter_deps


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = sorter_deps(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter_deps(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(5000)))
    output = sorter_deps(input)
    assert output == list(range(5000))