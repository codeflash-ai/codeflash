from codeflash.result.list_example import compare_lists


def test_list_example_1():
    a, b = [1, 2, 3, 4, 5, 6, 9], [3, 5, 7, 8, 9]

    results = compare_lists(a, b)
    assert results == (set([9, 3, 5]), set([1, 2, 4, 6]), set([8, 7]))

    a = [i for i in range(2500) if i % 3 == 0]
    b = [i for i in range(2500) if i % 4 == 0]

    results = compare_lists(a, b)
    print("hi")
