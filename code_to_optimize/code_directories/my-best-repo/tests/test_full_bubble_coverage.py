import pytest
from bubble_sort import sorter, sorter_one_level_depth, add_one_level_depth, add, multiply_and_add


def test_sort():
    input = [5, 4, 3, 2, 1, 0]
    output = sorter(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(5000)))
    output = sorter(input)
    assert output == list(range(5000))

def test_sorter_one_level_depth():
    input = [3, 2, 1]
    output = sorter_one_level_depth(input)
    assert output == [1, 2, 3]


def test_add_one_level_depth():
    assert add_one_level_depth(1, 2) == 3
    assert add_one_level_depth(-1, 1) == 0
    assert add_one_level_depth(0, 0) == 0


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_multiply_and_add():
    assert multiply_and_add(2, 3, 4) == 10
    assert multiply_and_add(0, 3, 4) == 4
    assert multiply_and_add(-1, 3, 4) == 1
    assert multiply_and_add(2, 0, 4) == 4