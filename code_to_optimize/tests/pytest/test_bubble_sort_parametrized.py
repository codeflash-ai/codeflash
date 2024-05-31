import pytest

from code_to_optimize.bubble_sort import sorter


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (list(reversed(range(5000))), list(range(5000))),
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter(input)
    assert output == expected_output
