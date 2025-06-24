from code_to_optimize.bubble_sort_cuda import sorter_cuda
import pytest

@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([5, 4, 3, 2, 1, 0], [0, 1, 2, 3, 4, 5]),
        ([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (list(reversed(range(5000))), list(range(5000))),
    ],
)
def test_sort_parametrized(input, expected_output):
    output = sorter_cuda(input)
    assert output == expected_output

class TestSorter:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_sort_in_pytest_class(self):
        input = [5, 4, 3, 2, 1, 0]
        output = sorter_cuda(input)
        assert output == [0, 1, 2, 3, 4, 5]

        input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        output = sorter_cuda(input)
        assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

        input = list(reversed(range(5000)))
        output = sorter_cuda(input)
        assert output == list(range(5000))

def test_sorter_cuda():
    input = [5, 4, 3, 2, 1, 0]
    output = sorter_cuda(input)
    assert output == [0, 1, 2, 3, 4, 5]

    input = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    output = sorter_cuda(input)
    assert output == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    input = list(reversed(range(5000)))
    output = sorter_cuda(input)
    assert output == list(range(5000))

    input = [5, 4, 3, 2, 1, 0]
    if len(input) > 0:
        assert sorter_cuda(input) == [0, 1, 2, 3, 4, 5]