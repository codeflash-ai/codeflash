import tempfile

from codeflash.code_utils.code_extractor import get_code
from codeflash.discovery.functions_to_optimize import FunctionToOptimize, FunctionParent


def test_get_code_function():
    code = """def test(self):
    return self._test"""

    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()

        new_code = get_code(FunctionToOptimize("test", f.name, []))
        assert new_code == code


def test_get_code_property():
    code = """class TestClass:
    def __init__(self):
        self._test = 5
    @property
    def test(self):
        return self._test"""
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()

        new_code = get_code(
            FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")])
        )
        assert new_code == code


def test_get_code_class():
    code = """
class TestClass:
    def __init__(self):
        self._test = 5

    def test_method(self):
        return self._test + 1
    @property
    def test(self):
        return self._test"""

    expected = """class TestClass:
    def __init__(self):
        self._test = 5
    @property
    def test(self):
        return self._test"""
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()

        new_code = get_code(
            FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")])
        )
        assert new_code == expected


def test_get_code_bubble_sort_class():
    code = """
def hi():
    pass


class BubbleSortClass:
    def __init__(self):
        pass

    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr

    def helper(self, arr, j):
        return arr[j] > arr[j + 1]

    """
    expected = """class BubbleSortClass:
    def __init__(self):
        pass
    def sorter(self, arr):
        for i in range(len(arr)):
            for j in range(len(arr) - 1):
                if arr[j] > arr[j + 1]:
                    temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
        return arr
"""
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()

        new_code = get_code(
            FunctionToOptimize("sorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")])
        )
        assert new_code == expected
