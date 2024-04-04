import tempfile

from codeflash.code_utils.code_extractor import get_code
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize


def test_get_code_function() -> None:
    code = """def test(self):
    return self._test"""

    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()

        new_code = get_code([FunctionToOptimize("test", f.name, [])])
        assert new_code == code


def test_get_code_property() -> None:
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
            [
                FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")]),
            ],
        )
        assert new_code == code


def test_get_code_class() -> None:
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
            [FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")])],
        )
        assert new_code == expected


def test_get_code_bubble_sort_class() -> None:
    code = """
def hi():
    pass


class BubbleSortClass:
    def __init__(self):
        pass

    def __call__(self):
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
    def __call__(self):
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
            [FunctionToOptimize("sorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")])],
        )
        assert new_code == expected


def test_get_code_indent() -> None:
    code = """def hi():
    pass

def hello():
    pass

class BubbleSortClass:
    def __init__(self):
        pass

    def unsorter(self, arr):
        return shuffle(arr)

    def __call__(self):
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

def oui():
    pass

def non():
    pass

    """
    expected = """class BubbleSortClass:
    def __init__(self):
        pass
    def __call__(self):
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
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()
        new_code = get_code(
            [
                FunctionToOptimize(
                    "sorter",
                    f.name,
                    [FunctionParent("BubbleSortClass", "ClassDef")],
                ),
                FunctionToOptimize(
                    "helper",
                    f.name,
                    [FunctionParent("BubbleSortClass", "ClassDef")],
                ),
            ],
        )
    assert new_code == expected

    expected2 = """class BubbleSortClass:
    def __init__(self):
        pass
    def __call__(self):
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
    def unsorter(self, arr):
        return shuffle(arr)
"""
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(code)
        f.flush()
        new_code = get_code(
            [
                FunctionToOptimize(
                    "sorter",
                    f.name,
                    [FunctionParent("BubbleSortClass", "ClassDef")],
                ),
                FunctionToOptimize(
                    "helper",
                    f.name,
                    [FunctionParent("BubbleSortClass", "ClassDef")],
                ),
                FunctionToOptimize(
                    "unsorter",
                    f.name,
                    [FunctionParent("BubbleSortClass", "ClassDef")],
                ),
            ],
        )
        assert new_code == expected2
