import tempfile
import os
from pathlib import Path

from codeflash.code_utils.code_extractor import get_code
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
def test_get_code_function() -> None:
    code = """def test(self):
    return self._test"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        new_code, contextual_dunder_methods = get_code([FunctionToOptimize("test", temp_file_path, [])])
        assert new_code == code
        assert contextual_dunder_methods == set()


def test_get_code_property() -> None:
    code = """class TestClass:
    def __init__(self):
        self._test = 5
    @property
    def test(self):
        return self._test"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("test", temp_file_path, [FunctionParent("TestClass", "ClassDef")])]
        )
        assert new_code == code
        assert contextual_dunder_methods == {("TestClass", "__init__")}


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
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("test", temp_file_path, [FunctionParent("TestClass", "ClassDef")])]
        )
        assert new_code == expected
        assert contextual_dunder_methods == {("TestClass", "__init__")}


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
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("sorter", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")])]
        )
        assert new_code == expected
        assert contextual_dunder_methods == {("BubbleSortClass", "__init__"), ("BubbleSortClass", "__call__")}


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
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()
        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize("sorter", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("helper", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")]),
            ]
        )
    assert new_code == expected
    assert contextual_dunder_methods == {("BubbleSortClass", "__init__"), ("BubbleSortClass", "__call__")}

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
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()
        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize("sorter", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("helper", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("unsorter", temp_file_path, [FunctionParent("BubbleSortClass", "ClassDef")]),
            ]
        )
        assert new_code == expected2
        assert contextual_dunder_methods == {("BubbleSortClass", "__init__"), ("BubbleSortClass", "__call__")}


def test_get_code_multiline_class_def() -> None:
    code = """class StatementAssignmentVariableConstantMutable(
    StatementAssignmentVariableMixin, StatementAssignmentVariableConstantMutableBase
):
    kind = "STATEMENT_ASSIGNMENT_VARIABLE_CONSTANT_MUTABLE"

    def postInitNode(self):
        self.variable_trace = None
        self.inplace_suspect = None

    def computeStatement(self, trace_collection):
        return self, None, None

    @staticmethod
    def hasVeryTrustedValue():
        return False
"""
    expected = """class StatementAssignmentVariableConstantMutable(
    StatementAssignmentVariableMixin, StatementAssignmentVariableConstantMutableBase
):
    def computeStatement(self, trace_collection):
        return self, None, None
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize(
                    "computeStatement",
                    temp_file_path,
                    [FunctionParent("StatementAssignmentVariableConstantMutable", "ClassDef")],
                )
            ]
        )
        assert new_code == expected
        assert contextual_dunder_methods == set()


def test_get_code_dataclass_attribute():
    code = """@dataclass
class CustomDataClass:
    name: str = ""
    data: List[int] = field(default_factory=list)"""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / "temp_file.py"
        with temp_file_path.open("w") as f:
            f.write(code)
            f.flush()

        # This is not something that should ever happen with the current implementation, as get_code only runs with a
        # single FunctionToOptimize instance, in the case where that instance has been filtered to represent a function
        # (with a definition).
        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("name", temp_file_path, [FunctionParent("CustomDataClass", "ClassDef")])]
        )
        assert new_code is None
        assert contextual_dunder_methods == set()
