import tempfile

from codeflash.code_utils.code_extractor import get_code
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import FunctionParent
import pytest
from pathlib import Path

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_get_code_function(temp_dir: Path) -> None:
    code = """def test(self):
    return self._test"""

    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        new_code, contextual_dunder_methods = get_code([FunctionToOptimize("test", f.name, [])])
        assert new_code == code
        assert contextual_dunder_methods == set()


def test_get_code_property(temp_dir: Path) -> None:
    code = """class TestClass:
    def __init__(self):
        self._test = 5
    @property
    def test(self):
        return self._test"""
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")])]
        )
        assert new_code == code
        assert contextual_dunder_methods == {("TestClass", "__init__")}


def test_get_code_class(temp_dir: Path) -> None:
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
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("test", f.name, [FunctionParent("TestClass", "ClassDef")])]
        )
        assert new_code == expected
        assert contextual_dunder_methods == {("TestClass", "__init__")}


def test_get_code_bubble_sort_class(temp_dir: Path) -> None:
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
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("sorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")])]
        )
        assert new_code == expected
        assert contextual_dunder_methods == {("BubbleSortClass", "__init__"), ("BubbleSortClass", "__call__")}


def test_get_code_indent(temp_dir: Path) -> None:
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
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()
        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize("sorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("helper", f.name, [FunctionParent("BubbleSortClass", "ClassDef")]),
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
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()
        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize("sorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("helper", f.name, [FunctionParent("BubbleSortClass", "ClassDef")]),
                FunctionToOptimize("unsorter", f.name, [FunctionParent("BubbleSortClass", "ClassDef")]),
            ]
        )
        assert new_code == expected2
        assert contextual_dunder_methods == {("BubbleSortClass", "__init__"), ("BubbleSortClass", "__call__")}


def test_get_code_multiline_class_def(temp_dir: Path) -> None:
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
    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        new_code, contextual_dunder_methods = get_code(
            [
                FunctionToOptimize(
                    "computeStatement",
                    f.name,
                    [FunctionParent("StatementAssignmentVariableConstantMutable", "ClassDef")],
                )
            ]
        )
        assert new_code == expected
        assert contextual_dunder_methods == set()


def test_get_code_dataclass_attribute(temp_dir: Path) -> None:
    code = """@dataclass
class CustomDataClass:
    name: str = ""
    data: List[int] = field(default_factory=list)"""

    with (temp_dir / "temp_file.py").open(mode="w") as f:
        f.write(code)
        f.flush()

        # This is not something that should ever happen with the current implementation, as get_code only runs with a
        # single FunctionToOptimize instance, in the case where that instance has been filtered to represent a function
        # (with a definition).
        new_code, contextual_dunder_methods = get_code(
            [FunctionToOptimize("name", f.name, [FunctionParent("CustomDataClass", "ClassDef")])]
        )
        assert new_code is None
        assert contextual_dunder_methods == set()
