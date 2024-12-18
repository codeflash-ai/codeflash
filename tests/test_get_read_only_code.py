from textwrap import dedent

import pytest
from codeflash.optimization.cst_manipulator import get_read_only_code


def test_basic_class() -> None:
    code = """
    class TestClass:
        class_var = "value"

        def target_method(self):
            print("This should be stubbed")

        def other_method(self):
            print("This too")
    """

    expected = """
    class TestClass:
        class_var = "value"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_dunder_methods() -> None:
    code = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("stub me")
    """

    expected = """
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_target_in_nested_class() -> None:
    """Test that attempting to find a target in a nested class raises an error."""
    code = """
    class Outer:
        outer_var = 1

        class Inner:
            inner_var = 2

            def target_method(self):
                print("stub this")
    """

    with pytest.raises(ValueError, match="No target functions found in the provided code"):
        get_read_only_code(dedent(code), {"Outer.Inner.target_method"})


def test_docstrings() -> None:
    code = """
    class TestClass:
        \"\"\"Class docstring.\"\"\"

        def target_method(self):
            \"\"\"Method docstring.\"\"\"
            print("stub this")

        def other_method(self):
            \"\"\"Other docstring.\"\"\"
            print("stub this too")
    """

    expected = """
    class TestClass:
        \"\"\"Class docstring.\"\"\"
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_method_signatures() -> None:
    code = """
    class TestClass:
        @property
        def target_method(self) -> str:
            \"\"\"Property docstring.\"\"\"
            return "value"

        @classmethod
        def class_method(cls, param: int = 42) -> None:
            print("class method")
    """

    expected = """"""

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_multiple_top_level_targets() -> None:
    code = """
    class TestClass:
        def target1(self):
            print("stub 1")

        def target2(self):
            print("stub 2")

        def __init__(self):
            self.x = 42
    """

    expected = """
    class TestClass:

        def __init__(self):
            self.x = 42
    """

    output = get_read_only_code(dedent(code), {"TestClass.target1", "TestClass.target2"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations() -> None:
    code = """
    class TestClass:
        var1: int = 42
        var2: str

        def target_method(self) -> None:
            self.var2 = "test"
    """

    expected = """
    class TestClass:
        var1: int = 42
        var2: str
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_if() -> None:
    code = """
    if True:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    """

    expected = """
    if True:
        class TestClass:
            var1: int = 42
            var2: str
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_try() -> None:
    code = """
    try:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    except Exception:
        continue
    """

    expected = """
    try:
        class TestClass:
            var1: int = 42
            var2: str
    except Exception:
        continue
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_else() -> None:
    code = """
    if x is True:
        class TestClass:
            var1: int = 42
            var2: str

            def wrong_method(self) -> None:
                print("wrong")
    else:
        class TestClass:
            var1: int = 42
            var2: str

            def target_method(self) -> None:
                self.var2 = "test"
    """

    expected = """
    if x is True:
        class TestClass:
            var1: int = 42
            var2: str

            def wrong_method(self) -> None:
                print("wrong")
    else:
        class TestClass:
            var1: int = 42
            var2: str
    """

    output = get_read_only_code(dedent(code), {"TestClass.target_method"})
    assert dedent(expected).strip() == output.strip()


def test_top_level_functions() -> None:
    code = """
    def target_function(self) -> None:
        self.var2 = "test"

    def some_function():
        print("wow")
    """

    expected = """"""

    output = get_read_only_code(dedent(code), {"target_function"})
    assert dedent(expected).strip() == output.strip()


def test_module_var() -> None:
    code = """
    def target_function(self) -> None:
        self.var2 = "test"

    x = 5

    def some_function():
        print("wow")
    """

    expected = """
    x = 5
    """

    output = get_read_only_code(dedent(code), {"target_function"})
    assert dedent(expected).strip() == output.strip()


def test_module_var_if() -> None:
    code = """
    def target_function(self) -> None:
        var2 = "test"

    if y:
        x = 5
    else: 
        z = 10
        def some_function():
            print("wow")

    def some_function():
        print("wow")
    """

    expected = """
    if y:
        x = 5
    else: 
        z = 10
    """

    output = get_read_only_code(dedent(code), {"target_function"})
    assert dedent(expected).strip() == output.strip()
