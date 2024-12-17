from textwrap import dedent

import pytest
from codeflash.optimization.cst_context import create_read_only_context, find_containing_classes, print_tree


def test_basic_class():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_dunder_methods():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_target_in_nested_class():
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
        find_containing_classes(dedent(code), {"Outer.Inner.target_method"})


def test_docstrings():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_method_signatures():
    code = """
    class TestClass:
        @property
        def target_method(self) -> str:
            \"\"\"Property docstring.\"\"\"
            return "value"

        @classmethod
        def class_method(cls, param: int = 42) -> None:
            print("stub this")
    """

    expected = """"""

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    print(output)
    assert dedent(expected).strip() == output.strip()


def test_multiple_top_level_targets():
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

    result = find_containing_classes(dedent(code), {"TestClass.target1", "TestClass.target2"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_class_annotations():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_if():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_try():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_class_annotations_else():
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

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    print_tree(result)
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_top_level_functions():
    code = """
    def target_function(self) -> None:
        self.var2 = "test"
    
    def some_function():
        print("wow")
    """

    expected = """"""

    result = find_containing_classes(dedent(code), {"target_function"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_module_scope_var():
    code = """
    y = 3
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("stub me")
    """

    expected = """
    y = 3
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

    """

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()


def test_module_scope_var():
    code = """
    if True:
        y = 3
        
        class OtherClass:
            def this_method(self):
                print("this method")
            def __init__(self):
                self.y = y
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

        def target_method(self):
            print("stub me")
    """

    expected = """
    if True:
        y = 3
    class TestClass:
        def __init__(self):
            self.x = 42

        def __str__(self):
            return f"Value: {self.x}"

    """

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    output = create_read_only_context(result)
    assert dedent(expected).strip() == output.strip()
