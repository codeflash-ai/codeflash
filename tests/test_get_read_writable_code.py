from textwrap import dedent

import pytest
from codeflash.context.code_context_extractor import  parse_code_and_prune_cst
from codeflash.models.models import CodeContextType


def test_simple_function() -> None:
    code = """
    def target_function():
        x = 1
        y = 2
        return x + y
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"target_function"})

    expected = dedent("""
    def target_function():
        x = 1
        y = 2
        return x + y
    """)
    assert result.strip() == expected.strip()


def test_class_method() -> None:
    code = """
    class MyClass:
        def target_function(self):
            x = 1
            y = 2
            return x + y
    """
    result = parse_code_and_prune_cst(dedent(code), CodeContextType.READ_WRITABLE, {"MyClass.target_function"})

    expected = dedent("""
    class MyClass:
        def target_function(self):
            x = 1
            y = 2
            return x + y
    """)
    assert result.strip() == expected.strip()


def test_class_with_attributes() -> None:
    code = """
    class MyClass:
        x: int = 1
        y: str = "hello"

        def target_method(self):
            return self.x + 42

        def other_method(self):
            print("this should be excluded")
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"MyClass.target_method"})

    expected = dedent("""
    class MyClass:

        def target_method(self):
            return self.x + 42
    """)
    assert result.strip() == expected.strip()


def test_basic_class_structure() -> None:
    """Test that nested classes are ignored for target function search."""
    code = """
    class Outer:
        x = 1
        def target_method(self):
            return 42

        class Inner:
            y = 2
            def not_findable(self):
                return 42
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"Outer.target_method"})

    expected = dedent("""
    class Outer:
        def target_method(self):
            return 42
    """)
    assert result.strip() == expected.strip()


def test_top_level_targets() -> None:
    code = """
    class OuterClass:
        x = 1
        def method1(self):
            return self.x

    def target_function():
        return 42
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"target_function"})

    expected = dedent("""
    def target_function():
        return 42
    """)
    assert result.strip() == expected.strip()


def test_multiple_top_level_classes() -> None:
    code = """
    class ClassA:
        def process(self):
            return "A"

    class ClassB:
        def process(self):
            return "B"

    class ClassC:
        def process(self):
            return "C"
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"ClassA.process", "ClassC.process"})

    expected = dedent("""
    class ClassA:
        def process(self):
            return "A"

    class ClassC:
        def process(self):
            return "C"
    """)
    assert result.strip() == expected.strip()


def test_try_except_structure() -> None:
    code = """
    try:
        class TargetClass:
            def target_method(self):
                return 42
    except ValueError:
        class ErrorClass:
            def handle_error(self):
                print("error")
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"TargetClass.target_method"})

    expected = dedent("""
    try:
        class TargetClass:
            def target_method(self):
                return 42
    except ValueError:
        class ErrorClass:
            def handle_error(self):
                print("error")
    """)
    assert result.strip() == expected.strip()


def test_init_method() -> None:
    code = """
    class MyClass:
        def __init__(self):
            self.x = 1

        def other_method(self):
            return "other"

        def target_method(self):
            return f"Value: {self.x}"
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"MyClass.target_method"})

    expected = dedent("""
    class MyClass:
        def __init__(self):
            self.x = 1

        def target_method(self):
            return f"Value: {self.x}"
    """)
    assert result.strip() == expected.strip()

def test_dunder_method() -> None:
    code = """
    class MyClass:
        def __repr__(self):
            return "MyClass"

        def other_method(self):
            return "other"

        def target_method(self):
            return f"Value: {self.x}"
    """
    result = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"MyClass.target_method"})

    expected = dedent("""
    class MyClass:

        def target_method(self):
            return f"Value: {self.x}"
    """)
    assert result.strip() == expected.strip()

def test_no_targets_found() -> None:
    code = """
    class MyClass:
        def method(self):
            pass

        class Inner:
            def target(self):
                pass
    """
    with pytest.raises(ValueError, match="No target functions found in the provided code"):
        parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"MyClass.Inner.target"})


def test_module_var() -> None:
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
    def target_function(self) -> None:
        var2 = "test"
    """

    output = parse_code_and_prune_cst(dedent(code),CodeContextType.READ_WRITABLE, {"target_function"})
    assert dedent(expected).strip() == output.strip()


