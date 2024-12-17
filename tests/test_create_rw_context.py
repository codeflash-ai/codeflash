from textwrap import dedent

import pytest
from codeflash.optimization.cst_context import create_read_write_context, find_containing_classes, print_tree


def test_simple_function():
    code = """
    def target_function():
        x = 1
        y = 2
        return x + y
    """
    root = find_containing_classes(dedent(code), {"target_function"})
    result = create_read_write_context(root)

    expected = dedent("""
    def target_function():
        x = 1
        y = 2
        return x + y
    """)
    assert result.strip() == expected.strip()


def test_class_method():
    code = """
    class MyClass:
        def target_function(self):
            x = 1
            y = 2
            return x + y
    """
    root = find_containing_classes(dedent(code), {"MyClass.target_function"})
    result = create_read_write_context(root)

    expected = dedent("""
    class MyClass:
        def target_function(self):
            x = 1
            y = 2
            return x + y
    """)
    assert result.strip() == expected.strip()


def test_class_with_attributes():
    code = """
    class MyClass:
        x: int = 1
        y: str = "hello"

        def target_method(self):
            return self.x + 42

        def other_method(self):
            print("this should be excluded")
    """
    root = find_containing_classes(dedent(code), {"MyClass.target_method"})
    result = create_read_write_context(root)

    expected = dedent("""
    class MyClass:
    
        def target_method(self):
            return self.x + 42
    """)
    assert result.strip() == expected.strip()


def test_basic_class_structure():
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
    root = find_containing_classes(dedent(code), {"Outer.target_method"})
    result = create_read_write_context(root)

    expected = dedent("""
    class Outer:
        def target_method(self):
            return 42
    """)
    assert result.strip() == expected.strip()


def test_top_level_targets():
    code = """
    class OuterClass:
        x = 1
        def method1(self):
            return self.x

    def target_function():
        return 42
    """
    root = find_containing_classes(dedent(code), {"target_function"})
    result = create_read_write_context(root)

    expected = dedent("""
    def target_function():
        return 42
    """)
    assert result.strip() == expected.strip()


def test_multiple_top_level_classes():
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
    root = find_containing_classes(dedent(code), {"ClassA.process", "ClassC.process"})
    result = create_read_write_context(root)

    expected = dedent("""
    class ClassA:
        def process(self):
            return "A"

    class ClassC:
        def process(self):
            return "C"
    """)
    assert result.strip() == expected.strip()


def test_try_except_structure():
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
    root = find_containing_classes(dedent(code), {"TargetClass.target_method"})
    print_tree(root)
    result = create_read_write_context(root)

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


def test_dunder_method():
    code = """
    class MyClass:
        def __init__(self):
            self.x = 1

        def other_method(self):
            return "other"

        def target_method(self):
            return f"Value: {self.x}"
    """
    root = find_containing_classes(dedent(code), {"MyClass.target_method"})
    result = create_read_write_context(root)

    expected = dedent("""
    class MyClass:
    
        def target_method(self):
            return f"Value: {self.x}"
    """)
    assert result.strip() == expected.strip()


def test_no_targets_found():
    code = """
    class MyClass:
        def method(self):
            pass

        class Inner:
            def target(self):
                pass
    """
    with pytest.raises(ValueError, match="No target functions found in the provided code"):
        find_containing_classes(dedent(code), {"MyClass.Inner.target"})
