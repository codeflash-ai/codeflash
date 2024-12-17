from textwrap import dedent

import libcst as cst
from codeflash.optimization.cst_context import find_containing_classes, print_tree


def test_basic_function_identification():
    """Test that target functions are correctly identified."""
    code = """
    def target_func():
        pass

    def other_func():
        pass
    """

    result = find_containing_classes(dedent(code), {"target_func"})
    # Should find exactly one function
    assert len(result.children["body"]) == 1
    assert result.children["body"][0].is_target_function
    assert isinstance(result.children["body"][0].cst_node, cst.FunctionDef)
    assert result.children["body"][0].cst_node.name.value == "target_func"


def test_class_with_target_function():
    """Test that classes containing target functions are included with only the class definition."""
    code = """
    class TestClass:
        def target_method(self):
            pass

        def other_method(self):
            pass
    """

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})
    print_tree(result)
    assert len(result.children["body"]) == 1
    class_node = result.children["body"][0]
    assert isinstance(class_node.cst_node, cst.ClassDef)
    assert class_node.cst_node.name.value == "TestClass"

    assert len(class_node.children["body"]) == 1
    assert class_node.children["body"][0].is_target_function
    assert class_node.children["body"][0].cst_node.name.value == "target_method"


def test_class_without_target_function():
    """Test that classes without target functions are not included."""
    code = """
    class TestClass:
        def method1(self):
            pass

        def method2(self):
            pass
    """

    result = find_containing_classes(dedent(code), {"other_func"})

    # Should have no children since no target functions were found
    assert "body" not in result.children or not result.children["body"]


def test_control_flow_structures():
    """Test handling of control flow structures containing target functions."""
    code = """
    if True:
        def target_func():
            pass
    else:
        def other_func():
            pass

    try:
        def another_target():
            pass
    except Exception:
        def handler_func():
            pass
    finally:
        def cleanup_func():
            pass
    """

    result = find_containing_classes(dedent(code), {"target_func", "another_target"})

    assert len(result.children["body"]) == 2
    if_context_node = result.children["body"][0]
    assert isinstance(if_context_node.cst_node, cst.If)
    target_context_node = if_context_node.children["body"][0]
    assert target_context_node.cst_node.name.value == "target_func"
    assert target_context_node.is_target_function

    try_context_node = result.children["body"][1]
    assert isinstance(try_context_node.cst_node, cst.Try)
    another_target_context_node = try_context_node.children["body"][0]
    assert another_target_context_node.cst_node.name.value == "another_target"
    assert another_target_context_node.is_target_function


def test_nested_classes():
    """Test handling of nested classes with target functions."""
    code = """
    class OuterClass:
        class InnerClass:
            def target_method(self):
                pass

        def other_method(self):
            pass
    """

    result = find_containing_classes(dedent(code), {"OuterClass.InnerClass.target_method"})

    # Verify the class hierarchy
    assert len(result.children["body"]) == 1
    outer_class = result.children["body"][0]
    assert isinstance(outer_class.cst_node, cst.ClassDef)
    assert outer_class.cst_node.name.value == "OuterClass"

    assert len(outer_class.children["body"]) == 1
    inner_class = outer_class.children["body"][0]
    assert isinstance(inner_class.cst_node, cst.ClassDef)
    assert inner_class.cst_node.name.value == "InnerClass"

    assert len(inner_class.children["body"]) == 1
    target_method = inner_class.children["body"][0]
    assert target_method.is_target_function
    assert target_method.cst_node.name.value == "target_method"


def test_no_classes():
    """Test handling of target functions without any classes."""
    code = """
    def function1():
        pass

    def target_function():
        def nested_function():
            pass
        return nested_function

    def function2():
        pass
    """

    result = find_containing_classes(dedent(code), {"target_function"})

    # Should find only the target function
    assert len(result.children["body"]) == 1
    assert result.children["body"][0].is_target_function
    assert result.children["body"][0].cst_node.name.value == "target_function"


def test_no_classes_if_else():
    """Test handling of target functions in if/else blocks."""
    code = """
    def function1():
        pass
    if x:
        def target_function():
            return "hello"
    else:
        def function2():
            pass
    """

    result = find_containing_classes(dedent(code), {"target_function"})

    assert result.children["body"][0].children["body"][0].is_target_function
    assert result.children["body"][0].children["body"][0].cst_node.name.value == "target_function"


def test_no_classes_else():
    """Test handling of target functions in else blocks."""
    code = """
    def function1():
        pass
    if x:
        x += 2
    else:
        def target_function():
            return "hello"
    """

    result = find_containing_classes(dedent(code), {"target_function"})

    assert result.children["body"][0].children["orelse"][0].is_target_function
    assert result.children["body"][0].children["orelse"][0].cst_node.name.value == "target_function"


def test_comments_and_decorators():
    """Test that comments and decorators are preserved."""
    code = """
    # Top level comment
    @decorator
    class TestClass:
        # Class comment
        @method_decorator
        def target_method(self):
            # Method comment
            pass
    """

    result = find_containing_classes(dedent(code), {"TestClass.target_method"})

    # Verify class has decorator
    assert len(result.children["body"]) == 1
    class_node = result.children["body"][0]
    assert len(class_node.cst_node.decorators) == 1

    # Verify method has decorator
    method_node = class_node.children["body"][0]
    assert len(method_node.cst_node.decorators) == 1


def test_same_name_different_paths():
    """Test handling of functions with same name but different qualified paths."""
    code = """
    class ClassA:
        def process(self):
            pass

    class ClassB:
        def process(self):
            pass

    def process():
        pass

    class Outer:
        class Inner:
            def process(self):
                pass

        def process(self):
            pass
    """

    # Test finding specific instances
    result = find_containing_classes(dedent(code), {"ClassA.process", "Outer.Inner.process"})

    # Test for just top-level process
    result_top = find_containing_classes(dedent(code), {"process"})
    assert len(result_top.children["body"]) == 1
    assert result_top.children["body"][0].is_target_function
    assert result_top.children["body"][0].cst_node.name.value == "process"

    # Test for just Outer.process
    result_outer = find_containing_classes(dedent(code), {"Outer.process"})
    assert result_outer.children["body"][0].cst_node.name.value == "Outer"
    assert result_outer.children["body"][0].children["body"][0].cst_node.name.value == "process"

    # Test for just Inner.process
    result_inner = find_containing_classes(dedent(code), {"Outer.Inner.process"})
    outer = result_inner.children["body"][0]
    assert outer.cst_node.name.value == "Outer"
    inner = outer.children["body"][0]
    assert inner.cst_node.name.value == "Inner"
    assert inner.children["body"][0].cst_node.name.value == "process"
