from typing import Set

import libcst as cst
from codeflash.optimization.context_cst import (
    prune_module,  # replace 'yourmodule' with the actual module where prune_module is defined
)


def test_top_level_target_function():
    code = """
def foo():
    pass

def bar():
    pass
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"bar"}

    pruned = prune_module(module, targets)
    expected = """
def bar():
    pass
"""
    assert pruned.code.strip() == expected.strip()


def test_no_targets_found():
    code = """
def foo():
    pass

x = 10
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"bar"}  # 'bar' doesn't exist in code

    pruned = prune_module(module, targets)
    expected = ""  # no targets found, return empty module
    assert pruned.code.strip() == expected.strip()


def test_class_with_target_function():
    code = """
class MyClass:
    def helper(self):
        pass

    def target_method(self):
        return 42

def unrelated():
    pass
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"MyClass.target_method"}

    pruned = prune_module(module, targets)
    # We expect to keep MyClass and only target_method in it
    expected = """
class MyClass:

    def target_method(self):
        return 42
"""
    assert pruned.code.strip() == expected.strip()


def test_nested_class_with_target_function():
    code = """
class Outer:
    def outer_method(self):
        pass

    class Inner:
        def inner_helper(self):
            pass

        def target_func(self):
            print("Target")

def top_level():
    pass
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"Outer.Inner.target_func"}

    pruned = prune_module(module, targets)
    # We must keep Outer, Inner, and the target_func inside Inner
    expected = """
class Outer:

    class Inner:

        def target_func(self):
            print("Target")
"""
    assert pruned.code.strip() == expected.strip()


def test_if_statements_leading_to_target():
    code = """
def foo():
    if True:
        def target():
            return "yes"
    else:
        print("nope")

def bar():
    pass
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"target"}

    pruned = prune_module(module, targets)
    # We keep foo, because inside its if body there's the target function.
    # The else is removed as it doesn't lead to a target. bar is removed as well.
    expected = ""
    assert pruned.code.strip() == expected.strip()


def test_class_with_no_target_functions():
    code = """
class A:
    def no_target(self):
        pass

x = 5
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"SomeOtherClass.some_func"}

    pruned = prune_module(module, targets)
    expected = ""  # no targets in code, empty result
    assert pruned.code.strip() == expected.strip()


def test_class_in_else_block():
    code = """
if y is False:
    x = 10
else:
    class MyClass:
        def not_target(self):
            pass

        def target_func(self):
            return "found me!"
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"MyClass.target_func"}

    pruned = prune_module(module, targets)
    # Even though MyClass is in the else block, we have a target method inside it.
    # We expect to keep the wrapper function, the else block, and the class with just the target method.
    expected = """
if y is False:
    pass
else:
    class MyClass:

        def target_func(self):
            return "found me!"
"""
    assert pruned.code.strip() == expected.strip()


def test_class_in_if_block():
    code = """
if y is False:
    class MyClass:
        def not_target(self):
            pass

        def target_func(self):
            return "found me!"
else:
    x = 10
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"MyClass.target_func"}

    pruned = prune_module(module, targets)
    # Even though MyClass is in the else block, we have a target method inside it.
    # We expect to keep the wrapper function, the else block, and the class with just the target method.
    expected = """
if y is False:
    class MyClass:

        def target_func(self):
            return "found me!"
else:
    pass
"""
    assert pruned.code.strip() == expected.strip()


def test_functions_same_name_different_scopes():
    code = """
def foo():
    return "top-level"

class Outer:
    def foo():
        return "in Outer"

class Another:
    def foo():
        return "in Another"
"""
    module = cst.parse_module(code)
    # Only match the "Outer.foo" function, not the top-level "foo" or "Another.foo"
    targets: Set[str] = {"Outer.foo"}

    pruned = prune_module(module, targets)
    expected = """
class Outer:
    def foo():
        return "in Outer"
"""
    assert pruned.code.strip() == expected.strip()


def test_if_elif_else_block():
    code = """
if condition:
    def not_target():
        pass
elif other_condition:
    def another_not_target():
        pass
else:
    def target_in_else():
        return "Found"

def unrelated():
    return "no"
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"target_in_else"}

    pruned = prune_module(module, targets)
    # We keep the whole if-elif-else structure but prune out the non-target branches.
    expected = """
if condition:
    pass
elif other_condition:
    pass
else:
    def target_in_else():
        return "Found"
"""
    assert pruned.code.strip() == expected.strip()


def test_nested_if_in_else_block():
    code = """
if top_level:
    def no_target():
        return 1
else:
    if nested_condition:
        def target_func():
            return "nested target"
    else:
        def another_no():
            pass

def outside():
    pass
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"target_func"}

    pruned = prune_module(module, targets)
    # We keep the top if-else structure, and inside the else, we keep the nested if block with the target.
    expected = """
if top_level:
    pass
else:
    if nested_condition:
        def target_func():
            return "nested target"
"""
    assert pruned.code.strip() == expected.strip()


def test_class_in_try_block():
    code = """
try:
    class MyClass:
        def target_method(self):
            return "reached!"
except ValueError:
    def not_target():
        return "no"
else:
    def also_not_target():
        return "no"
finally:
    def final_not_target():
        return "no"
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"MyClass.target_method"}

    pruned = prune_module(module, targets)
    # We keep the try block because it contains the target class,
    # remove except, else, and finally since they don't lead to the target.
    expected = """
try:
    class MyClass:

        def target_method(self):
            return "reached!"
except ValueError:
    pass
else:
    pass
finally:
    pass
"""
    assert pruned.code.strip() == expected.strip()


def test_target_in_except_block():
    code = """
try:
    def no_target():
        return "no"
except KeyError:
    def target_func():
        return "caught target"
except ValueError:
    def also_no():
        return "no"
else:
    def no_again():
        return "no"
finally:
    def final_no():
        return "no"
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"target_func"}

    pruned = prune_module(module, targets)
    # Remove the try, else, and finally bodies that don't lead to the target.
    # Keep the except KeyError block with the target, and leave other except blocks as pass.
    expected = """
try:
    pass
except KeyError:
    def target_func():
        return "caught target"
except ValueError:
    pass
else:
    pass
finally:
    pass
"""
    assert pruned.code.strip() == expected.strip()


def test_target_in_multiple_except_blocks():
    code = """
try:
    def first_no():
        return 1
except KeyError:
    def second_no():
        return 2
except ValueError:
    def target_in_value():
        return 3
except TypeError:
    def another_no():
        return 4
else:
    def else_no():
        return 5
finally:
    def final_no():
        return 6
"""
    module = cst.parse_module(code)
    targets: Set[str] = {"target_in_value"}

    pruned = prune_module(module, targets)
    # We only keep the try/except/finally structure necessary to reach target_in_value.
    # The except for ValueError must remain with the target, others become pass.
    expected = """
try:
    pass
except KeyError:
    pass
except ValueError:
    def target_in_value():
        return 3
except TypeError:
    pass
else:
    pass
finally:
    pass
"""
    assert pruned.code.strip() == expected.strip()
