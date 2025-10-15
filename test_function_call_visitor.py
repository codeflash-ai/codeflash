"""
Test and demonstrate the FunctionCallVisitor capabilities.
"""

import ast
from function_call_visitor import FunctionCallVisitor, analyze_code, analyze_file


def test_basic_calls():
    """Test basic function call detection."""
    code = """
def example():
    print("Hello")
    len([1, 2, 3])
    max([4, 5, 6])
    print("World")
"""
    results = analyze_code(code, ['print', 'len'])
    print("Test: Basic Calls")
    print(f"  Found {results['total_calls']} calls")
    for call in results['all_calls']:
        print(f"    {call}")
    print()


def test_loop_detection():
    """Test detection of calls within loops."""
    code = """
def process():
    print("Start")  # Outside loop

    for i in range(10):
        print(f"Item {i}")  # In for loop
        len(str(i))  # In for loop

    x = 0
    while x < 5:
        print(f"While {x}")  # In while loop
        x += len([1, 2])  # In while loop

    print("End")  # Outside loop
"""
    results = analyze_code(code, ['print', 'len'])
    print("Test: Loop Detection")
    print(f"  Total calls: {results['total_calls']}")
    print(f"  In loops: {results['calls_in_loops']}")
    print(f"  Outside loops: {results['calls_outside_loops']}")
    print("  Loop calls:")
    for call in results['loop_calls']:
        print(f"    {call}")
    print()


def test_nested_loops():
    """Test detection in nested loops."""
    code = """
def nested():
    for i in range(3):
        print(f"Outer {i}")
        for j in range(2):
            print(f"Inner {i},{j}")
            while j < 1:
                print(f"Innermost")
                j += 1
"""
    results = analyze_code(code, ['print'])
    print("Test: Nested Loops")
    for call in results['all_calls']:
        print(f"  {call}")
    print()


def test_method_calls():
    """Test detection of method calls."""
    code = """
class MyClass:
    def __init__(self):
        self.data = []

    def process(self):
        for item in [1, 2, 3]:
            self.data.append(item)
            self.validate(item)

    def validate(self, item):
        if len(str(item)) > 0:
            self.data.append(item * 2)

    @classmethod
    def create(cls):
        instance = cls()
        instance.data.append(0)
        return instance

    @staticmethod
    def helper():
        result = []
        result.append(1)
        return result

obj = MyClass()
obj.process()
obj.data.append(99)
MyClass.create()
MyClass.helper()
"""
    results = analyze_code(code, ['append', 'validate', 'len'])
    print("Test: Method Calls")
    print(f"  Found {results['total_calls']} calls")
    for call in results['all_calls']:
        print(f"    {call}")
    print()


def test_module_calls():
    """Test detection of module function calls."""
    code = """
import os.path
import numpy as np
from math import sqrt

def example():
    # Module function calls
    os.path.join("a", "b")
    np.array([1, 2, 3])
    sqrt(16)

    for i in range(3):
        os.path.exists(f"file_{i}")
        np.zeros((2, 2))

    # Nested module calls
    result = os.path.dirname(os.path.join("x", "y"))
"""
    results = analyze_code(code, ['os.path.join', 'np.array', 'sqrt', 'os.path.exists', 'np.zeros', 'os.path.dirname'])
    print("Test: Module Calls")
    print(f"  Total calls: {results['total_calls']}")
    print("  All calls:")
    for call in results['all_calls']:
        print(f"    {call}")
    print()


def test_complex_expressions():
    """Test calls in complex expressions."""
    code = """
def complex_example():
    # Calls in list comprehensions
    result = [len(x) for x in ["a", "bb", "ccc"]]

    # Calls in generator expressions
    gen = (print(x) for x in range(3))

    # Nested calls
    value = max(len("hello"), len("world"))

    # Calls in lambda
    func = lambda x: len(x) + len(x.strip())

    # Calls in conditionals
    if len("test") > 0:
        print("Has length")

    # Calls in dict comprehensions
    d = {x: len(x) for x in ["key1", "key2"]}
"""
    results = analyze_code(code, ['len', 'print', 'max'])
    print("Test: Complex Expressions")
    print(f"  Found {results['total_calls']} calls")
    for call in results['all_calls']:
        print(f"    {call}")
    print()


def test_async_code():
    """Test async function calls."""
    code = """
async def async_example():
    print("Starting async")

    async for item in async_generator():
        print(f"Processing {item}")
        await process_item(item)

    print("Done")

async def async_generator():
    for i in range(3):
        yield i

async def process_item(item):
    print(f"Item: {item}")
"""
    results = analyze_code(code, ['print', 'process_item'])
    print("Test: Async Code")
    for call in results['all_calls']:
        print(f"  {call}")
    print()


def test_partial_matching():
    """Test partial name matching."""
    code = """
import os
import os.path
from pathlib import Path

def file_operations():
    # These should all be caught when looking for 'join'
    os.path.join("a", "b")
    # path.join("c", "d")  # Would need path to be defined
    # something.else.join("x")  # Would need something to be defined

    # Looking for any 'append' method
    list1 = []
    list1.append(1)
    list2 = []
    list2.append(2)
    # some_obj.data.append(3)  # Would need some_obj to be defined
"""
    results = analyze_code(code, ['join', 'append'])
    print("Test: Partial Matching")
    print(f"  Tracking 'join' and 'append'")
    for call in results['all_calls']:
        print(f"    {call}")
    print()


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("FunctionCallVisitor Test Suite")
    print("=" * 60)
    print()

    test_basic_calls()
    test_loop_detection()
    test_nested_loops()
    test_method_calls()
    test_module_calls()
    test_complex_expressions()
    test_async_code()
    test_partial_matching()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

    # Example of analyzing an actual file
    print("\nExample: Analyzing the visitor file itself")
    print("-" * 60)
    try:
        results = analyze_file("function_call_visitor.py", ['isinstance', 'append', 'len'])
        print(f"Found {results['total_calls']} calls in function_call_visitor.py")
        print(f"  In loops: {results['calls_in_loops']}")
        print(f"  Outside loops: {results['calls_outside_loops']}")
        if results['loop_calls']:
            print("\nCalls in loops:")
            for call in results['loop_calls'][:5]:  # Show first 5
                print(f"  {call}")
    except FileNotFoundError:
        print("  (File not found - run from the same directory)")