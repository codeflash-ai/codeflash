from __future__ import annotations

import tempfile
from pathlib import Path

from codeflash.benchmarking.instrument_codeflash_trace import add_codeflash_decorator_to_code, \
    instrument_codeflash_trace_decorator
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize


def test_add_decorator_to_normal_function() -> None:
    """Test adding decorator to a normal function."""
    code = """
def normal_function():
    return "Hello, World!"
"""

    fto = FunctionToOptimize(
        function_name="normal_function",
        file_path=Path("dummy_path.py"),
        parents=[]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
@codeflash_trace
def normal_function():
    return "Hello, World!"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_normal_method() -> None:
    """Test adding decorator to a normal method."""
    code = """
class TestClass:
    def normal_method(self):
        return "Hello from method"
"""

    fto = FunctionToOptimize(
        function_name="normal_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @codeflash_trace
    def normal_method(self):
        return "Hello from method"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_classmethod() -> None:
    """Test adding decorator to a classmethod."""
    code = """
class TestClass:
    @classmethod
    def class_method(cls):
        return "Hello from classmethod"
"""

    fto = FunctionToOptimize(
        function_name="class_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @classmethod
    @codeflash_trace
    def class_method(cls):
        return "Hello from classmethod"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_staticmethod() -> None:
    """Test adding decorator to a staticmethod."""
    code = """
class TestClass:
    @staticmethod
    def static_method():
        return "Hello from staticmethod"
"""

    fto = FunctionToOptimize(
        function_name="static_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @staticmethod
    @codeflash_trace
    def static_method():
        return "Hello from staticmethod"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_init_function() -> None:
    """Test adding decorator to an __init__ function."""
    code = """
class TestClass:
    def __init__(self, value):
        self.value = value
"""

    fto = FunctionToOptimize(
        function_name="__init__",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @codeflash_trace
    def __init__(self, value):
        self.value = value
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_with_multiple_decorators() -> None:
    """Test adding decorator to a function with multiple existing decorators."""
    code = """
class TestClass:
    @property
    @other_decorator
    def property_method(self):
        return self._value
"""

    fto = FunctionToOptimize(
        function_name="property_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @property
    @other_decorator
    @codeflash_trace
    def property_method(self):
        return self._value
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_function_in_multiple_classes() -> None:
    """Test that only the right class's method gets the decorator."""
    code = """
class TestClass:
    def test_method(self):
        return "This should get decorated"

class OtherClass:
    def test_method(self):
        return "This should NOT get decorated"
"""

    fto = FunctionToOptimize(
        function_name="test_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="TestClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class TestClass:
    @codeflash_trace
    def test_method(self):
        return "This should get decorated"

class OtherClass:
    def test_method(self):
        return "This should NOT get decorated"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_nonexistent_function() -> None:
    """Test that code remains unchanged when function doesn't exist."""
    code = """
def existing_function():
    return "This exists"
"""

    fto = FunctionToOptimize(
        function_name="nonexistent_function",
        file_path=Path("dummy_path.py"),
        parents=[]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    # Code should remain unchanged
    assert modified_code.strip() == code.strip()


def test_add_decorator_to_multiple_functions() -> None:
    """Test adding decorator to multiple functions."""
    code = """
def function_one():
    return "First function"

class TestClass:
    def method_one(self):
        return "First method"

    def method_two(self):
        return "Second method"

def function_two():
    return "Second function"
"""

    functions_to_optimize = [
        FunctionToOptimize(
            function_name="function_one",
            file_path=Path("dummy_path.py"),
            parents=[]
        ),
        FunctionToOptimize(
            function_name="method_two",
            file_path=Path("dummy_path.py"),
            parents=[FunctionParent(name="TestClass", type="ClassDef")]
        ),
        FunctionToOptimize(
            function_name="function_two",
            file_path=Path("dummy_path.py"),
            parents=[]
        )
    ]

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=functions_to_optimize
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
@codeflash_trace
def function_one():
    return "First function"

class TestClass:
    def method_one(self):
        return "First method"

    @codeflash_trace
    def method_two(self):
        return "Second method"

@codeflash_trace
def function_two():
    return "Second function"
"""

    assert modified_code.strip() == expected_code.strip()


def test_instrument_codeflash_trace_decorator_single_file() -> None:
    """Test instrumenting codeflash trace decorator on a single file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test Python file
        test_file_path = Path(temp_dir) / "test_module.py"
        test_file_content = """
def function_one():
    return "First function"

class TestClass:
    def method_one(self):
        return "First method"

    def method_two(self):
        return "Second method"

def function_two():
    return "Second function"
"""
        test_file_path.write_text(test_file_content, encoding="utf-8")

        # Define functions to optimize
        functions_to_optimize = [
            FunctionToOptimize(
                function_name="function_one",
                file_path=test_file_path,
                parents=[]
            ),
            FunctionToOptimize(
                function_name="method_two",
                file_path=test_file_path,
                parents=[FunctionParent(name="TestClass", type="ClassDef")]
            )
        ]

        # Execute the function being tested
        instrument_codeflash_trace_decorator({test_file_path: functions_to_optimize})

        # Read the modified file
        modified_content = test_file_path.read_text(encoding="utf-8")

        # Define expected content (with isort applied)
        expected_content = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace


@codeflash_trace
def function_one():
    return "First function"

class TestClass:
    def method_one(self):
        return "First method"

    @codeflash_trace
    def method_two(self):
        return "Second method"

def function_two():
    return "Second function"
"""

        # Compare the modified content with expected content
        assert modified_content.strip() == expected_content.strip()


def test_instrument_codeflash_trace_decorator_multiple_files() -> None:
    """Test instrumenting codeflash trace decorator on multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first test Python file
        test_file_1_path = Path(temp_dir) / "module_a.py"
        test_file_1_content = """
def function_a():
    return "Function in module A"

class ClassA:
    def method_a(self):
        return "Method in ClassA"
"""
        test_file_1_path.write_text(test_file_1_content, encoding="utf-8")

        # Create second test Python file
        test_file_2_path = Path(temp_dir) / "module_b.py"
        test_file_2_content ="""
def function_b():
    return "Function in module B"

class ClassB:
    @staticmethod
    def static_method_b():
        return "Static method in ClassB"
"""
        test_file_2_path.write_text(test_file_2_content, encoding="utf-8")

        # Define functions to optimize
        file_to_funcs_to_optimize = {
            test_file_1_path: [
                FunctionToOptimize(
                    function_name="function_a",
                    file_path=test_file_1_path,
                    parents=[]
                )
            ],
            test_file_2_path: [
                FunctionToOptimize(
                    function_name="static_method_b",
                    file_path=test_file_2_path,
                    parents=[FunctionParent(name="ClassB", type="ClassDef")]
                )
            ]
        }

        # Execute the function being tested
        instrument_codeflash_trace_decorator(file_to_funcs_to_optimize)

        # Read the modified files
        modified_content_1 = test_file_1_path.read_text(encoding="utf-8")
        modified_content_2 = test_file_2_path.read_text(encoding="utf-8")

        # Define expected content for first file (with isort applied)
        expected_content_1 = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace


@codeflash_trace
def function_a():
    return "Function in module A"

class ClassA:
    def method_a(self):
        return "Method in ClassA"
"""

        # Define expected content for second file (with isort applied)
        expected_content_2 = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace


def function_b():
    return "Function in module B"

class ClassB:
    @staticmethod
    @codeflash_trace
    def static_method_b():
        return "Static method in ClassB"
"""

        # Compare the modified content with expected content
        assert modified_content_1.strip() == expected_content_1.strip()
        assert modified_content_2.strip() == expected_content_2.strip()


def test_add_decorator_to_method_after_nested_class() -> None:
    """Test adding decorator to a method that appears after a nested class definition."""
    code = """
class OuterClass:
    class NestedClass:
        def nested_method(self):
            return "Hello from nested class method"

    def target_method(self):
        return "Hello from target method after nested class"
"""

    fto = FunctionToOptimize(
        function_name="target_method",
        file_path=Path("dummy_path.py"),
        parents=[FunctionParent(name="OuterClass", type="ClassDef")]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
class OuterClass:
    class NestedClass:
        def nested_method(self):
            return "Hello from nested class method"

    @codeflash_trace
    def target_method(self):
        return "Hello from target method after nested class"
"""

    assert modified_code.strip() == expected_code.strip()


def test_add_decorator_to_function_after_nested_function() -> None:
    """Test adding decorator to a function that appears after a function with a nested function."""
    code = """
def function_with_nested():
    def inner_function():
        return "Hello from inner function"

    return inner_function()

def target_function():
    return "Hello from target function after nested function"
"""

    fto = FunctionToOptimize(
        function_name="target_function",
        file_path=Path("dummy_path.py"),
        parents=[]
    )

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        functions_to_optimize=[fto]
    )

    expected_code = """
from codeflash.benchmarking.codeflash_trace import codeflash_trace
def function_with_nested():
    def inner_function():
        return "Hello from inner function"

    return inner_function()

@codeflash_trace
def target_function():
    return "Hello from target function after nested function"
"""

    assert modified_code.strip() == expected_code.strip()