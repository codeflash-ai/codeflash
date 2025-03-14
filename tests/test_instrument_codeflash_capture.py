from __future__ import annotations

from codeflash.benchmarking.instrument_codeflash_trace import add_codeflash_decorator_to_code


def test_add_decorator_to_normal_function() -> None:
    """Test adding decorator to a normal function."""
    code = """
def normal_function():
    return "Hello, World!"
"""

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="normal_function"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="normal_method",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="class_method",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="static_method",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="__init__",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="property_method",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="test_method",
        class_name="TestClass"
    )

    expected_code = """
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

    modified_code = add_codeflash_decorator_to_code(
        code=code,
        function_name="nonexistent_function"
    )

    # Code should remain unchanged
    assert modified_code.strip() == code.strip()
