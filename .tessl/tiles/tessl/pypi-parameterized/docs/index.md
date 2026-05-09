# Parameterized

Parameterized testing with any Python test framework. This library provides comprehensive parameterized testing capabilities that eliminate repetitive test code by allowing developers to define test logic once and execute it across multiple parameter sets. It maintains compatibility with multiple test frameworks including nose, pytest, unittest, and unittest2.

## Package Information

- **Package Name**: parameterized
- **Language**: Python
- **Installation**: `pip install parameterized`
- **Requires Python**: >=3.7

## Core Imports

```python
from parameterized import parameterized, param, parameterized_class
```

Version information:

```python { .api }
__version__: str  # Package version ("0.9.0")
```

## Basic Usage

```python
from parameterized import parameterized, param, parameterized_class
import unittest
import math

# Basic parameterized test function
@parameterized([
    (2, 2, 4),
    (2, 3, 8),
    (1, 9, 1),
    (0, 9, 0),
])
def test_pow(base, exponent, expected):
    assert math.pow(base, exponent) == expected

# Parameterized test in unittest class
class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand([
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ])
    def test_floor(self, name, input, expected):
        self.assertEqual(math.floor(input), expected)

# Parameterized test class
@parameterized_class(('a', 'b', 'expected_sum', 'expected_product'), [
    (1, 2, 3, 2),
    (3, 4, 7, 12),
])
class TestArithmetic(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(self.a + self.b, self.expected_sum)
    
    def test_product(self):
        self.assertEqual(self.a * self.b, self.expected_product)
```

## Architecture

The parameterized library uses a decorator-based architecture that transforms test functions and classes at import time:

- **parameterized decorator**: Transforms functions into test generators for nose/pytest or standalone functions for unittest
- **param helper**: Represents individual parameter sets with positional and keyword arguments
- **Test runner detection**: Automatically detects the test framework and adapts behavior accordingly
- **Mock patch compatibility**: Maintains compatibility with mock.patch decorators through patch re-application

This design enables seamless integration with existing test suites while providing consistent parameterization across different test frameworks.

## Capabilities

### Function Parameterization

The core `@parameterized` decorator for creating parameterized test functions that work with nose, pytest, and standalone execution.

```python { .api }
class parameterized:
    def __init__(self, input, doc_func=None, skip_on_empty=False):
        """
        Initialize parameterized decorator.
        
        Parameters:
        - input: Iterable of parameter sets or callable returning iterable
        - doc_func: Function to generate test documentation (func, num, param) -> str
        - skip_on_empty: If True, skip test when input is empty; if False, raise ValueError
        """
    
    @classmethod
    def to_safe_name(cls, s):
        """
        Convert string to safe name for test case naming.
        
        Parameters:
        - s: String to convert
        
        Returns:
        String with non-alphanumeric characters replaced by underscores
        """
```

### Class Method Parameterization

The `@parameterized.expand` class method for parameterizing test methods within unittest.TestCase subclasses.

```python { .api }
@classmethod
def expand(cls, input, name_func=None, doc_func=None, skip_on_empty=False, namespace=None, **legacy):
    """
    Parameterize test methods in unittest.TestCase subclasses.
    
    Parameters:
    - input: Iterable of parameter sets or callable returning iterable
    - name_func: Function to generate test method names (func, num, param) -> str
    - doc_func: Function to generate test documentation (func, num, param) -> str
    - skip_on_empty: If True, skip test when input is empty; if False, raise ValueError
    - namespace: Namespace to inject test methods (defaults to caller's locals)
    
    Returns:
    Decorator function for test methods
    """
```

### Parameter Specification

The `param` class for specifying individual parameter sets with both positional and keyword arguments.

```python { .api }
class param:
    def __new__(cls, *args, **kwargs):
        """
        Create parameter set with positional and keyword arguments.
        
        Parameters:
        - *args: Positional arguments for test function
        - **kwargs: Keyword arguments for test function
        
        Returns:
        param instance containing args and kwargs
        """
    
    @classmethod
    def explicit(cls, args=None, kwargs=None):
        """
        Create param by explicitly specifying args and kwargs lists.
        
        Parameters:
        - args: Tuple/list of positional arguments
        - kwargs: Dictionary of keyword arguments
        
        Returns:
        param instance
        """
    
    @classmethod
    def from_decorator(cls, args):
        """
        Create param from decorator arguments with automatic type handling.
        
        Parameters:
        - args: Single value, tuple, or param instance
        
        Returns:
        param instance
        """
```

### Class Parameterization

Function for creating multiple parameterized test classes with different attribute values.

```python { .api }
def parameterized_class(attrs, input_values=None, class_name_func=None, classname_func=None):
    """
    Parameterize test classes by setting attributes.
    
    Parameters:
    - attrs: String/list of attribute names or list of dicts with attribute values
    - input_values: List of tuples with values for each attrs (when attrs is string/list)
    - class_name_func: Function to generate class names (cls, idx, params_dict) -> str
    - classname_func: Deprecated, use class_name_func instead
    
    Returns:
    Decorator function for test classes
    """
```

### Utility Functions

Helper functions for parameter processing, test naming, and framework compatibility.

```python { .api }
def parameterized_argument_value_pairs(func, p):
    """
    Return tuples of parameterized arguments and their values.
    
    Parameters:
    - func: Test function to analyze
    - p: param instance with arguments
    
    Returns:
    List of (arg_name, value) tuples
    """

def short_repr(x, n=64):
    """
    Create shortened string representation guaranteed to be unicode.
    
    Parameters:
    - x: Object to represent
    - n: Maximum length (default: 64)
    
    Returns:
    Unicode string representation
    """

def default_doc_func(func, num, p):
    """
    Default function for generating test documentation.
    
    Parameters:
    - func: Test function
    - num: Parameter set number
    - p: param instance
    
    Returns:
    Documentation string or None
    """

def default_name_func(func, num, p):
    """
    Default function for generating test names.
    
    Parameters:
    - func: Test function
    - num: Parameter set number
    - p: param instance
    
    Returns:
    Test name string
    """

def set_test_runner(name):
    """
    Override automatic test runner detection.
    
    Parameters:
    - name: Test runner name ("unittest", "unittest2", "nose", "nose2", "pytest")
    
    Raises:
    TypeError: If runner name is invalid
    """

def detect_runner():
    """
    Detect current test runner by examining the call stack.
    
    Returns:
    String name of detected test runner or None
    """
```

## Usage Examples

### Advanced Parameter Specification

```python
from parameterized import parameterized, param

@parameterized([
    param(1, 2, expected=3),
    param("hello", " world", expected="hello world"),
    param([1, 2], [3, 4], expected=[1, 2, 3, 4]),
])
def test_operations(a, b, expected):
    if isinstance(a, str):
        result = a + b
    elif isinstance(a, list):
        result = a + b
    else:
        result = a + b
    
    assert result == expected
```

### Custom Naming and Documentation

```python
from parameterized import parameterized

def custom_name_func(func, num, p):
    return f"{func.__name__}_case_{num}_{p.args[0]}"

def custom_doc_func(func, num, p):
    return f"Test case {num}: {func.__name__} with input {p.args[0]}"

@parameterized([
    ("positive",), ("negative",), ("zero",)
], name_func=custom_name_func, doc_func=custom_doc_func)
def test_number_type(number_type):
    assert number_type in ["positive", "negative", "zero"]
```

### Mock Patch Compatibility

```python
from unittest.mock import patch
from parameterized import parameterized

class TestWithMocks(unittest.TestCase):
    @parameterized.expand([
        ("user1", "data1"),
        ("user2", "data2"),
    ])
    @patch('my_module.external_service')
    def test_service_calls(self, username, expected_data, mock_service):
        mock_service.get_data.return_value = expected_data
        result = my_module.process_user(username)
        mock_service.get_data.assert_called_once_with(username)
        self.assertEqual(result, expected_data)
```

### Complex Class Parameterization

```python
from parameterized import parameterized_class

@parameterized_class([
    {"database": "sqlite", "connection_string": ":memory:", "supports_transactions": True},
    {"database": "mysql", "connection_string": "mysql://localhost/test", "supports_transactions": True},
    {"database": "redis", "connection_string": "redis://localhost", "supports_transactions": False},
])
class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        self.db = create_database_connection(self.database, self.connection_string)
    
    def test_basic_operations(self):
        self.db.insert("key", "value")
        result = self.db.get("key")
        self.assertEqual(result, "value")
    
    def test_transaction_support(self):
        if self.supports_transactions:
            with self.db.transaction():
                self.db.insert("key", "value")
        else:
            with self.assertRaises(NotImplementedError):
                self.db.transaction()
```