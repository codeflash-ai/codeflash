import tempfile
from pathlib import Path

import pytest

from codeflash.code_utils.instrument_existing_tests import (
    add_async_decorator_to_function,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


def test_async_decorator_application_behavior_mode():
    async_function_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    expected_decorated_code = '''
import asyncio
from pathlib import Path

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_behavior_async


@codeflash_behavior_async(tests_project_root = Path(r"/tmp/tests"))
async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    func = FunctionToOptimize(
        function_name='async_function',
        file_path=Path('test_async.py'),
        parents=[],
        is_async=True
    )
    
    tests_root = Path("/tmp/tests")
    modified_code, decorator_added = add_async_decorator_to_function(
        async_function_code, func, tests_root, TestingMode.BEHAVIOR
    )
    
    assert decorator_added
    assert modified_code.strip() == expected_decorated_code.strip()


def test_async_decorator_application_performance_mode():
    async_function_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    expected_decorated_code = '''
import asyncio
from pathlib import Path

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_performance_async


@codeflash_performance_async(tests_project_root = Path(r"/tmp/tests"))
async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    func = FunctionToOptimize(
        function_name='async_function',
        file_path=Path('test_async.py'),
        parents=[],
        is_async=True
    )
    
    tests_root = Path("/tmp/tests")
    modified_code, decorator_added = add_async_decorator_to_function(
        async_function_code, func, tests_root, TestingMode.PERFORMANCE
    )
    
    assert decorator_added
    assert modified_code.strip() == expected_decorated_code.strip()


def test_async_class_method_decorator_application():
    async_class_code = '''
import asyncio

class Calculator:
    """Test class with async methods."""
    
    async def async_method(self, a: int, b: int) -> int:
        """Async method in class."""
        await asyncio.sleep(0.005)
        return a ** b
        
    def sync_method(self, a: int, b: int) -> int:
        """Sync method in class."""
        return a - b
'''
    
    expected_decorated_code = '''
import asyncio
from pathlib import Path

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_behavior_async


class Calculator:
    """Test class with async methods."""
    
    @codeflash_behavior_async(tests_project_root = Path(r"/tmp/tests"))
    async def async_method(self, a: int, b: int) -> int:
        """Async method in class."""
        await asyncio.sleep(0.005)
        return a ** b
        
    def sync_method(self, a: int, b: int) -> int:
        """Sync method in class."""
        return a - b
'''
    
    func = FunctionToOptimize(
        function_name='async_method',
        file_path=Path('test_async.py'),
        parents=[{'name': 'Calculator', 'type': 'ClassDef'}],
        is_async=True
    )
    
    tests_root = Path("/tmp/tests")
    modified_code, decorator_added = add_async_decorator_to_function(
        async_class_code, func, tests_root, TestingMode.BEHAVIOR
    )
    
    assert decorator_added
    assert modified_code.strip() == expected_decorated_code.strip()


def test_async_decorator_no_duplicate_application():
    already_decorated_code = '''
from codeflash.code_utils.codeflash_wrap_decorator import codeflash_behavior_async
import asyncio

@codeflash_behavior_async
async def async_function(x: int, y: int) -> int:
    """Already decorated async function."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    expected_reformatted_code = '''
import asyncio

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_behavior_async


@codeflash_behavior_async
async def async_function(x: int, y: int) -> int:
    """Already decorated async function."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    func = FunctionToOptimize(
        function_name='async_function',
        file_path=Path('test_async.py'),
        parents=[],
        is_async=True
    )
    
    tests_root = Path("/tmp/tests")
    modified_code, decorator_added = add_async_decorator_to_function(
        already_decorated_code, func, tests_root, TestingMode.BEHAVIOR
    )
    
    assert not decorator_added
    assert modified_code.strip() == expected_reformatted_code.strip()


def test_inject_profiling_async_function_behavior_mode(temp_dir):
    source_module_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    source_file = temp_dir / "my_module.py"
    source_file.write_text(source_module_code)
    
    async_test_code = '''
import asyncio
import pytest
from my_module import async_function

@pytest.mark.asyncio
async def test_async_function():
    """Test async function behavior."""
    result = await async_function(5, 3)
    assert result == 15
    
    result2 = await async_function(2, 4)
    assert result2 == 8
'''
    
    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_test_code)
    
    func = FunctionToOptimize(
        function_name="async_function",
        parents=[],
        file_path=Path("my_module.py"),
        is_async=True
    )
    
    # First instrument the source module
    from codeflash.code_utils.instrument_existing_tests import instrument_source_module_with_async_decorators
    source_success, instrumented_source = instrument_source_module_with_async_decorators(
        source_file, func, temp_dir, TestingMode.BEHAVIOR
    )
    
    assert source_success
    assert instrumented_source is not None
    assert '@codeflash_behavior_async' in instrumented_source
    assert 'from codeflash.code_utils.codeflash_wrap_decorator import' in instrumented_source
    assert 'codeflash_behavior_async' in instrumented_source
    
    source_file.write_text(instrumented_source)
    
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file,
        [CodePosition(8, 18), CodePosition(11, 19)],
        func,
        temp_dir,
        "pytest",
        mode=TestingMode.BEHAVIOR
    )
    
    assert not success
    assert instrumented_test_code is None


def test_inject_profiling_async_function_performance_mode(temp_dir):
    source_module_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    source_file = temp_dir / "my_module.py"
    source_file.write_text(source_module_code)
    
    # Create the test file
    async_test_code = '''
import asyncio
import pytest
from my_module import async_function

@pytest.mark.asyncio
async def test_async_function():
    """Test async function performance."""
    result = await async_function(5, 3)
    assert result == 15
'''
    
    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_test_code)
    
    func = FunctionToOptimize(
        function_name="async_function",
        parents=[],
        file_path=Path("my_module.py"),
        is_async=True
    )
    
    # First instrument the source module
    from codeflash.code_utils.instrument_existing_tests import instrument_source_module_with_async_decorators
    source_success, instrumented_source = instrument_source_module_with_async_decorators(
        source_file, func, temp_dir, TestingMode.PERFORMANCE
    )
    
    assert source_success
    assert instrumented_source is not None
    assert '@codeflash_performance_async' in instrumented_source
    # Check for the import with line continuation formatting
    assert 'from codeflash.code_utils.codeflash_wrap_decorator import' in instrumented_source
    assert 'codeflash_performance_async' in instrumented_source
    
    # Write the instrumented source back
    source_file.write_text(instrumented_source)
    
    # Now test the full pipeline with source module path
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file,
        [CodePosition(8, 18)],
        func,
        temp_dir,
        "pytest",
        mode=TestingMode.PERFORMANCE
    )
    
    assert not success
    assert instrumented_test_code is None


def test_mixed_sync_async_instrumentation(temp_dir):
    source_module_code = '''
import asyncio

def sync_function(x: int, y: int) -> int:
    """Regular sync function."""
    return x * y

async def async_function(x: int, y: int) -> int:
    """Simple async function."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    source_file = temp_dir / "my_module.py"
    source_file.write_text(source_module_code)
    
    mixed_test_code = '''
import asyncio
import pytest
from my_module import sync_function, async_function

@pytest.mark.asyncio
async def test_mixed_functions():
    """Test both sync and async functions."""
    sync_result = sync_function(10, 5)
    assert sync_result == 50
    
    async_result = await async_function(3, 4)
    assert async_result == 12
'''
    
    test_file = temp_dir / "test_mixed.py"
    test_file.write_text(mixed_test_code)
    
    async_func = FunctionToOptimize(
        function_name="async_function",
        parents=[],
        file_path=Path("my_module.py"),
        is_async=True
    )
    
    from codeflash.code_utils.instrument_existing_tests import instrument_source_module_with_async_decorators
    source_success, instrumented_source = instrument_source_module_with_async_decorators(
        source_file, async_func, temp_dir, TestingMode.BEHAVIOR
    )
    
    assert source_success
    assert instrumented_source is not None
    assert '@codeflash_behavior_async' in instrumented_source
    assert 'from codeflash.code_utils.codeflash_wrap_decorator import' in instrumented_source
    assert 'codeflash_behavior_async' in instrumented_source
    # Sync function should remain unchanged
    assert 'def sync_function(x: int, y: int) -> int:' in instrumented_source
    
    # Write instrumented source back
    source_file.write_text(instrumented_source)
    
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file,
        [CodePosition(8, 18), CodePosition(11, 19)],
        async_func,
        temp_dir,
        "pytest",
        mode=TestingMode.BEHAVIOR
    )
    
    # Async functions should not be instrumented at the test level
    assert not success
    assert instrumented_test_code is None



def test_async_function_qualified_name_handling():
    nested_async_code = '''
import asyncio

class OuterClass:
    """Outer class container."""
    
    class InnerClass:
        """Inner class with async method."""
        
        async def nested_async_method(self, x: int) -> int:
            """Nested async method."""
            await asyncio.sleep(0.001)
            return x * 2
'''
    
    func = FunctionToOptimize(
        function_name='nested_async_method',
        file_path=Path('test_nested.py'),
        parents=[
            {'name': 'OuterClass', 'type': 'ClassDef'},
            {'name': 'InnerClass', 'type': 'ClassDef'}
        ],
        is_async=True
    )
    
    modified_code, decorator_added = add_async_decorator_to_function(
        nested_async_code, func, TestingMode.BEHAVIOR
    )
    
    assert decorator_added
    assert '@codeflash_behavior_async' in modified_code
    assert 'async def nested_async_method(self, x: int) -> int:' in modified_code


def test_async_decorator_with_existing_decorators():
    """Test async decorator application when function already has other decorators."""
    decorated_async_code = '''
import asyncio
from functools import wraps

def my_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper

@my_decorator
async def async_function(x: int, y: int) -> int:
    """Async function with existing decorator."""
    await asyncio.sleep(0.01)
    return x * y
'''
    
    func = FunctionToOptimize(
        function_name='async_function',
        file_path=Path('test_async.py'),
        parents=[],
        is_async=True
    )
    
    modified_code, decorator_added = add_async_decorator_to_function(
        decorated_async_code, func, TestingMode.BEHAVIOR
    )
    
    assert decorator_added
    # Should add codeflash decorator above existing decorators
    assert '@codeflash_behavior_async' in modified_code
    assert '@my_decorator' in modified_code
    # Codeflash decorator should come first
    codeflash_pos = modified_code.find('@codeflash_behavior_async')
    my_decorator_pos = modified_code.find('@my_decorator')
    assert codeflash_pos < my_decorator_pos


def test_sync_function_not_affected_by_async_logic():
    """Test that sync functions are not affected by async decorator logic."""
    sync_function_code = '''
def sync_function(x: int, y: int) -> int:
    """Regular sync function."""
    return x + y
'''
    
    sync_func = FunctionToOptimize(
        function_name='sync_function',
        file_path=Path('test_sync.py'),
        parents=[],
        is_async=False  # Explicitly sync
    )
    
    # This should not apply async decorators
    modified_code, decorator_added = add_async_decorator_to_function(
        sync_function_code, sync_func, TestingMode.BEHAVIOR
    )
    
    assert not decorator_added
    assert modified_code == sync_function_code


def test_qualified_name_with_nested_parents():
    from codeflash.models.models import FunctionParent
    
    func_no_parents = FunctionToOptimize(
        function_name='simple_function',
        file_path=Path('test.py'),
        parents=[],
        is_async=False
    )
    assert func_no_parents.qualified_name == 'simple_function'
    
    # Test function with one parent
    func_one_parent = FunctionToOptimize(
        function_name='method',
        file_path=Path('test.py'),
        parents=[FunctionParent('MyClass', 'ClassDef')],
        is_async=False
    )
    assert func_one_parent.qualified_name == 'MyClass.method'
    
    func_nested_parents = FunctionToOptimize(
        function_name='nested_method',
        file_path=Path('test.py'),
        parents=[
            FunctionParent('OuterClass', 'ClassDef'),
            FunctionParent('MiddleClass', 'ClassDef'),
            FunctionParent('InnerClass', 'ClassDef')
        ],
        is_async=True
    )
    assert func_nested_parents.qualified_name == 'OuterClass.MiddleClass.InnerClass.nested_method'
    
    func_mixed_parents = FunctionToOptimize(
        function_name='inner_function',
        file_path=Path('test.py'),
        parents=[
            FunctionParent('MyClass', 'ClassDef'),
            FunctionParent('outer_function', 'FunctionDef')
        ],
        is_async=False
    )
    assert func_mixed_parents.qualified_name == 'MyClass.outer_function.inner_function'
