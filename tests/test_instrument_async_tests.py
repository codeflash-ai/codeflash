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

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_behavior_async


@codeflash_behavior_async
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
    
    modified_code, decorator_added = add_async_decorator_to_function(
        async_function_code, func, TestingMode.BEHAVIOR
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

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_performance_async


@codeflash_performance_async
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
    
    modified_code, decorator_added = add_async_decorator_to_function(
        async_function_code, func, TestingMode.PERFORMANCE
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

from codeflash.code_utils.codeflash_wrap_decorator import \\
    codeflash_behavior_async


class Calculator:
    """Test class with async methods."""
    
    @codeflash_behavior_async
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
    
    modified_code, decorator_added = add_async_decorator_to_function(
        async_class_code, func, TestingMode.BEHAVIOR
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
    
    modified_code, decorator_added = add_async_decorator_to_function(
        already_decorated_code, func, TestingMode.BEHAVIOR
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
        source_file, func, TestingMode.BEHAVIOR
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
        source_file, func, TestingMode.PERFORMANCE
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
        source_file, async_func, TestingMode.BEHAVIOR
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


def test_inject_profiling_async_multiple_calls_same_test(temp_dir):
    """Test that multiple async function calls within the same test function get correctly numbered 0, 1, 2, etc."""
    source_module_code = '''
import asyncio

async def async_sorter(items):
    """Simple async sorter for testing."""
    await asyncio.sleep(0.001)
    return sorted(items)
'''
    
    source_file = temp_dir / "async_sorter.py"
    source_file.write_text(source_module_code)
    
    test_code_multiple_calls = '''
import asyncio
import pytest
from async_sorter import async_sorter

@pytest.mark.asyncio
async def test_single_call():
    result = await async_sorter([42])
    assert result == [42]

@pytest.mark.asyncio
async def test_multiple_calls():
    result1 = await async_sorter([3, 1, 2])
    result2 = await async_sorter([5, 4])  
    result3 = await async_sorter([9, 8, 7, 6])
    assert result1 == [1, 2, 3]
    assert result2 == [4, 5]
    assert result3 == [6, 7, 8, 9]
'''
    
    test_file = temp_dir / "test_async_sorter.py"
    test_file.write_text(test_code_multiple_calls)
    
    func = FunctionToOptimize(
        function_name="async_sorter",
        parents=[],
        file_path=Path("async_sorter.py"),
        is_async=True
    )
    
    # First instrument the source module with async decorators
    from codeflash.code_utils.instrument_existing_tests import instrument_source_module_with_async_decorators
    source_success, instrumented_source = instrument_source_module_with_async_decorators(
        source_file, func, TestingMode.BEHAVIOR
    )
    
    assert source_success
    assert instrumented_source is not None
    assert '@codeflash_behavior_async' in instrumented_source
    
    # Write the instrumented source back
    source_file.write_text(instrumented_source)
    
    # Now test injection with multiple call positions
    # Parse the test file to get exact positions for async calls
    import ast
    tree = ast.parse(test_code_multiple_calls)
    call_positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            if hasattr(node.value.func, 'id') and node.value.func.id == 'async_sorter':
                call_positions.append(CodePosition(node.lineno, node.col_offset))
            elif hasattr(node.value.func, 'attr') and node.value.func.attr == 'async_sorter':
                call_positions.append(CodePosition(node.lineno, node.col_offset))
    
    # Should find 4 calls total: 1 in test_single_call + 3 in test_multiple_calls
    assert len(call_positions) == 4
    
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file,
        call_positions,
        func,
        temp_dir,
        "pytest",
        mode=TestingMode.BEHAVIOR
    )
    
    assert success
    assert instrumented_test_code is not None
    
    # Verify the instrumentation adds correct line_id assignments
    # Each test function should start from 0
    assert "os.environ['CODEFLASH_CURRENT_LINE_ID'] = '0'" in instrumented_test_code
    
    # Count occurrences of each line_id to verify numbering
    line_id_0_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '0'")
    line_id_1_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '1'")
    line_id_2_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '2'")
    
    # Should have:
    # - 2 occurrences of '0' (first call in each test function)
    # - 1 occurrence of '1' (second call in test_multiple_calls)
    # - 1 occurrence of '2' (third call in test_multiple_calls)
    assert line_id_0_count == 2, f"Expected 2 occurrences of line_id '0', got {line_id_0_count}"
    assert line_id_1_count == 1, f"Expected 1 occurrence of line_id '1', got {line_id_1_count}"
    assert line_id_2_count == 1, f"Expected 1 occurrence of line_id '2', got {line_id_2_count}"
    
    # Verify no higher numbers
    line_id_3_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '3'")
    assert line_id_3_count == 0, f"Unexpected occurrence of line_id '3'"
    
    # Check that imports are added
    assert 'import os' in instrumented_test_code


def test_sync_functions_do_not_get_async_instrumentation(temp_dir):
    """Test that sync functions do NOT get async instrumentation (os.environ assignments)."""
    # Create a sync function module
    sync_module_code = '''
def sync_sorter(items):
    """Simple sync sorter for testing."""
    return sorted(items)
'''
    
    source_file = temp_dir / "sync_sorter.py"
    source_file.write_text(sync_module_code)
    
    # Create test code with sync function calls
    sync_test_code = '''
import pytest
from sync_sorter import sync_sorter

def test_single_call():
    result = sync_sorter([42])
    assert result == [42]

def test_multiple_calls():
    result1 = sync_sorter([3, 1, 2])
    result2 = sync_sorter([5, 4])  
    result3 = sync_sorter([9, 8, 7, 6])
    assert result1 == [1, 2, 3]
    assert result2 == [4, 5]
    assert result3 == [6, 7, 8, 9]
'''
    
    test_file = temp_dir / "test_sync_sorter.py"
    test_file.write_text(sync_test_code)
    
    sync_func = FunctionToOptimize(
        function_name="sync_sorter",
        parents=[],
        file_path=Path("sync_sorter.py"),
        is_async=False  # SYNC function
    )
    
    # Parse the test file to get exact positions for sync calls
    import ast
    tree = ast.parse(sync_test_code)
    call_positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id') and node.func.id == 'sync_sorter':
                call_positions.append(CodePosition(node.lineno, node.col_offset))
            elif hasattr(node.func, 'attr') and node.func.attr == 'sync_sorter':
                call_positions.append(CodePosition(node.lineno, node.col_offset))
    
    # Should find 4 calls total: 1 in test_single_call + 3 in test_multiple_calls
    assert len(call_positions) == 4
    
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file,
        call_positions,
        sync_func,
        temp_dir,
        "pytest",
        mode=TestingMode.BEHAVIOR
    )
    
    assert success
    assert instrumented_test_code is not None
    
    # Verify the sync function does NOT get async instrumentation
    assert "os.environ['CODEFLASH_CURRENT_LINE_ID']" not in instrumented_test_code
    
    # But should get proper sync instrumentation
    assert 'codeflash_wrap' in instrumented_test_code
    assert 'codeflash_loop_index' in instrumented_test_code
    assert 'sqlite3' in instrumented_test_code  # sync behavior mode includes sqlite
    
    # Verify the line_id values are correct for sync functions (statement-based)
    # Sync functions use statement index, not per-test-function counter
    assert "'0'" in instrumented_test_code  # first call in test_single_call
    assert "'0'" in instrumented_test_code  # first call in test_multiple_calls (second occurrence)
    assert "'1'" in instrumented_test_code  # second call in test_multiple_calls
    assert "'2'" in instrumented_test_code  # third call in test_multiple_calls
