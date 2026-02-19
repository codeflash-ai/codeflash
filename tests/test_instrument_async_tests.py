import os
import sys
import tempfile
from pathlib import Path

import pytest

from codeflash.code_utils.instrument_existing_tests import (
    ASYNC_HELPER_FILENAME,
    add_async_decorator_to_function,
    get_decorator_name_for_mode,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


# @pytest.fixture
# def unique_test_iteration():
#     """Provide a unique test iteration ID and clean up database after test."""
#     # Generate unique iteration ID
#     iteration_id = str(uuid.uuid4())[:8]

#     # Store original environment variable
#     original_iteration = os.environ.get("CODEFLASH_TEST_ITERATION")

#     # Set unique iteration for this test
#     os.environ["CODEFLASH_TEST_ITERATION"] = iteration_id

#     try:
#         yield iteration_id
#     finally:
#         # Cleanup: restore original environment and delete database file
#         if original_iteration is not None:
#             os.environ["CODEFLASH_TEST_ITERATION"] = original_iteration
#         elif "CODEFLASH_TEST_ITERATION" in os.environ:
#             del os.environ["CODEFLASH_TEST_ITERATION"]

#         # Clean up database file
#         try:
#             from codeflash.code_utils.codeflash_wrap_decorator import get_run_tmp_file

#             db_path = get_run_tmp_file(Path(f"test_return_values_{iteration_id}.sqlite"))
#             if db_path.exists():
#                 db_path.unlink()
#         except Exception:
#             pass  # Ignore cleanup errors


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_application_behavior_mode(temp_dir):
    async_function_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''

    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_function_code)

    func = FunctionToOptimize(function_name="async_function", file_path=test_file, parents=[], is_async=True)

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.BEHAVIOR)

    assert decorator_added
    modified_code = test_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.BEHAVIOR)
    code_with_decorator = async_function_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert modified_code.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_application_performance_mode(temp_dir):
    async_function_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''

    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_function_code)

    func = FunctionToOptimize(function_name="async_function", file_path=test_file, parents=[], is_async=True)

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.PERFORMANCE)

    assert decorator_added
    modified_code = test_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.PERFORMANCE)
    code_with_decorator = async_function_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert modified_code.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_application_concurrency_mode(temp_dir):
    """Test that CONCURRENCY mode applies the codeflash_concurrency_async decorator."""
    async_function_code = '''
import asyncio

async def async_function(x: int, y: int) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.01)
    return x * y
'''

    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_function_code)

    func = FunctionToOptimize(function_name="async_function", file_path=test_file, parents=[], is_async=True)

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.CONCURRENCY)

    assert decorator_added
    modified_code = test_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.CONCURRENCY)
    code_with_decorator = async_function_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert modified_code.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_class_method_decorator_application(temp_dir):
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

    test_file = temp_dir / "test_async.py"
    test_file.write_text(async_class_code)

    func = FunctionToOptimize(
        function_name="async_method",
        file_path=test_file,
        parents=[{"name": "Calculator", "type": "ClassDef"}],
        is_async=True,
    )

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.BEHAVIOR)

    assert decorator_added
    modified_code = test_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.BEHAVIOR)
    code_with_decorator = async_class_code.replace(
        "    async def async_method", f"    @{decorator_name}\n    async def async_method"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert modified_code.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_no_duplicate_application(temp_dir):
    # Case 1: Old-style import already present — injector should detect and skip
    already_decorated_code = '''
from codeflash.code_utils.codeflash_wrap_decorator import codeflash_behavior_async
import asyncio

@codeflash_behavior_async
async def async_function(x: int, y: int) -> int:
    """Already decorated async function."""
    await asyncio.sleep(0.01)
    return x * y
'''

    test_file = temp_dir / "test_async.py"
    test_file.write_text(already_decorated_code)

    func = FunctionToOptimize(function_name="async_function", file_path=test_file, parents=[], is_async=True)

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.BEHAVIOR)

    # Should not add duplicate decorator
    assert not decorator_added

    # Case 2: Inline definition already present — injector should detect and skip
    already_inline_code = '''
import asyncio

def codeflash_behavior_async(func):
    return func

@codeflash_behavior_async
async def async_function(x: int, y: int) -> int:
    """Already decorated async function."""
    await asyncio.sleep(0.01)
    return x * y
'''

    test_file2 = temp_dir / "test_async2.py"
    test_file2.write_text(already_inline_code)

    func2 = FunctionToOptimize(function_name="async_function", file_path=test_file2, parents=[], is_async=True)

    decorator_added2 = add_async_decorator_to_function(test_file2, func2, TestingMode.BEHAVIOR)

    # Should not add duplicate decorator
    assert not decorator_added2


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
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

    func = FunctionToOptimize(function_name="async_function", parents=[], file_path=Path("my_module.py"), is_async=True)

    # First instrument the source module
    from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

    source_success = add_async_decorator_to_function(source_file, func, TestingMode.BEHAVIOR)

    assert source_success is True

    # Verify the file was modified with exact expected output
    instrumented_source = source_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.BEHAVIOR)
    code_with_decorator = source_module_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert instrumented_source.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()

    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file, [CodePosition(8, 18), CodePosition(11, 19)], func, temp_dir, mode=TestingMode.BEHAVIOR
    )

    # For async functions, once source is decorated, test injection should fail
    # This is expected behavior - async instrumentation happens at the decorator level
    assert success is False
    assert instrumented_test_code is None


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
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

    func = FunctionToOptimize(function_name="async_function", parents=[], file_path=Path("my_module.py"), is_async=True)

    # First instrument the source module
    from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

    source_success = add_async_decorator_to_function(source_file, func, TestingMode.PERFORMANCE)

    assert source_success is True

    # Verify the file was modified with exact expected output
    instrumented_source = source_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.PERFORMANCE)
    code_with_decorator = source_module_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert instrumented_source.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()

    # Now test the full pipeline with source module path
    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file, [CodePosition(8, 18)], func, temp_dir, mode=TestingMode.PERFORMANCE
    )

    # For async functions, once source is decorated, test injection should fail
    # This is expected behavior - async instrumentation happens at the decorator level
    assert success is False
    assert instrumented_test_code is None


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
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
        function_name="async_function", parents=[], file_path=Path("my_module.py"), is_async=True
    )

    from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

    source_success = add_async_decorator_to_function(source_file, async_func, TestingMode.BEHAVIOR)

    assert source_success

    # Verify the file was modified
    instrumented_source = source_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.BEHAVIOR)
    code_with_decorator = source_module_code.replace(
        "async def async_function", f"@{decorator_name}\nasync def async_function"
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert instrumented_source.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()

    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file, [CodePosition(8, 18), CodePosition(11, 19)], async_func, temp_dir, mode=TestingMode.BEHAVIOR
    )

    # Async functions should not be instrumented at the test level
    assert not success
    assert instrumented_test_code is None


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_function_qualified_name_handling(temp_dir):
    nested_async_code = '''
import asyncio

class OuterClass:    
    class InnerClass:        
        async def nested_async_method(self, x: int) -> int:
            """Nested async method."""
            await asyncio.sleep(0.001)
            return x * 2
'''

    test_file = temp_dir / "test_nested.py"
    test_file.write_text(nested_async_code)

    func = FunctionToOptimize(
        function_name="nested_async_method",
        file_path=test_file,
        parents=[{"name": "OuterClass", "type": "ClassDef"}, {"name": "InnerClass", "type": "ClassDef"}],
        is_async=True,
    )

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.BEHAVIOR)

    assert decorator_added
    modified_code = test_file.read_text()
    from codeflash.code_utils.formatter import sort_imports

    decorator_name = get_decorator_name_for_mode(TestingMode.BEHAVIOR)
    code_with_decorator = nested_async_code.replace(
        "        async def nested_async_method",
        f"        @{decorator_name}\n        async def nested_async_method",
    )
    code_with_import = f"from codeflash_async_wrapper import {decorator_name}\n{code_with_decorator}"
    expected = sort_imports(code=code_with_import, float_to_top=True)
    assert modified_code.strip() == expected.strip()
    assert (temp_dir / ASYNC_HELPER_FILENAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_with_existing_decorators(temp_dir):
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

    test_file = temp_dir / "test_async.py"
    test_file.write_text(decorated_async_code)

    func = FunctionToOptimize(function_name="async_function", file_path=test_file, parents=[], is_async=True)

    decorator_added = add_async_decorator_to_function(test_file, func, TestingMode.BEHAVIOR)

    assert decorator_added
    modified_code = test_file.read_text()
    # Should add codeflash decorator above existing decorators
    assert "@codeflash_behavior_async" in modified_code
    assert "@my_decorator" in modified_code
    # Codeflash decorator should come first
    codeflash_pos = modified_code.find("@codeflash_behavior_async")
    my_decorator_pos = modified_code.find("@my_decorator")
    assert codeflash_pos < my_decorator_pos


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_sync_function_not_affected_by_async_logic(temp_dir):
    sync_function_code = '''
def sync_function(x: int, y: int) -> int:
    """Regular sync function."""
    return x + y
'''

    test_file = temp_dir / "test_sync.py"
    test_file.write_text(sync_function_code)

    sync_func = FunctionToOptimize(function_name="sync_function", file_path=test_file, parents=[], is_async=False)

    decorator_added = add_async_decorator_to_function(test_file, sync_func, TestingMode.BEHAVIOR)

    assert not decorator_added
    # File should not be modified for sync functions
    modified_code = test_file.read_text()
    assert modified_code == sync_function_code


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
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

    test_code_multiple_calls = """
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
"""

    test_file = temp_dir / "test_async_sorter.py"
    test_file.write_text(test_code_multiple_calls)

    func = FunctionToOptimize(
        function_name="async_sorter", parents=[], file_path=Path("async_sorter.py"), is_async=True
    )

    # First instrument the source module with async decorators
    from codeflash.code_utils.instrument_existing_tests import add_async_decorator_to_function

    source_success = add_async_decorator_to_function(source_file, func, TestingMode.BEHAVIOR)

    assert source_success

    # Verify the file was modified
    instrumented_source = source_file.read_text()
    assert "@codeflash_behavior_async" in instrumented_source

    import ast

    tree = ast.parse(test_code_multiple_calls)
    call_positions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            if (hasattr(node.value.func, "id") and node.value.func.id == "async_sorter") or (
                hasattr(node.value.func, "attr") and node.value.func.attr == "async_sorter"
            ):
                call_positions.append(CodePosition(node.lineno, node.col_offset))

    assert len(call_positions) == 4

    success, instrumented_test_code = inject_profiling_into_existing_test(
        test_file, call_positions, func, temp_dir, mode=TestingMode.BEHAVIOR
    )

    assert success
    assert instrumented_test_code is not None

    assert "os.environ['CODEFLASH_CURRENT_LINE_ID'] = '0'" in instrumented_test_code

    # Count occurrences of each line_id to verify numbering
    line_id_0_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '0'")
    line_id_1_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '1'")
    line_id_2_count = instrumented_test_code.count("os.environ['CODEFLASH_CURRENT_LINE_ID'] = '2'")

    assert line_id_0_count == 2, f"Expected 2 occurrences of line_id '0', got {line_id_0_count}"
    assert line_id_1_count == 1, f"Expected 1 occurrence of line_id '1', got {line_id_1_count}"
    assert line_id_2_count == 1, f"Expected 1 occurrence of line_id '2', got {line_id_2_count}"


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_behavior_decorator_return_values_and_test_ids():
    """Test that async behavior decorator correctly captures return values, test IDs, and stores data in database."""
    import asyncio
    import sqlite3
    from pathlib import Path

    import dill as pickle

    from codeflash.code_utils.codeflash_wrap_decorator import codeflash_behavior_async

    @codeflash_behavior_async
    async def test_async_multiply(x: int, y: int) -> int:
        """Simple async function for testing."""
        await asyncio.sleep(0.001)  # Small delay to simulate async work
        return x * y

    test_env = {
        "CODEFLASH_TEST_MODULE": "test_module",
        "CODEFLASH_TEST_CLASS": None,
        "CODEFLASH_TEST_FUNCTION": "test_async_multiply_function",
        "CODEFLASH_CURRENT_LINE_ID": "0",
        "CODEFLASH_LOOP_INDEX": "1",
        "CODEFLASH_TEST_ITERATION": "2",
    }

    original_env = {k: os.environ.get(k) for k in test_env}
    for k, v in test_env.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]

    try:
        result = asyncio.run(test_async_multiply(6, 7))

        assert result == 42, f"Expected return value 42, got {result}"

        from codeflash.code_utils.codeflash_wrap_decorator import get_run_tmp_file

        db_path = get_run_tmp_file(Path("test_return_values_2.sqlite"))

        # Verify database exists and has data
        assert db_path.exists(), f"Database file not created at {db_path}"

        # Read and verify database contents
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cur.execute("SELECT * FROM test_results")
        rows = cur.fetchall()

        assert len(rows) == 1, f"Expected 1 database row, got {len(rows)}"

        row = rows[0]
        (
            test_module,
            test_class,
            test_function,
            function_name,
            loop_index,
            iteration_id,
            runtime,
            return_value_blob,
            verification_type,
        ) = row

        assert test_module == "test_module", f"Expected test_module 'test_module', got '{test_module}'"
        assert test_class is None, f"Expected test_class None, got '{test_class}'"
        assert test_function == "test_async_multiply_function", (
            f"Expected test_function 'test_async_multiply_function', got '{test_function}'"
        )
        assert function_name == "test_async_multiply", (
            f"Expected function_name 'test_async_multiply', got '{function_name}'"
        )
        assert loop_index == 1, f"Expected loop_index 1, got {loop_index}"
        assert iteration_id == "0_0", f"Expected iteration_id '0_0', got '{iteration_id}'"
        assert verification_type == "function_call", (
            f"Expected verification_type 'function_call', got '{verification_type}'"
        )
        unpickled_data = pickle.loads(return_value_blob)
        args, kwargs, actual_return_value = unpickled_data

        assert args == (6, 7), f"Expected args (6, 7), got {args}"
        assert kwargs == {}, f"Expected empty kwargs, got {kwargs}"

        assert actual_return_value == 42, f"Expected stored return value 42, got {actual_return_value}"

        con.close()

    finally:
        for k, v in original_env.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]


@pytest.mark.skipif(sys.platform == "win32", reason="pending support for asyncio on windows")
def test_async_decorator_comprehensive_return_values_and_test_ids():
    import asyncio
    import sqlite3
    from pathlib import Path

    import dill as pickle

    from codeflash.code_utils.codeflash_wrap_decorator import codeflash_behavior_async, get_run_tmp_file

    @codeflash_behavior_async
    async def async_multiply_add(x: int, y: int, z: int = 1) -> int:
        """Async function that multiplies x*y then adds z."""
        await asyncio.sleep(0.001)
        result = (x * y) + z
        return result

    test_env = {
        "CODEFLASH_TEST_MODULE": "test_comprehensive_module",
        "CODEFLASH_TEST_CLASS": "AsyncTestClass",
        "CODEFLASH_TEST_FUNCTION": "test_comprehensive_async_function",
        "CODEFLASH_CURRENT_LINE_ID": "3",
        "CODEFLASH_LOOP_INDEX": "2",
        "CODEFLASH_TEST_ITERATION": "3",
    }

    original_env = {k: os.environ.get(k) for k in test_env}
    for k, v in test_env.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]

    try:
        test_cases = [
            {"args": (5, 3), "kwargs": {}, "expected": 16},  # (5 * 3) + 1 = 16
            {"args": (2, 4), "kwargs": {"z": 10}, "expected": 18},  # (2 * 4) + 10 = 18
            {"args": (7, 6), "kwargs": {}, "expected": 43},  # (7 * 6) + 1 = 43
        ]

        results = []
        for test_case in test_cases:
            result = asyncio.run(async_multiply_add(*test_case["args"], **test_case["kwargs"]))
            results.append(result)

            # Verify each return value is exactly correct
            assert result == test_case["expected"], (
                f"Expected {test_case['expected']}, got {result} for args {test_case['args']}, kwargs {test_case['kwargs']}"
            )

        db_path = get_run_tmp_file(Path("test_return_values_3.sqlite"))
        assert db_path.exists(), f"Database not created at {db_path}"

        con = sqlite3.connect(db_path)
        cur = con.cursor()

        cur.execute(
            "SELECT test_module_path, test_class_name, test_function_name, function_getting_tested, loop_index, iteration_id, runtime, return_value, verification_type FROM test_results ORDER BY rowid"
        )
        rows = cur.fetchall()

        assert len(rows) == 3, f"Expected 3 database rows, got {len(rows)}"

        for i, (
            test_module,
            test_class,
            test_function,
            function_name,
            loop_index,
            iteration_id,
            runtime,
            return_value_blob,
            verification_type,
        ) in enumerate(rows):
            assert test_module == "test_comprehensive_module", (
                f"Row {i}: Expected test_module 'test_comprehensive_module', got '{test_module}'"
            )
            assert test_class == "AsyncTestClass", f"Row {i}: Expected test_class 'AsyncTestClass', got '{test_class}'"
            assert test_function == "test_comprehensive_async_function", (
                f"Row {i}: Expected test_function 'test_comprehensive_async_function', got '{test_function}'"
            )
            assert function_name == "async_multiply_add", (
                f"Row {i}: Expected function_name 'async_multiply_add', got '{function_name}'"
            )
            assert loop_index == 2, f"Row {i}: Expected loop_index 2, got {loop_index}"
            assert verification_type == "function_call", (
                f"Row {i}: Expected verification_type 'function_call', got '{verification_type}'"
            )

            expected_iteration_id = f"3_{i}"
            assert iteration_id == expected_iteration_id, (
                f"Row {i}: Expected iteration_id '{expected_iteration_id}', got '{iteration_id}'"
            )

            args, kwargs, actual_return_value = pickle.loads(return_value_blob)
            expected_args = test_cases[i]["args"]
            expected_kwargs = test_cases[i]["kwargs"]
            expected_return = test_cases[i]["expected"]

            assert args == expected_args, f"Row {i}: Expected args {expected_args}, got {args}"
            assert kwargs == expected_kwargs, f"Row {i}: Expected kwargs {expected_kwargs}, got {kwargs}"
            assert actual_return_value == expected_return, (
                f"Row {i}: Expected return value {expected_return}, got {actual_return_value}"
            )

        con.close()

    finally:
        for k, v in original_env.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]
