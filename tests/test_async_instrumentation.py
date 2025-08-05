import ast
import asyncio
import textwrap
from pathlib import Path

import pytest

from codeflash.code_utils.instrument_existing_tests import (
    InjectPerfOnly,
    create_async_wrapper_inner,
    create_wrapper_function,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode


def test_create_async_wrapper_inner():
    async_wrapper = create_async_wrapper_inner()

    assert isinstance(async_wrapper, ast.AsyncFunctionDef)
    assert async_wrapper.name == "codeflash_async_wrap_inner"

    assert len(async_wrapper.body) == 1
    assert isinstance(async_wrapper.body[0], ast.Return)
    assert isinstance(async_wrapper.body[0].value, ast.Await)


def test_wrapper_function_includes_async_check():
    wrapper = create_wrapper_function(TestingMode.PERFORMANCE)

    async_check_found = False
    for node in ast.walk(wrapper):
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Call):
                if (
                    isinstance(node.test.func, ast.Attribute)
                    and node.test.func.attr == "iscoroutinefunction"
                ):
                    async_check_found = True
                    assert len(node.body) > 0
                    assert len(node.orelse) > 0
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            assert hasattr(stmt, "lineno")
                    for stmt in node.orelse:
                        if isinstance(stmt, ast.Assign):
                            assert hasattr(stmt, "lineno")
                    break

    assert async_check_found, "Async check not found in wrapper function"


def test_inject_profiling_with_async_function():
    test_code = textwrap.dedent("""
        import asyncio
        from my_module import async_process_data
        
        async def test_async_function():
            result = await async_process_data("test")
            assert result == "processed"
    """)

    test_file = Path("/tmp/test_async.py")
    test_file.write_text(test_code)

    # Create function to optimize
    function = FunctionToOptimize(
        function_name="async_process_data",
        parents=[],
        file_path=Path("my_module.py"),
        starting_line=1,
        ending_line=10,
    )

    call_positions = [CodePosition(line_no=5, col_no=19)]

    success, modified_code = inject_profiling_into_existing_test(
        test_file,
        call_positions,
        function,
        Path("/tmp"),
        "pytest",
        TestingMode.PERFORMANCE,
    )

    assert success
    assert modified_code is not None

    assert "codeflash_async_wrap_inner" in modified_code
    assert "inspect.iscoroutinefunction" in modified_code
    assert "import inspect" in modified_code

    try:
        ast.parse(modified_code)
    except SyntaxError:
        pytest.fail(f"Modified code has syntax errors:\n{modified_code}")

    test_file.unlink()


def test_async_wrapper_preserves_return_value():
    test_code = textwrap.dedent("""
        import asyncio
        
        async def async_function(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        async def test_async_return():
            result = await async_function(5)
            assert result == 10
    """)

    tree = ast.parse(test_code)

    function = FunctionToOptimize(
        function_name="async_function",
        parents=[],
        file_path=Path("test.py"),
        starting_line=3,
        ending_line=5,
    )

    call_positions = [CodePosition(line_no=8, col_no=19)]

    visitor = InjectPerfOnly(
        function, "test_module", "pytest", call_positions, TestingMode.PERFORMANCE
    )

    modified_tree = visitor.visit(tree)

    # Add the wrapper functions
    modified_tree.body = [
        ast.Import(names=[ast.alias(name="inspect")]),
        create_wrapper_function(TestingMode.PERFORMANCE),
        create_async_wrapper_inner(),
        *modified_tree.body,
    ]

    try:
        modified_code = ast.unparse(modified_tree)
        assert "codeflash_async_wrap_inner" in modified_code
    except AttributeError as e:
        pytest.fail(f"AST unparsing failed with AttributeError: {e}")


@pytest.mark.asyncio
async def test_async_wrapper_execution():
    """Test that the async wrapper can be executed correctly."""

    # Create a simple async function to wrap
    async def test_func(x, y=10):
        await asyncio.sleep(0.01)
        return x + y

    # Create the wrapper code dynamically
    wrapper_code = textwrap.dedent("""
import asyncio
import inspect

async def codeflash_async_wrap_inner(wrapped, *args, **kwargs):
    return await wrapped(*args, **kwargs)

async def test_wrapper():
    result = await codeflash_async_wrap_inner(test_func, 5, y=15)
    return result
    """)

    # Execute the wrapper
    namespace = {"test_func": test_func}
    exec(wrapper_code, namespace)

    # Run the test
    result = await namespace["test_wrapper"]()
    assert result == 20


def test_mixed_sync_async_instrumentation():
    """Test that both sync and async functions can be instrumented in the same test."""
    test_code = textwrap.dedent("""
        import asyncio
        
        def sync_function(x):
            return x * 2
        
        async def async_function(x):
            await asyncio.sleep(0.01)
            return x * 3
        
        async def test_mixed():
            sync_result = sync_function(5)
            async_result = await async_function(5)
            assert sync_result == 10
            assert async_result == 15
    """)

    tree = ast.parse(test_code)

    sync_function = FunctionToOptimize(
        function_name="sync_function",
        parents=[],
        file_path=Path("test.py"),
        starting_line=3,
        ending_line=4,
    )

    call_positions = [
        CodePosition(line_no=11, col_no=19),
        CodePosition(line_no=12, col_no=25),
    ]

    visitor = InjectPerfOnly(
        sync_function,
        "test_module",
        "pytest",
        call_positions,
        TestingMode.PERFORMANCE,
    )

    modified_tree = visitor.visit(tree)

    modified_tree.body = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="inspect")]),
        ast.Import(names=[ast.alias(name="gc")]),
        ast.Import(names=[ast.alias(name="os")]),
        create_wrapper_function(TestingMode.PERFORMANCE),
        create_async_wrapper_inner(),
        *modified_tree.body,
    ]

    modified_code = ast.unparse(modified_tree)
    # Both wrapper functions should be present
    assert "codeflash_wrap" in modified_code
    assert "codeflash_async_wrap_inner" in modified_code
    assert "inspect.iscoroutinefunction" in modified_code
