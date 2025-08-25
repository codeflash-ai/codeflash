from __future__ import annotations

import ast
import math
import os
import sys
import tempfile
from pathlib import Path

from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.code_utils.instrument_existing_tests import (
    FunctionImportedAsVisitor,
    inject_profiling_into_existing_test,
)
from codeflash.code_utils.line_profile_utils import add_decorator_imports
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import (
    CodeOptimizationContext,
    CodePosition,
    FunctionParent,
    TestFile,
    TestFiles,
    TestingMode,
    TestsInFile,
    TestType,
)
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig
from codeflash.code_utils.code_utils import get_run_tmp_file
import pytest

async_codeflash_wrap_string = """async def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{{test_module_name}}:{{test_class_name}}:{{test_name}}:{{line_id}}:{{loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{line_id}}_{{codeflash_test_index}}'
    test_stdout_tag = f'{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}'
    print(f'!$######{{test_stdout_tag}}######$!')
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        ret = wrapped(*args, **kwargs)
        if inspect.isawaitable(ret):
            counter = time.perf_counter_ns()
            return_value = await ret
        else:
            return_value = ret
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{{test_stdout_tag}}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (test_module_name, test_class_name, test_name, function_name, loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value
"""

async_codeflash_wrap_perfonly_string = """async def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, loop_index, *args, **kwargs):
    test_id = f'{{test_module_name}}:{{test_class_name}}:{{test_name}}:{{line_id}}:{{loop_index}}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {{}}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{{line_id}}_{{codeflash_test_index}}'
    test_stdout_tag = f'{{test_module_name}}:{{(test_class_name + '.' if test_class_name else '')}}{{test_name}}:{{function_name}}:{{loop_index}}:{{invocation_id}}'
    print(f'!$######{{test_stdout_tag}}######$!')
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        ret = wrapped(*args, **kwargs)
        if inspect.isawaitable(ret):
            counter = time.perf_counter_ns()
            return_value = await ret
        else:
            return_value = ret
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{{test_stdout_tag}}:{{codeflash_duration}}######!')
    if exception:
        raise exception
    return return_value
"""


def test_asyncio_gather_remover():
    from codeflash.code_utils.instrument_existing_tests import AsyncIOGatherRemover
    
    test_code_with_gather = '''
import asyncio

def test_normal_function():
    result = some_function()
    assert result == 42

async def some_async_function():
    await asyncio.sleep(1)

@pytest.mark.asyncio
async def test_with_asyncio_gather():
    results = await asyncio.gather(
        async_func1(),
        async_func2(),
        async_func3()
    )
    assert len(results) == 3

def test_with_direct_gather():
    from asyncio import gather
    results = await gather(async_func1(), async_func2())
    assert len(results) == 2

@pytest.mark.asyncio
async def test_another_normal_async():
    result = await some_async_function()
    assert result == 42
'''
    

    tree = ast.parse(test_code_with_gather)
    
    remover = AsyncIOGatherRemover()
    modified_tree = remover.visit(tree)
    
    modified_code = ast.unparse(modified_tree)

    expected = '''
import asyncio

def test_normal_function():
    result = some_function()
    assert result == 42

async def some_async_function():
    await asyncio.sleep(1)

@pytest.mark.asyncio
async def test_another_normal_async():
    result = await some_async_function()
    assert result == 42
'''
    assert modified_code.strip() == expected.strip()


def test_async_perfinjector_simple_add() -> None:
    """Test async instrumentation with a simple async function."""
    code = """import asyncio
import pytest
from code_to_optimize.async_adder import async_add


@pytest.mark.asyncio
async def test_async_add():
    result = await async_add(5, 3)
    assert result == 8

    result = await async_add(10, 20)
    assert result == 30
"""
    
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        async_add_file = Path(__file__).parent.parent / "code_to_optimize" / "async_adder.py"
        func = FunctionToOptimize(function_name="async_add", parents=[], file_path=async_add_file, is_async=True)
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name),
            [CodePosition(8, 18), CodePosition(11, 18)],
            func,
            Path(f.name).parent,
            "pytest",
        )
        os.chdir(original_cwd)
        
    assert success
    
    expected = (
        """import asyncio
import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import pytest

from code_to_optimize.async_adder import async_add


"""
        + async_codeflash_wrap_string
        + """
@pytest.mark.asyncio
async def test_async_add():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{tmp_dir_path}_{{codeflash_iteration}}.sqlite')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    result = await codeflash_wrap(async_add, '{module_path}', None, 'test_async_add', 'async_add', '0', codeflash_loop_index, codeflash_cur, codeflash_con, 5, 3)
    assert result == 8
    result = await codeflash_wrap(async_add, '{module_path}', None, 'test_async_add', 'async_add', '2', codeflash_loop_index, codeflash_cur, codeflash_con, 10, 20)
    assert result == 30
    codeflash_con.close()
"""
    )
    
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    ).replace('"', "'")


def test_async_perfinjector_unittest_style() -> None:
    """Test async instrumentation with unittest style async tests."""
    code = """import asyncio
import unittest
from code_to_optimize.async_adder import async_add


class TestAsyncAdd(unittest.IsolatedAsyncioTestCase):
    async def test_async_add_basic(self):
        result = await async_add(1, 2)
        self.assertEqual(result, 3)

        result = await async_add(-5, 10)
        self.assertEqual(result, 5)
"""
    
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        # Point to the actual file where async_add is defined, not the temp test file
        async_add_file = Path(__file__).parent.parent / "code_to_optimize" / "async_adder.py"
        func = FunctionToOptimize(function_name="async_add", parents=[], file_path=async_add_file, is_async=True)
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name),
            [CodePosition(8, 22), CodePosition(11, 22)],
            func,
            Path(f.name).parent,
            "unittest",
        )
        os.chdir(original_cwd)
        
    assert success
    
    expected = (
        """import asyncio
import gc
import inspect
import os
import sqlite3
import time
import unittest

import dill as pickle
import timeout_decorator

from code_to_optimize.async_adder import async_add


"""
        + async_codeflash_wrap_string
        + """
class TestAsyncAdd(unittest.IsolatedAsyncioTestCase):

    @timeout_decorator.timeout(15)
    async def test_async_add_basic(self):
        result = await codeflash_wrap(async_add, '{module_path}', 'TestAsyncAdd', 'test_async_add_basic', 'async_add', '0', codeflash_loop_index, codeflash_cur, codeflash_con, 1, 2)
        self.assertEqual(result, 3)
        result = await codeflash_wrap(async_add, '{module_path}', 'TestAsyncAdd', 'test_async_add_basic', 'async_add', '2', codeflash_loop_index, codeflash_cur, codeflash_con, -5, 10)
        self.assertEqual(result, 5)
"""
    )
    
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    ).replace('"', "'")


def test_async_perfinjector_parametrized_tests() -> None:
    """Test async instrumentation with parametrized async tests."""
    code = """import asyncio
import pytest
from code_to_optimize.async_adder import async_add


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),
        (5, 7, 12),
        (-3, 8, 5),
        (0, 0, 0),
    ],
)
@pytest.mark.asyncio
async def test_async_add_parametrized(a, b, expected):
    result = await async_add(a, b)
    assert result == expected
"""
    
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        # Point to the actual file where async_add is defined, not the temp test file
        async_add_file = Path(__file__).parent.parent / "code_to_optimize" / "async_adder.py"
        func = FunctionToOptimize(function_name="async_add", parents=[], file_path=async_add_file, is_async=True)
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name),
            [CodePosition(16, 18)],
            func,
            Path(f.name).parent,
            "pytest",
        )
        os.chdir(original_cwd)
        
    assert success
    
    expected = (
        """import asyncio
import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import pytest

from code_to_optimize.async_adder import async_add


"""
        + async_codeflash_wrap_string
        + """
@pytest.mark.parametrize('a, b, expected', [(1, 2, 3), (5, 7, 12), (-3, 8, 5), (0, 0, 0)])
@pytest.mark.asyncio
async def test_async_add_parametrized(a, b, expected):
    result = await async_add(a, b)
    assert result == expected
"""
    )
    
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    ).replace('"', "'")


def test_async_perfinjector_performance_mode() -> None:
    """Test async instrumentation in performance mode (without return value storage)."""
    code = """import asyncio
import pytest
from code_to_optimize.async_adder import async_add


@pytest.mark.asyncio
async def test_async_add_perf():
    result = await async_add(100, 200)
    assert result == 300
"""
    
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(code)
        f.flush()
        # Point to the actual file where async_add is defined, not the temp test file
        async_add_file = Path(__file__).parent.parent / "code_to_optimize" / "async_adder.py"
        func = FunctionToOptimize(function_name="async_add", parents=[], file_path=async_add_file, is_async=True)
        original_cwd = Path.cwd()
        run_cwd = Path(__file__).parent.parent.resolve()
        os.chdir(run_cwd)
        success, new_test = inject_profiling_into_existing_test(
            Path(f.name),
            [CodePosition(8, 18)],
            func,
            Path(f.name).parent,
            "pytest",
            mode=TestingMode.PERFORMANCE,
        )
        os.chdir(original_cwd)
        
    assert success
    
    expected = (
        """import asyncio
import gc
import inspect
import os
import time

import pytest

from code_to_optimize.async_adder import async_add


"""
        + async_codeflash_wrap_perfonly_string
        + """
@pytest.mark.asyncio
async def test_async_add_perf():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = await codeflash_wrap(async_add, '{module_path}', None, 'test_async_add_perf', 'async_add', '0', codeflash_loop_index, 100, 200)
    assert result == 300
"""
    )
    
    assert new_test.replace('"', "'") == expected.format(
        module_path=Path(f.name).name, tmp_dir_path=get_run_tmp_file(Path("test_return_values"))
    ).replace('"', "'")