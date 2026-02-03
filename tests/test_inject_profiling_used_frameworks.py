"""Unit tests for inject_profiling_into_existing_test with different used_frameworks values.

These tests verify that the wrapper function is correctly generated with GPU device
synchronization code for different framework imports (torch, tensorflow, jax).
"""

from __future__ import annotations

import re
from pathlib import Path

from codeflash.code_utils.instrument_existing_tests import (
    detect_frameworks_from_code,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode


def normalize_instrumented_code(code: str) -> str:
    """Normalize instrumented code by replacing dynamic paths with placeholders.

    This allows comparing instrumented code across test runs where temp paths differ.
    Also normalizes f-string quoting differences between Python versions (Python 3.12+
    allows single quotes inside single-quoted f-strings via PEP 701, but libcst
    generates double-quoted f-strings for compatibility with older versions).
    """
    # Normalize database path
    code = re.sub(r"sqlite3\.connect\(f'[^']+'", "sqlite3.connect(f'{CODEFLASH_DB_PATH}'", code)
    # Normalize f-string that contains the test_stdout_tag assignment
    # This specific f-string has internal single quotes, so libcst uses double quotes
    # on Python < 3.12, but single quotes on Python 3.12+
    code = re.sub(r'test_stdout_tag = f"([^"]+)"', r"test_stdout_tag = f'\1'", code)
    return code


# ============================================================================
# Expected instrumented code for BEHAVIOR mode
# ============================================================================

EXPECTED_NO_FRAMEWORKS_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TORCH_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TORCH_ALIASED_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import torch as th
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = th.cuda.is_available() and th.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(th.backends, 'mps') and th.backends.mps.is_available() and hasattr(th.mps, 'synchronize')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            th.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            th.mps.synchronize()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            th.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            th.mps.synchronize()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TORCH_SUBMODULE_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import torch
from mymodule import my_function
from torch import nn


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TENSORFLOW_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import tensorflow
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TENSORFLOW_ALIASED_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import tensorflow as tf
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_tf = hasattr(tf.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_tf:
            tf.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_tf:
            tf.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_JAX_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import jax
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_jax:
            jax.block_until_ready(return_value)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_JAX_ALIASED_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import jax as jnp
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_jax = hasattr(jnp, 'block_until_ready')
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_jax:
            jnp.block_until_ready(return_value)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TORCH_TENSORFLOW_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import tensorflow
import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    _codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_ALL_FRAMEWORKS_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import jax
import tensorflow
import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    _codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')
    _codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_jax:
            jax.block_until_ready(return_value)
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

# ============================================================================
# Expected instrumented code for PERFORMANCE mode
# ============================================================================

EXPECTED_NO_FRAMEWORKS_PERFORMANCE = """import gc
import os
import time

from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""

EXPECTED_TORCH_PERFORMANCE = """import gc
import os
import time

import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""

EXPECTED_TENSORFLOW_PERFORMANCE = """import gc
import os
import time

import tensorflow
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""

EXPECTED_JAX_PERFORMANCE = """import gc
import os
import time

import jax
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')
    gc.disable()
    try:
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_jax:
            jax.block_until_ready(return_value)
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""

EXPECTED_ALL_FRAMEWORKS_PERFORMANCE = """import gc
import os
import time

import jax
import tensorflow
import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    _codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')
    _codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')
    gc.disable()
    try:
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        counter = time.perf_counter_ns()
        return_value = codeflash_wrapped(*args, **kwargs)
        if _codeflash_should_sync_cuda:
            torch.cuda.synchronize()
        elif _codeflash_should_sync_mps:
            torch.mps.synchronize()
        if _codeflash_should_sync_jax:
            jax.block_until_ready(return_value)
        if _codeflash_should_sync_tf:
            tensorflow.test.experimental.sync_devices()
        codeflash_duration = time.perf_counter_ns() - counter
    except Exception as e:
        codeflash_duration = time.perf_counter_ns() - counter
        exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""


# ============================================================================
# Tests for detect_frameworks_from_code
# ============================================================================


class TestDetectFrameworksFromCode:
    """Tests for the detect_frameworks_from_code helper function."""

    def test_no_frameworks(self) -> None:
        """Test detection with no GPU framework imports."""
        code = """import os
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {}
        assert result == expected

    def test_torch_import(self) -> None:
        """Test detection with torch import."""
        code = """import torch
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "torch"}
        assert result == expected

    def test_torch_aliased_import(self) -> None:
        """Test detection with torch imported as alias."""
        code = """import torch as th
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "th"}
        assert result == expected

    def test_torch_submodule_import(self) -> None:
        """Test detection with torch submodule import (from torch import nn)."""
        code = """from torch import nn
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "torch"}
        assert result == expected

    def test_torch_dotted_import(self) -> None:
        """Test detection with torch.cuda or torch.nn import."""
        code = """import torch.cuda
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "torch"}
        assert result == expected

    def test_tensorflow_import(self) -> None:
        """Test detection with tensorflow import."""
        code = """import tensorflow
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"tensorflow": "tensorflow"}
        assert result == expected

    def test_tensorflow_aliased_import(self) -> None:
        """Test detection with tensorflow imported as alias."""
        code = """import tensorflow as tf
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"tensorflow": "tf"}
        assert result == expected

    def test_tensorflow_submodule_import(self) -> None:
        """Test detection with tensorflow submodule import."""
        code = """from tensorflow import keras
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"tensorflow": "tensorflow"}
        assert result == expected

    def test_jax_import(self) -> None:
        """Test detection with jax import."""
        code = """import jax
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"jax": "jax"}
        assert result == expected

    def test_jax_aliased_import(self) -> None:
        """Test detection with jax imported as alias."""
        code = """import jax as jnp
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"jax": "jnp"}
        assert result == expected

    def test_jax_submodule_import(self) -> None:
        """Test detection with jax submodule import."""
        code = """from jax import numpy as jnp
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"jax": "jax"}
        assert result == expected

    def test_multiple_frameworks(self) -> None:
        """Test detection with multiple framework imports."""
        code = """import torch
import tensorflow
import jax
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "torch", "tensorflow": "tensorflow", "jax": "jax"}
        assert result == expected

    def test_multiple_frameworks_aliased(self) -> None:
        """Test detection with multiple aliased framework imports."""
        code = """import torch as th
import tensorflow as tf
import jax as jnp
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        expected = {"torch": "th", "tensorflow": "tf", "jax": "jnp"}
        assert result == expected

    def test_syntax_error_returns_empty(self) -> None:
        """Test that syntax errors return empty dict."""
        code = """this is not valid python code !!!"""
        result = detect_frameworks_from_code(code)
        expected = {}
        assert result == expected


# ============================================================================
# Tests for inject_profiling_into_existing_test - BEHAVIOR mode
# ============================================================================


class TestInjectProfilingBehaviorMode:
    """Tests for inject_profiling_into_existing_test in BEHAVIOR mode."""

    def test_no_frameworks_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with no GPU framework imports in BEHAVIOR mode."""
        code = """from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(4, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_NO_FRAMEWORKS_BEHAVIOR
        assert result == expected

    def test_torch_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch import in BEHAVIOR mode."""
        code = """import torch
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_BEHAVIOR
        assert result == expected

    def test_torch_aliased_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch imported as alias in BEHAVIOR mode."""
        code = """import torch as th
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_ALIASED_BEHAVIOR
        assert result == expected

    def test_torch_submodule_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch submodule import in BEHAVIOR mode."""
        code = """from torch import nn
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_SUBMODULE_BEHAVIOR
        assert result == expected

    def test_tensorflow_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with TensorFlow import in BEHAVIOR mode."""
        code = """import tensorflow
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TENSORFLOW_BEHAVIOR
        assert result == expected

    def test_tensorflow_aliased_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with TensorFlow imported as alias in BEHAVIOR mode."""
        code = """import tensorflow as tf
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TENSORFLOW_ALIASED_BEHAVIOR
        assert result == expected

    def test_jax_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with JAX import in BEHAVIOR mode."""
        code = """import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_JAX_BEHAVIOR
        assert result == expected

    def test_jax_aliased_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with JAX imported as alias in BEHAVIOR mode."""
        code = """import jax as jnp
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_JAX_ALIASED_BEHAVIOR
        assert result == expected

    def test_torch_and_tensorflow_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with both PyTorch and TensorFlow imports in BEHAVIOR mode."""
        code = """import torch
import tensorflow
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(6, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_TENSORFLOW_BEHAVIOR
        assert result == expected

    def test_all_three_frameworks_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch, TensorFlow, and JAX imports in BEHAVIOR mode."""
        code = """import torch
import tensorflow
import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_ALL_FRAMEWORKS_BEHAVIOR
        assert result == expected


# ============================================================================
# Tests for inject_profiling_into_existing_test - PERFORMANCE mode
# ============================================================================


class TestInjectProfilingPerformanceMode:
    """Tests for inject_profiling_into_existing_test in PERFORMANCE mode."""

    def test_no_frameworks_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with no GPU framework imports in PERFORMANCE mode."""
        code = """from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(4, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_NO_FRAMEWORKS_PERFORMANCE
        assert result == expected

    def test_torch_import_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch import in PERFORMANCE mode."""
        code = """import torch
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_PERFORMANCE
        assert result == expected

    def test_tensorflow_import_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with TensorFlow import in PERFORMANCE mode."""
        code = """import tensorflow
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TENSORFLOW_PERFORMANCE
        assert result == expected

    def test_jax_import_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with JAX import in PERFORMANCE mode."""
        code = """import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_JAX_PERFORMANCE
        assert result == expected

    def test_all_frameworks_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch, TensorFlow, and JAX imports in PERFORMANCE mode."""
        code = """import torch
import tensorflow
import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_ALL_FRAMEWORKS_PERFORMANCE
        assert result == expected


# ============================================================================
# Expected instrumented code for GPU timing mode
# ============================================================================

EXPECTED_TORCH_GPU_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_use_gpu_timer = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    if _codeflash_use_gpu_timer:
        try:
            _codeflash_start_event = torch.cuda.Event(enable_timing=True)
            _codeflash_end_event = torch.cuda.Event(enable_timing=True)
            _codeflash_start_event.record()
            return_value = codeflash_wrapped(*args, **kwargs)
            _codeflash_end_event.record()
            torch.cuda.synchronize()
            codeflash_duration = int(_codeflash_start_event.elapsed_time(_codeflash_end_event) * 1000000)
        except Exception as e:
            torch.cuda.synchronize()
            codeflash_duration = 0
            exception = e
    else:
        try:
            if _codeflash_should_sync_cuda:
                torch.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                torch.mps.synchronize()
            counter = time.perf_counter_ns()
            return_value = codeflash_wrapped(*args, **kwargs)
            if _codeflash_should_sync_cuda:
                torch.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                torch.mps.synchronize()
            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""

EXPECTED_TORCH_GPU_PERFORMANCE = """import gc
import os
import time

import torch
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_use_gpu_timer = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and hasattr(torch.mps, 'synchronize')
    gc.disable()
    if _codeflash_use_gpu_timer:
        try:
            _codeflash_start_event = torch.cuda.Event(enable_timing=True)
            _codeflash_end_event = torch.cuda.Event(enable_timing=True)
            _codeflash_start_event.record()
            return_value = codeflash_wrapped(*args, **kwargs)
            _codeflash_end_event.record()
            torch.cuda.synchronize()
            codeflash_duration = int(_codeflash_start_event.elapsed_time(_codeflash_end_event) * 1000000)
        except Exception as e:
            torch.cuda.synchronize()
            codeflash_duration = 0
            exception = e
    else:
        try:
            if _codeflash_should_sync_cuda:
                torch.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                torch.mps.synchronize()
            counter = time.perf_counter_ns()
            return_value = codeflash_wrapped(*args, **kwargs)
            if _codeflash_should_sync_cuda:
                torch.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                torch.mps.synchronize()
            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}:{codeflash_duration}######!')
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, 1, 2)
    assert result == 3
"""

EXPECTED_TORCH_ALIASED_GPU_BEHAVIOR = """import gc
import inspect
import os
import sqlite3
import time

import dill as pickle
import torch as th
from mymodule import my_function


def codeflash_wrap(codeflash_wrapped, codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_line_id, codeflash_loop_index, codeflash_cur, codeflash_con, *args, **kwargs):
    test_id = f'{codeflash_test_module_name}:{codeflash_test_class_name}:{codeflash_test_name}:{codeflash_line_id}:{codeflash_loop_index}'
    if not hasattr(codeflash_wrap, 'index'):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f'{codeflash_line_id}_{codeflash_test_index}'
    test_stdout_tag = f'{codeflash_test_module_name}:{(codeflash_test_class_name + '.' if codeflash_test_class_name else '')}{codeflash_test_name}:{codeflash_function_name}:{codeflash_loop_index}:{invocation_id}'
    print(f'!$######{test_stdout_tag}######$!')
    exception = None
    _codeflash_use_gpu_timer = th.cuda.is_available() and th.cuda.is_initialized()
    _codeflash_should_sync_cuda = th.cuda.is_available() and th.cuda.is_initialized()
    _codeflash_should_sync_mps = not _codeflash_should_sync_cuda and hasattr(th.backends, 'mps') and th.backends.mps.is_available() and hasattr(th.mps, 'synchronize')
    gc.disable()
    if _codeflash_use_gpu_timer:
        try:
            _codeflash_start_event = th.cuda.Event(enable_timing=True)
            _codeflash_end_event = th.cuda.Event(enable_timing=True)
            _codeflash_start_event.record()
            return_value = codeflash_wrapped(*args, **kwargs)
            _codeflash_end_event.record()
            th.cuda.synchronize()
            codeflash_duration = int(_codeflash_start_event.elapsed_time(_codeflash_end_event) * 1000000)
        except Exception as e:
            th.cuda.synchronize()
            codeflash_duration = 0
            exception = e
    else:
        try:
            if _codeflash_should_sync_cuda:
                th.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                th.mps.synchronize()
            counter = time.perf_counter_ns()
            return_value = codeflash_wrapped(*args, **kwargs)
            if _codeflash_should_sync_cuda:
                th.cuda.synchronize()
            elif _codeflash_should_sync_mps:
                th.mps.synchronize()
            codeflash_duration = time.perf_counter_ns() - counter
        except Exception as e:
            codeflash_duration = time.perf_counter_ns() - counter
            exception = e
    gc.enable()
    print(f'!######{test_stdout_tag}######!')
    pickled_return_value = pickle.dumps(exception) if exception else pickle.dumps(return_value)
    codeflash_cur.execute('INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (codeflash_test_module_name, codeflash_test_class_name, codeflash_test_name, codeflash_function_name, codeflash_loop_index, invocation_id, codeflash_duration, pickled_return_value, 'function_call'))
    codeflash_con.commit()
    if exception:
        raise exception
    return return_value

def test_my_function():
    codeflash_loop_index = int(os.environ['CODEFLASH_LOOP_INDEX'])
    codeflash_iteration = os.environ['CODEFLASH_TEST_ITERATION']
    codeflash_con = sqlite3.connect(f'{CODEFLASH_DB_PATH}')
    codeflash_cur = codeflash_con.cursor()
    codeflash_cur.execute('CREATE TABLE IF NOT EXISTS test_results (test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, runtime INTEGER, return_value BLOB, verification_type TEXT)')
    _call__bound__arguments = inspect.signature(my_function).bind(1, 2)
    _call__bound__arguments.apply_defaults()
    result = codeflash_wrap(my_function, 'test_example', None, 'test_my_function', 'my_function', '0', codeflash_loop_index, codeflash_cur, codeflash_con, *_call__bound__arguments.args, **_call__bound__arguments.kwargs)
    assert result == 3
    codeflash_con.close()
"""


# ============================================================================
# Tests for GPU timing mode
# ============================================================================


class TestInjectProfilingGpuTimingMode:
    """Tests for inject_profiling_into_existing_test with gpu=True."""

    def test_torch_gpu_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch and gpu=True in BEHAVIOR mode."""
        code = """import torch
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
            gpu=True,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_GPU_BEHAVIOR
        assert result == expected

    def test_torch_gpu_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch and gpu=True in PERFORMANCE mode."""
        code = """import torch
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
            gpu=True,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_GPU_PERFORMANCE
        assert result == expected

    def test_torch_aliased_gpu_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with PyTorch alias and gpu=True in BEHAVIOR mode."""
        code = """import torch as th
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
            gpu=True,
        )

        result = normalize_instrumented_code(instrumented_code)
        expected = EXPECTED_TORCH_ALIASED_GPU_BEHAVIOR
        assert result == expected

    def test_no_torch_gpu_flag_uses_cpu_timing(self, tmp_path: Path) -> None:
        """Test that gpu=True without torch uses standard CPU timing."""
        code = """from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(4, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
            gpu=True,
        )

        result = normalize_instrumented_code(instrumented_code)
        # gpu=True without torch should produce the same result as gpu=False
        expected = EXPECTED_NO_FRAMEWORKS_PERFORMANCE
        assert result == expected

    def test_gpu_false_with_torch_uses_device_sync(self, tmp_path: Path) -> None:
        """Test that gpu=False with torch uses device sync (existing behavior)."""
        code = """import torch
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
            gpu=False,
        )

        result = normalize_instrumented_code(instrumented_code)
        # gpu=False with torch should produce device sync code
        expected = EXPECTED_TORCH_PERFORMANCE
        assert result == expected

    def test_torch_submodule_import_gpu_mode(self, tmp_path: Path) -> None:
        """Test that gpu=True works with torch submodule imports like 'from torch import nn'."""
        code = """from torch import nn
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
            gpu=True,
        )

        assert success
        # Verify GPU timing code is present (torch detected from submodule import)
        assert "_codeflash_use_gpu_timer = torch.cuda.is_available()" in instrumented_code
        assert "torch.cuda.Event(enable_timing=True)" in instrumented_code
        assert "elapsed_time" in instrumented_code

    def test_torch_dotted_import_gpu_mode(self, tmp_path: Path) -> None:
        """Test that gpu=True works with torch dotted imports like 'import torch.nn'."""
        code = """import torch.nn
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(function_name="my_function", parents=[], file_path=Path("mymodule.py"))

        success, instrumented_code = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
            gpu=True,
        )

        assert success
        # Verify GPU timing code is present (torch detected from dotted import)
        assert "_codeflash_use_gpu_timer = torch.cuda.is_available()" in instrumented_code
        assert "torch.cuda.Event(enable_timing=True)" in instrumented_code
        assert "elapsed_time" in instrumented_code
