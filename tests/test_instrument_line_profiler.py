import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codeflash.code_utils.line_profile_utils import add_decorator_imports, contains_jit_decorator
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def test_add_decorator_imports_helper_in_class():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sort_classmethod", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file.as_posix()}')

from code_to_optimize.bubble_sort_in_class import BubbleSortClass


@codeflash_line_profile
def sort_classmethod(x):
    y = BubbleSortClass()
    return y.sorter(x)
"""
        expected_code_helper = """from line_profiler import profile as codeflash_line_profile


def hi():
    pass


class BubbleSortClass:
    @codeflash_line_profile
    def __init__(self):
        pass

    @codeflash_line_profile
    def sorter(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    def helper(self, arr, j):
        return arr[j] > arr[j + 1]
"""
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions[0].file_path.read_text("utf-8") == expected_code_helper
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_in_nested_class():
    #Need to invert the assert once the helper detection is fixed
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_nested_classmethod.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sort_classmethod", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file.as_posix()}')

from code_to_optimize.bubble_sort_in_nested_class import WrapperClass


@codeflash_line_profile
def sort_classmethod(x):
    y = WrapperClass.BubbleSortClass()
    return y.sorter(x)
"""
        assert code_path.read_text("utf-8") == expected_code_main
        # WrapperClass.__init__ is now detected as a helper since WrapperClass.BubbleSortClass() instantiates it
        assert len(code_context.helper_functions) == 1
        assert code_context.helper_functions[0].qualified_name == "WrapperClass.__init__"
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_nodeps():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file.as_posix()}')


@codeflash_line_profile
def sorter(arr):
    print("codeflash stdout: Sorting list")
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
    print(f"result: {{arr}}")
    return arr
"""
        assert code_path.read_text("utf-8") == expected_code_main
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_outside():
    code_path = (Path(__file__).parent.resolve() / "../code_to_optimize/bubble_sort_deps.py").resolve()
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = (Path(__file__).parent / "..").resolve()
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter_deps", parents=[], file_path=code_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file.as_posix()}')

from code_to_optimize.bubble_sort_dep1_helper import dep1_comparer
from code_to_optimize.bubble_sort_dep2_swap import dep2_swap


@codeflash_line_profile
def sorter_deps(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if dep1_comparer(arr, j):
                dep2_swap(arr, j)
    return arr

"""
        expected_code_helper1 = """from line_profiler import profile as codeflash_line_profile


@codeflash_line_profile
def dep1_comparer(arr, j: int) -> bool:
    return arr[j] > arr[j + 1]
"""
        expected_code_helper2="""from line_profiler import profile as codeflash_line_profile


@codeflash_line_profile
def dep2_swap(arr, j):
    temp = arr[j]
    arr[j] = arr[j + 1]
    arr[j + 1] = temp
"""
        assert code_path.read_text("utf-8") == expected_code_main
        assert code_context.helper_functions[0].file_path.read_text("utf-8") == expected_code_helper1
        assert code_context.helper_functions[1].file_path.read_text("utf-8") == expected_code_helper2
    finally:
        func_optimizer.write_code_and_helpers(
            func_optimizer.function_to_optimize_source_code, original_helper_code, func_optimizer.function_to_optimize.file_path
        )

def test_add_decorator_imports_helper_in_dunder_class():
    code_str = """def sorter(arr):
    ans = helper(arr)
    return ans
class helper:
    def __init__(self, arr):
        return arr.sort()"""
    code_path = TemporaryDirectory()
    code_write_path = Path(code_path.name) / "dunder_class.py"
    code_write_path.write_text(code_str,"utf-8")
    tests_root = Path(__file__).parent.resolve() / "../code_to_optimize/tests/pytest/"
    project_root_path = Path(code_path.name)
    run_cwd = Path(__file__).parent.parent.resolve()
    test_config = TestConfig(
        tests_root=tests_root,
        tests_project_rootdir=project_root_path,
        project_root_path=project_root_path,
        test_framework="pytest",
        pytest_cmd="pytest",
    )
    func = FunctionToOptimize(function_name="sorter", parents=[], file_path=code_write_path)
    func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
    os.chdir(run_cwd)
    #func_optimizer = pass
    try:
        ctx_result = func_optimizer.get_code_optimization_context()
        code_context: CodeOptimizationContext = ctx_result.unwrap()
        original_helper_code: dict[Path, str] = {}
        helper_function_paths = {hf.file_path for hf in code_context.helper_functions}
        for helper_function_path in helper_function_paths:
            with helper_function_path.open(encoding="utf8") as f:
                helper_code = f.read()
                original_helper_code[helper_function_path] = helper_code
        line_profiler_output_file = add_decorator_imports(
            func_optimizer.function_to_optimize, code_context)
        expected_code_main = f"""from line_profiler import profile as codeflash_line_profile
codeflash_line_profile.enable(output_prefix='{line_profiler_output_file.as_posix()}')


@codeflash_line_profile
def sorter(arr):
    ans = helper(arr)
    return ans
class helper:
    @codeflash_line_profile
    def __init__(self, arr):
        return arr.sort()
"""
        assert code_write_path.read_text("utf-8") == expected_code_main
    finally:
        pass


# ============================================================================
# Tests for contains_jit_decorator
# ============================================================================


class TestContainsJitDecoratorNumba:
    """Tests for numba JIT decorator detection."""

    def test_numba_jit_with_module_prefix(self):
        code = """
import numba

@numba.jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_jit_with_alias(self):
        code = """
import numba as nb

@nb.jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_jit_direct_import(self):
        code = """
from numba import jit

@jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_jit_direct_import_with_alias(self):
        code = """
from numba import jit as my_jit

@my_jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_jit_with_arguments(self):
        code = """
import numba

@numba.jit(nopython=True)
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_jit_direct_import_with_arguments(self):
        code = """
from numba import jit

@jit(nopython=True, cache=True)
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_njit(self):
        code = """
from numba import njit

@njit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_njit_with_module_prefix(self):
        code = """
import numba

@numba.njit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_vectorize(self):
        code = """
from numba import vectorize

@vectorize
def my_func(x):
    return x * 2
"""
        assert contains_jit_decorator(code)

    def test_numba_guvectorize(self):
        code = """
import numba

@numba.guvectorize(['void(float64[:], float64[:])'], '(n)->(n)')
def my_func(x, res):
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_stencil(self):
        code = """
from numba import stencil

@stencil
def my_kernel(a):
    return a[0, 0] + a[0, 1]
"""
        assert contains_jit_decorator(code)

    def test_numba_cfunc(self):
        code = """
from numba import cfunc

@cfunc("float64(float64)")
def my_func(x):
    return x * 2
"""
        assert contains_jit_decorator(code)

    def test_numba_generated_jit(self):
        code = """
from numba import generated_jit

@generated_jit
def my_func(x):
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_cuda_jit(self):
        code = """
import numba

@numba.cuda.jit
def my_kernel():
    pass
"""
        assert contains_jit_decorator(code)

    def test_numba_cuda_jit_with_alias(self):
        code = """
import numba as nb

@nb.cuda.jit
def my_kernel():
    pass
"""
        assert contains_jit_decorator(code)


class TestContainsJitDecoratorTorch:
    """Tests for torch JIT decorator detection."""

    def test_torch_compile(self):
        code = """
import torch

@torch.compile
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_compile_with_alias(self):
        code = """
import torch as th

@th.compile
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_compile_direct_import(self):
        code = """
from torch import compile

@compile
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_compile_with_arguments(self):
        code = """
import torch

@torch.compile(mode="reduce-overhead")
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_jit_script(self):
        code = """
import torch

@torch.jit.script
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_jit_script_with_alias(self):
        code = """
import torch as th

@th.jit.script
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_jit_trace(self):
        code = """
import torch

@torch.jit.trace
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_jit_imported_then_script(self):
        code = """
from torch import jit

@jit.script
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_torch_jit_imported_then_trace(self):
        code = """
from torch import jit

@jit.trace
def my_func():
    pass
"""
        assert contains_jit_decorator(code)


class TestContainsJitDecoratorTensorFlow:
    """Tests for TensorFlow JIT decorator detection."""

    def test_tensorflow_function_with_tf_alias(self):
        code = """
import tensorflow as tf

@tf.function
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_tensorflow_function_full_name(self):
        code = """
import tensorflow

@tensorflow.function
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_tensorflow_function_direct_import(self):
        code = """
from tensorflow import function

@function
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_tensorflow_function_with_arguments(self):
        code = """
import tensorflow as tf

@tf.function(jit_compile=True)
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_tf_function_direct_import_alias(self):
        code = """
from tensorflow import function as tf_func

@tf_func
def my_func():
    pass
"""
        assert contains_jit_decorator(code)


class TestContainsJitDecoratorJax:
    """Tests for JAX JIT decorator detection."""

    def test_jax_jit(self):
        code = """
import jax

@jax.jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_jax_jit_with_alias(self):
        code = """
import jax as j

@j.jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_jax_jit_direct_import(self):
        code = """
from jax import jit

@jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_jax_jit_direct_import_with_alias(self):
        code = """
from jax import jit as jax_jit

@jax_jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_jax_jit_with_arguments(self):
        code = """
import jax

@jax.jit(static_argnums=(0,))
def my_func(x, y):
    pass
"""
        assert contains_jit_decorator(code)


class TestContainsJitDecoratorNegativeCases:
    """Tests that should NOT detect JIT decorators."""

    def test_no_decorators(self):
        code = """
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_other_decorator(self):
        code = """
import functools

@functools.lru_cache
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_custom_decorator(self):
        code = """
def my_decorator(func):
    return func

@my_decorator
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_property_decorator(self):
        code = """
class MyClass:
    @property
    def my_prop(self):
        return self._value
"""
        assert not contains_jit_decorator(code)

    def test_staticmethod_decorator(self):
        code = """
class MyClass:
    @staticmethod
    def my_func():
        pass
"""
        assert not contains_jit_decorator(code)

    def test_classmethod_decorator(self):
        code = """
class MyClass:
    @classmethod
    def my_func(cls):
        pass
"""
        assert not contains_jit_decorator(code)

    def test_jit_in_comment(self):
        code = """
# @numba.jit
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_jit_in_string(self):
        code = '''
def my_func():
    """This function could use @numba.jit decorator."""
    pass
'''
        assert not contains_jit_decorator(code)

    def test_unrelated_jit_name(self):
        code = """
def jit():
    pass

@jit
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_unrelated_module_with_jit_attribute(self):
        code = """
import my_module

@my_module.jit
def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_numba_import_but_no_decorator(self):
        code = """
import numba

def my_func():
    pass
"""
        assert not contains_jit_decorator(code)

    def test_jit_variable_not_decorator(self):
        code = """
from numba import jit

def my_func():
    x = jit
    pass
"""
        assert not contains_jit_decorator(code)


class TestContainsJitDecoratorEdgeCases:
    """Edge case tests for JIT decorator detection."""

    def test_multiple_decorators_with_jit(self):
        code = """
import numba
import functools

@functools.lru_cache
@numba.jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_multiple_decorators_jit_first(self):
        code = """
import numba
import functools

@numba.jit
@functools.lru_cache
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_async_function_with_jit(self):
        code = """
import numba

@numba.jit
async def my_func():
    pass
"""
        assert contains_jit_decorator(code) is False

    def test_method_in_class_with_jit(self):
        code = """
import numba

class MyClass:
    @numba.jit
    def my_method(self):
        pass
"""
        assert contains_jit_decorator(code)

    def test_nested_class_method_with_jit(self):
        code = """
import numba

class Outer:
    class Inner:
        @numba.jit
        def my_method(self):
            pass
"""
        assert contains_jit_decorator(code)

    def test_multiple_functions_one_with_jit(self):
        code = """
import numba

def func_a():
    pass

@numba.jit
def func_b():
    pass

def func_c():
    pass
"""
        assert contains_jit_decorator(code)

    def test_multiple_jit_functions(self):
        code = """
import numba
import jax

@numba.jit
def func_a():
    pass

@jax.jit
def func_b():
    pass
"""
        assert contains_jit_decorator(code)

    def test_empty_code(self):
        code = ""
        assert not contains_jit_decorator(code)

    def test_syntax_error_code(self):
        code = """
def func(
    pass
"""
        assert not contains_jit_decorator(code)

    def test_whitespace_only(self):
        code = "   \n\n   \t\t\n"
        assert not contains_jit_decorator(code)

    def test_only_imports(self):
        code = """
import numba
from jax import jit
"""
        assert not contains_jit_decorator(code)

    def test_lambda_cannot_have_decorator(self):
        # Lambdas cannot have decorators in Python
        code = """
import numba

f = lambda x: x * 2
"""
        assert not contains_jit_decorator(code)

    def test_mixed_imports_and_aliases(self):
        code = """
import numba as nb
from torch import compile as torch_compile
import jax

@nb.jit
def func_a():
    pass
"""
        assert contains_jit_decorator(code)

    def test_decorator_in_different_module_context(self):
        code = """
# Import numba for numeric computation
import numba

# Some other code
x = 5

class Processor:
    @numba.njit
    def process(self, data):
        return data * 2
"""
        assert contains_jit_decorator(code)

    def test_from_import_star_not_tracked(self):
        # Star imports are not tracked, so @jit won't be detected
        code = """
from numba import *

@jit
def my_func():
    pass
"""
        # Star imports are not tracked, so this returns False
        assert not contains_jit_decorator(code)

    def test_multiple_from_imports_same_module(self):
        code = """
from numba import jit
from numba import njit

@njit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)

    def test_reimport_with_different_alias(self):
        code = """
from numba import jit
from numba import jit as fast_jit

@fast_jit
def my_func():
    pass
"""
        assert contains_jit_decorator(code)


class TestContainsJitDecoratorComplexCases:
    """Complex real-world scenarios for JIT decorator detection."""

    def test_realistic_numba_code(self):
        code = """
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def compute_sum(arr):
    total = 0.0
    for i in prange(len(arr)):
        total += arr[i]
    return total

def main():
    data = np.random.rand(1000000)
    result = compute_sum(data)
    print(result)
"""
        assert contains_jit_decorator(code)

    def test_realistic_torch_code(self):
        code = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    @torch.compile
    def forward(self, x):
        return self.linear(x)
"""
        assert contains_jit_decorator(code)

    def test_realistic_jax_code(self):
        code = """
import jax
import jax.numpy as jnp
from jax import jit, grad

@jit
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

grad_fn = grad(loss_fn)
"""
        assert contains_jit_decorator(code)

    def test_realistic_tensorflow_code(self):
        code = """
import tensorflow as tf

@tf.function(jit_compile=True)
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.reduce_mean(tf.square(predictions - y))
    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients
"""
        assert contains_jit_decorator(code)

    def test_file_with_many_functions_one_jit(self):
        code = """
import os
import sys
import numpy as np
from numba import njit

def helper_a():
    return 1

def helper_b():
    return 2

class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self):
        pass

@njit
def fast_compute(x, y):
    return x + y

def main():
    result = fast_compute(1, 2)
    print(result)

if __name__ == "__main__":
    main()
"""
        assert contains_jit_decorator(code)
