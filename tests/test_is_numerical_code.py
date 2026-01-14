"""Comprehensive unit tests for is_numerical_code function."""

from unittest.mock import patch

import pytest

from codeflash.code_utils.code_extractor import is_numerical_code


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestBasicNumpyUsage:
    """Test basic numpy library detection (with numba available)."""

    def test_numpy_with_standard_alias(self):
        code = """
import numpy as np
def process_data(x):
    return np.sum(x)
"""
        assert is_numerical_code(code, "process_data") is True

    def test_numpy_without_alias(self):
        code = """
import numpy
def process_data(x):
    return numpy.array(x)
"""
        assert is_numerical_code(code, "process_data") is True

    def test_numpy_from_import(self):
        code = """
from numpy import array, zeros
def create_array():
    return array([1, 2, 3])
"""
        assert is_numerical_code(code, "create_array") is True

    def test_numpy_from_import_with_alias(self):
        code = """
from numpy import array as arr
def create_array():
    return arr([1, 2, 3])
"""
        assert is_numerical_code(code, "create_array") is True

    def test_numpy_custom_alias(self):
        code = """
import numpy as custom_name
def func(x):
    return custom_name.array(x)
"""
        assert is_numerical_code(code, "func") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestNumpySubmodules:
    """Test numpy submodule imports (with numba available)."""

    def test_numpy_linalg_direct(self):
        code = """
import numpy.linalg
def func(x):
    return numpy.linalg.norm(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_linalg_aliased(self):
        code = """
import numpy.linalg as la
def func(x):
    return la.norm(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_random_aliased(self):
        code = """
import numpy.random as rng
def func():
    return rng.randint(0, 10)
"""
        assert is_numerical_code(code, "func") is True

    def test_from_numpy_import_submodule(self):
        code = """
from numpy import linalg
def func(x):
    return linalg.norm(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_from_numpy_linalg_import_function(self):
        code = """
from numpy.linalg import norm
def func(x):
    return norm(x)
"""
        assert is_numerical_code(code, "func") is True


class TestTorchUsage:
    """Test PyTorch library detection."""

    def test_torch_basic(self):
        code = """
import torch
def train_model(model):
    return torch.nn.functional.relu(model)
"""
        assert is_numerical_code(code, "train_model") is True

    def test_torch_standard_alias(self):
        code = """
import torch as th
def func(x):
    return th.tensor(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_torch_nn_alias(self):
        code = """
import torch.nn as nn
def func():
    return nn.Linear(10, 10)
"""
        assert is_numerical_code(code, "func") is True

    def test_torch_functional_alias(self):
        code = """
import torch.nn.functional as F
def func(x):
    return F.relu(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_torch_from_import(self):
        code = """
from torch.nn.functional import relu
def func(x):
    return relu(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_torch_from_import_aliased(self):
        code = """
from torch.nn.functional import softmax as sm
def func(x):
    return sm(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_torch_utils_data(self):
        code = """
import torch.utils.data as data
def func():
    return data.DataLoader([])
"""
        assert is_numerical_code(code, "func") is True


class TestTensorflowUsage:
    """Test TensorFlow library detection."""

    def test_tensorflow_basic(self):
        code = """
import tensorflow
def func():
    return tensorflow.Variable(1)
"""
        assert is_numerical_code(code, "func") is True

    def test_tensorflow_standard_alias(self):
        code = """
import tensorflow as tf
def build_model():
    return tf.keras.Sequential()
"""
        assert is_numerical_code(code, "build_model") is True

    def test_tensorflow_keras_alias(self):
        code = """
import tensorflow.keras as keras
def func():
    return keras.Sequential()
"""
        assert is_numerical_code(code, "func") is True

    def test_tensorflow_keras_layers_alias(self):
        code = """
import tensorflow.keras.layers as layers
def func():
    return layers.Dense(10)
"""
        assert is_numerical_code(code, "func") is True

    def test_tensorflow_from_import(self):
        code = """
from tensorflow import keras
def func():
    return keras.Model()
"""
        assert is_numerical_code(code, "func") is True


class TestJaxUsage:
    """Test JAX library detection."""

    def test_jax_basic(self):
        code = """
import jax
def func(x):
    return jax.grad(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_jax_numpy_alias(self):
        code = """
import jax.numpy as jnp
def func(x):
    return jnp.sum(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_from_jax_import_numpy(self):
        code = """
from jax import numpy as jnp
def func(x):
    return jnp.array(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_jax_from_import(self):
        code = """
from jax import grad, jit
def func(f):
    return grad(f)
"""
        assert is_numerical_code(code, "func") is True


class TestNumbaUsage:
    """Test Numba library detection."""

    def test_numba_jit_decorator(self):
        code = """
from numba import jit
@jit
def fast_func(x):
    return x * 2
"""
        assert is_numerical_code(code, "fast_func") is True

    def test_numba_cuda(self):
        code = """
import numba.cuda as cuda
def func():
    return cuda.device_array(10)
"""
        assert is_numerical_code(code, "func") is True

    def test_numba_basic(self):
        code = """
import numba
@numba.njit
def func(x):
    return x + 1
"""
        assert is_numerical_code(code, "func") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestScipyUsage:
    """Test SciPy library detection (with numba available)."""

    def test_scipy_basic(self):
        code = """
import scipy
def func(x):
    return scipy.integrate.quad(x, 0, 1)
"""
        assert is_numerical_code(code, "func") is True

    def test_scipy_stats(self):
        code = """
from scipy import stats
def analyze(data):
    return stats.describe(data)
"""
        assert is_numerical_code(code, "analyze") is True

    def test_scipy_stats_from_import(self):
        code = """
from scipy.stats import norm
def func(x):
    return norm.pdf(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_scipy_optimize_alias(self):
        code = """
import scipy.optimize as opt
def func(f, x0):
    return opt.minimize(f, x0)
"""
        assert is_numerical_code(code, "func") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestMathUsage:
    """Test math standard library detection (with numba available)."""

    def test_math_basic(self):
        code = """
import math
def calculate(x):
    return math.sqrt(x)
"""
        assert is_numerical_code(code, "calculate") is True

    def test_math_from_import(self):
        code = """
from math import sqrt, sin, cos
def calculate(x):
    return sqrt(sin(x) ** 2 + cos(x) ** 2)
"""
        assert is_numerical_code(code, "calculate") is True

    def test_math_aliased(self):
        code = """
import math as m
def calculate(x):
    return m.pi * x
"""
        assert is_numerical_code(code, "calculate") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestClassMethods:
    """Test detection in class methods, staticmethods, and classmethods (with numba available)."""

    def test_regular_method_with_numpy(self):
        code = """
import numpy as np
class DataProcessor:
    def process(self, data):
        return np.mean(data)
"""
        assert is_numerical_code(code, "DataProcessor.process") is True

    def test_regular_method_without_numerical(self):
        code = """
import numpy as np
class DataProcessor:
    def process(self, data):
        return np.mean(data)

    def other(self, x):
        return x + 1
"""
        assert is_numerical_code(code, "DataProcessor.other") is False

    def test_staticmethod_with_numpy(self):
        code = """
import numpy as np
class Calculator:
    @staticmethod
    def compute(x):
        return np.dot(x, x)
"""
        assert is_numerical_code(code, "Calculator.compute") is True

    def test_classmethod_with_torch(self):
        code = """
import torch
class Model:
    @classmethod
    def from_pretrained(cls, path):
        return torch.load(path)
"""
        assert is_numerical_code(code, "Model.from_pretrained") is True

    def test_multiple_decorators(self):
        code = """
import functools
import numpy as np
class MyClass:
    @staticmethod
    @functools.lru_cache
    def cached_compute(x):
        return np.sum(x)
"""
        assert is_numerical_code(code, "MyClass.cached_compute") is True


class TestNoNumericalUsage:
    """Test that non-numerical code returns False."""

    def test_simple_function(self):
        code = """
def simple_func(x):
    return x + 1
"""
        assert is_numerical_code(code, "simple_func") is False

    def test_string_manipulation(self):
        code = """
def process_string(s):
    return s.upper().strip()
"""
        assert is_numerical_code(code, "process_string") is False

    def test_list_operations(self):
        code = """
def process_list(lst):
    return [x * 2 for x in lst]
"""
        assert is_numerical_code(code, "process_list") is False

    def test_with_non_numerical_imports(self):
        code = """
import os
import json
from pathlib import Path

def process_file(path):
    return Path(path).read_text()
"""
        assert is_numerical_code(code, "process_file") is False

    def test_class_method_without_numerical(self):
        code = """
class Helper:
    def format(self, data):
        return str(data)
"""
        assert is_numerical_code(code, "Helper.format") is False


class TestFalsePositivePrevention:
    """Test that false positives are avoided."""

    def test_function_named_numpy(self):
        code = """
def numpy():
    return 1
def func():
    return numpy()
"""
        assert is_numerical_code(code, "func") is False

    def test_function_named_torch(self):
        code = """
def torch():
    return "fire"
def func():
    return torch()
"""
        assert is_numerical_code(code, "func") is False

    def test_variable_named_np(self):
        code = """
def func():
    np = 5
    return np + 1
"""
        assert is_numerical_code(code, "func") is False

    def test_class_named_math(self):
        code = """
class math:
    pass
def func():
    return math()
"""
        assert is_numerical_code(code, "func") is False


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestEdgeCases:
    """Test edge cases and special scenarios (with numba available)."""

    def test_nonexistent_function(self):
        code = """
import numpy as np
def process_data(x):
    return np.sum(x)
"""
        assert is_numerical_code(code, "nonexistent") is False

    def test_empty_function(self):
        code = """
import numpy as np
def empty_func():
    pass
"""
        assert is_numerical_code(code, "empty_func") is False

    def test_syntax_error_code(self):
        code = """
def broken_func(
    return 1
"""
        assert is_numerical_code(code, "broken_func") is False

    def test_empty_code_string(self):
        assert is_numerical_code("", "func") is False

    def test_type_annotation_with_numpy(self):
        code = """
import numpy as np
def func(x: np.ndarray):
    return x + 1
"""
        assert is_numerical_code(code, "func") is True

    def test_default_argument_with_numpy(self):
        code = """
import numpy as np
def func(dtype=np.float32):
    return dtype
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_in_docstring_only(self):
        code = """
def func(x):
    '''Uses numpy internally.'''
    return x + 1
"""
        assert is_numerical_code(code, "func") is False

    def test_async_function_with_numpy(self):
        code = """
import numpy as np
async def async_process(x):
    return np.sum(x)
"""
        assert is_numerical_code(code, "async_process") is False


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestStarImports:
    """Test handling of star imports (with numba available).

    Note: Star imports are difficult to track precisely since we'd need to
    resolve what names are actually imported from the module. The current
    implementation has limited support for star imports.
    """

    def test_star_import_with_module_reference(self):
        # Star imports are detected when the module name is still referenced
        code = """
from numpy import *
import numpy
def func(x):
    return numpy.array(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_star_import_bare_name_not_detected(self):
        # Bare names from star imports are not tracked (limitation)
        code = """
from numpy import *
def func(x):
    return array(x)
"""
        # This is a known limitation - star import names aren't resolved
        assert is_numerical_code(code, "func") is False

    def test_star_import_math_bare_name_not_detected(self):
        # Same limitation applies to math
        code = """
from math import *
def func(x):
    return sqrt(x)
"""
        # Known limitation
        assert is_numerical_code(code, "func") is False


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestNestedUsage:
    """Test nested numerical library usage patterns (with numba available)."""

    def test_numpy_in_lambda(self):
        code = """
import numpy as np
def func():
    f = lambda x: np.sum(x)
    return f
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_in_list_comprehension(self):
        code = """
import numpy as np
def func(arrays):
    return [np.mean(arr) for arr in arrays]
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_in_conditional(self):
        code = """
import numpy as np
def func(x, use_numpy=True):
    if use_numpy:
        return np.sum(x)
    return sum(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_numpy_in_try_except(self):
        code = """
import numpy as np
def func(x):
    try:
        return np.sum(x)
    except Exception:
        return 0
"""
        assert is_numerical_code(code, "func") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestMultipleLibraries:
    """Test code using multiple numerical libraries (with numba available)."""

    def test_numpy_and_torch(self):
        code = """
import numpy as np
import torch
def func(x):
    arr = np.array(x)
    return torch.from_numpy(arr)
"""
        assert is_numerical_code(code, "func") is True

    def test_scipy_and_numpy(self):
        code = """
import numpy as np
from scipy import stats
def analyze(data):
    arr = np.array(data)
    return stats.describe(arr)
"""
        assert is_numerical_code(code, "analyze") is True


@patch("codeflash.code_utils.code_extractor.has_numba", True)
class TestQualifiedNames:
    """Test various qualified name patterns (with numba available)."""

    def test_simple_function_name(self):
        code = """
import numpy as np
def my_func():
    return np.array([1])
"""
        assert is_numerical_code(code, "my_func") is True

    def test_class_dot_method(self):
        code = """
import numpy as np
class MyClass:
    def my_method(self):
        return np.sum([1, 2])
"""
        assert is_numerical_code(code, "MyClass.my_method") is True

    def test_invalid_qualified_name_too_deep(self):
        code = """
import numpy as np
class Outer:
    class Inner:
        def method(self):
            return np.sum([1])
"""
        # Nested classes are not supported
        assert is_numerical_code(code, "Outer.Inner.method") is False

    def test_method_in_wrong_class(self):
        code = """
import numpy as np
class ClassA:
    def method(self):
        return np.sum([1])
class ClassB:
    def method(self):
        return 1
"""
        assert is_numerical_code(code, "ClassA.method") is True
        assert is_numerical_code(code, "ClassB.method") is False


@patch("codeflash.code_utils.code_extractor.has_numba", False)
class TestNumbaNotAvailable:
    """Test behavior when numba is NOT available in the environment.

    When numba is not installed, code using only math/numpy/scipy should return False,
    since numba is required to optimize such code. Code using torch/jax/tensorflow/numba
    should still return True as these libraries don't require numba for optimization.
    """

    def test_numpy_returns_false_without_numba(self):
        """Numpy usage should return False when numba is not available."""
        code = """
import numpy as np
def process_data(x):
    return np.sum(x)
"""
        assert is_numerical_code(code, "process_data") is False

    def test_scipy_returns_false_without_numba(self):
        """Scipy usage should return False when numba is not available."""
        code = """
from scipy import stats
def analyze(data):
    return stats.describe(data)
"""
        assert is_numerical_code(code, "analyze") is False

    def test_math_returns_false_without_numba(self):
        """Math usage should return False when numba is not available."""
        code = """
import math
def calculate(x):
    return math.sqrt(x)
"""
        assert is_numerical_code(code, "calculate") is False

    def test_torch_returns_true_without_numba(self):
        """Torch usage should return True even when numba is not available."""
        code = """
import torch
def train_model(model):
    return torch.nn.functional.relu(model)
"""
        assert is_numerical_code(code, "train_model") is True

    def test_jax_returns_true_without_numba(self):
        """JAX usage should return True even when numba is not available."""
        code = """
import jax
def func(x):
    return jax.grad(x)
"""
        assert is_numerical_code(code, "func") is True

    def test_tensorflow_returns_true_without_numba(self):
        """TensorFlow usage should return True even when numba is not available."""
        code = """
import tensorflow as tf
def build_model():
    return tf.keras.Sequential()
"""
        assert is_numerical_code(code, "build_model") is True

    def test_numba_import_returns_true_without_numba(self):
        """Code that imports numba should return True (numba is in modules_used)."""
        code = """
from numba import jit
@jit
def fast_func(x):
    return x * 2
"""
        assert is_numerical_code(code, "fast_func") is True

    def test_numpy_and_torch_returns_true_without_numba(self):
        """Mixed numpy+torch usage should return True since torch doesn't require numba."""
        code = """
import numpy as np
import torch
def func(x):
    arr = np.array(x)
    return torch.from_numpy(arr)
"""
        # Returns True because torch is in modules_used and torch doesn't require numba
        assert is_numerical_code(code, "func") is True

    def test_numpy_and_jax_returns_true_without_numba(self):
        """Mixed numpy+jax usage should return True since jax doesn't require numba."""
        code = """
import numpy as np
import jax.numpy as jnp
def func(x):
    arr = np.array(x)
    return jnp.sum(arr)
"""
        # Returns True because jax is in modules_used and jax doesn't require numba
        assert is_numerical_code(code, "func") is True

    def test_scipy_and_tensorflow_returns_true_without_numba(self):
        """Mixed scipy+tensorflow usage should return True since tensorflow doesn't require numba."""
        code = """
from scipy import stats
import tensorflow as tf
def analyze_and_build(data):
    result = stats.describe(data)
    return tf.keras.Sequential()
"""
        # Returns True because tensorflow is in modules_used and doesn't require numba
        assert is_numerical_code(code, "analyze_and_build") is True

    def test_numpy_submodule_returns_false_without_numba(self):
        """Numpy submodule usage should return False when numba is not available."""
        code = """
import numpy.linalg as la
def func(x):
    return la.norm(x)
"""
        assert is_numerical_code(code, "func") is False

    def test_math_from_import_returns_false_without_numba(self):
        """Math from import should return False when numba is not available."""
        code = """
from math import sqrt, sin, cos
def calculate(x):
    return sqrt(sin(x) ** 2 + cos(x) ** 2)
"""
        assert is_numerical_code(code, "calculate") is False
