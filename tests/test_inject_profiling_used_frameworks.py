"""Unit tests for inject_profiling_into_existing_test with different used_frameworks values.

These tests verify that the wrapper function is correctly generated with GPU device
synchronization code for different framework imports (torch, tensorflow, jax).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codeflash.code_utils.instrument_existing_tests import (
    detect_frameworks_from_code,
    inject_profiling_into_existing_test,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodePosition, TestingMode


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
        assert result == {}

    def test_torch_import(self) -> None:
        """Test detection with torch import."""
        code = """import torch
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"torch": "torch"}

    def test_torch_aliased_import(self) -> None:
        """Test detection with torch imported as alias."""
        code = """import torch as th
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"torch": "th"}

    def test_torch_submodule_import(self) -> None:
        """Test detection with torch submodule import (from torch import nn)."""
        code = """from torch import nn
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"torch": "torch"}

    def test_torch_dotted_import(self) -> None:
        """Test detection with torch.cuda or torch.nn import."""
        code = """import torch.cuda
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"torch": "torch"}

    def test_tensorflow_import(self) -> None:
        """Test detection with tensorflow import."""
        code = """import tensorflow
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"tensorflow": "tensorflow"}

    def test_tensorflow_aliased_import(self) -> None:
        """Test detection with tensorflow imported as alias."""
        code = """import tensorflow as tf
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"tensorflow": "tf"}

    def test_tensorflow_submodule_import(self) -> None:
        """Test detection with tensorflow submodule import."""
        code = """from tensorflow import keras
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"tensorflow": "tensorflow"}

    def test_jax_import(self) -> None:
        """Test detection with jax import."""
        code = """import jax
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"jax": "jax"}

    def test_jax_aliased_import(self) -> None:
        """Test detection with jax imported as alias."""
        code = """import jax as jnp
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"jax": "jnp"}

    def test_jax_submodule_import(self) -> None:
        """Test detection with jax submodule import."""
        code = """from jax import numpy as jnp
from mymodule import my_function

def test_something():
    pass
"""
        result = detect_frameworks_from_code(code)
        assert result == {"jax": "jax"}

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
        assert result == {"torch": "torch", "tensorflow": "tensorflow", "jax": "jax"}

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
        assert result == {"torch": "th", "tensorflow": "tf", "jax": "jnp"}

    def test_syntax_error_returns_empty(self) -> None:
        """Test that syntax errors return empty dict."""
        code = """this is not valid python code !!!"""
        result = detect_frameworks_from_code(code)
        assert result == {}


# ============================================================================
# Tests for inject_profiling_into_existing_test - No Frameworks
# ============================================================================


class TestInjectProfilingNoFrameworks:
    """Tests for inject_profiling_into_existing_test with no GPU frameworks."""

    def test_no_frameworks_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with no GPU framework imports in BEHAVIOR mode."""
        code = """from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(4, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify no GPU sync code is present
        assert "_codeflash_should_sync_cuda" not in result
        assert "_codeflash_should_sync_mps" not in result
        assert "_codeflash_should_sync_jax" not in result
        assert "_codeflash_should_sync_tf" not in result
        assert "torch.cuda.synchronize()" not in result
        assert "torch.mps.synchronize()" not in result
        assert "jax.block_until_ready" not in result
        assert "tensorflow.test.experimental.sync_devices()" not in result

    def test_no_frameworks_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with no GPU framework imports in PERFORMANCE mode."""
        code = """from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(4, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        assert success
        assert result is not None
        # Verify no GPU sync code is present
        assert "_codeflash_should_sync_cuda" not in result
        assert "_codeflash_should_sync_mps" not in result
        assert "_codeflash_should_sync_jax" not in result
        assert "_codeflash_should_sync_tf" not in result


# ============================================================================
# Tests for inject_profiling_into_existing_test - PyTorch
# ============================================================================


class TestInjectProfilingTorch:
    """Tests for inject_profiling_into_existing_test with PyTorch."""

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code is present
        assert "_codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()" in result
        assert "_codeflash_should_sync_mps" in result
        assert "torch.cuda.synchronize()" in result
        assert "torch.mps.synchronize()" in result
        # Verify other frameworks are not present
        assert "_codeflash_should_sync_jax" not in result
        assert "_codeflash_should_sync_tf" not in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code uses the alias
        assert "_codeflash_should_sync_cuda = th.cuda.is_available() and th.cuda.is_initialized()" in result
        assert "th.cuda.synchronize()" in result
        assert "th.mps.synchronize()" in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code uses 'torch' (the default name)
        assert "_codeflash_should_sync_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()" in result
        assert "torch.cuda.synchronize()" in result
        # Verify torch import was added
        assert "import torch" in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "torch.cuda.synchronize()" in result


# ============================================================================
# Tests for inject_profiling_into_existing_test - TensorFlow
# ============================================================================


class TestInjectProfilingTensorFlow:
    """Tests for inject_profiling_into_existing_test with TensorFlow."""

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify TensorFlow sync code is present
        assert "_codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        # Verify other frameworks are not present
        assert "_codeflash_should_sync_cuda" not in result
        assert "_codeflash_should_sync_jax" not in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify TensorFlow sync code uses the alias
        assert "_codeflash_should_sync_tf = hasattr(tf.test.experimental, 'sync_devices')" in result
        assert "tf.test.experimental.sync_devices()" in result

    def test_tensorflow_submodule_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with TensorFlow submodule import in BEHAVIOR mode."""
        code = """from tensorflow import keras
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify TensorFlow sync code uses 'tensorflow' (the default name)
        assert "_codeflash_should_sync_tf = hasattr(tensorflow.test.experimental, 'sync_devices')" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        # Verify tensorflow import was added
        assert "import tensorflow" in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        assert success
        assert result is not None
        # Verify TensorFlow sync code is present
        assert "_codeflash_should_sync_tf" in result
        assert "tensorflow.test.experimental.sync_devices()" in result


# ============================================================================
# Tests for inject_profiling_into_existing_test - JAX
# ============================================================================


class TestInjectProfilingJax:
    """Tests for inject_profiling_into_existing_test with JAX."""

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify JAX sync code is present
        assert "_codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')" in result
        assert "jax.block_until_ready(return_value)" in result
        # Verify other frameworks are not present
        assert "_codeflash_should_sync_cuda" not in result
        assert "_codeflash_should_sync_tf" not in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify JAX sync code uses the alias
        assert "_codeflash_should_sync_jax = hasattr(jnp, 'block_until_ready')" in result
        assert "jnp.block_until_ready(return_value)" in result

    def test_jax_submodule_import_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with JAX submodule import in BEHAVIOR mode."""
        code = """from jax import numpy as jnp
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify JAX sync code uses 'jax' (the default name)
        assert "_codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')" in result
        assert "jax.block_until_ready(return_value)" in result
        # Verify jax import was added
        assert "import jax" in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        assert success
        assert result is not None
        # Verify JAX sync code is present
        assert "_codeflash_should_sync_jax" in result
        assert "jax.block_until_ready(return_value)" in result


# ============================================================================
# Tests for inject_profiling_into_existing_test - Multiple Frameworks
# ============================================================================


class TestInjectProfilingMultipleFrameworks:
    """Tests for inject_profiling_into_existing_test with multiple GPU frameworks."""

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(6, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify both PyTorch and TensorFlow sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "_codeflash_should_sync_mps" in result
        assert "_codeflash_should_sync_tf" in result
        assert "torch.cuda.synchronize()" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        # Verify JAX is not present
        assert "_codeflash_should_sync_jax" not in result

    def test_torch_and_jax_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with both PyTorch and JAX imports in BEHAVIOR mode."""
        code = """import torch
import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(6, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify both PyTorch and JAX sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "_codeflash_should_sync_jax" in result
        assert "torch.cuda.synchronize()" in result
        assert "jax.block_until_ready(return_value)" in result
        # Verify TensorFlow is not present
        assert "_codeflash_should_sync_tf" not in result

    def test_tensorflow_and_jax_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with both TensorFlow and JAX imports in BEHAVIOR mode."""
        code = """import tensorflow
import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(6, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify both TensorFlow and JAX sync code is present
        assert "_codeflash_should_sync_tf" in result
        assert "_codeflash_should_sync_jax" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        assert "jax.block_until_ready(return_value)" in result
        # Verify PyTorch is not present
        assert "_codeflash_should_sync_cuda" not in result

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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify all three frameworks sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "_codeflash_should_sync_mps" in result
        assert "_codeflash_should_sync_tf" in result
        assert "_codeflash_should_sync_jax" in result
        assert "torch.cuda.synchronize()" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        assert "jax.block_until_ready(return_value)" in result

    def test_all_three_frameworks_aliased_behavior_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with all three frameworks imported as aliases in BEHAVIOR mode."""
        code = """import torch as th
import tensorflow as tf
import jax as jnp
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify all three frameworks sync code uses their aliases
        assert "th.cuda.is_available()" in result
        assert "th.cuda.synchronize()" in result
        assert "tf.test.experimental.sync_devices()" in result
        assert "jnp.block_until_ready(return_value)" in result

    def test_all_three_frameworks_performance_mode(self, tmp_path: Path) -> None:
        """Test instrumentation with all three frameworks in PERFORMANCE mode."""
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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.PERFORMANCE,
        )

        assert success
        assert result is not None
        # Verify all three frameworks sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "_codeflash_should_sync_tf" in result
        assert "_codeflash_should_sync_jax" in result
        assert "torch.cuda.synchronize()" in result
        assert "tensorflow.test.experimental.sync_devices()" in result
        assert "jax.block_until_ready(return_value)" in result


# ============================================================================
# Tests for inject_profiling_into_existing_test - Edge Cases
# ============================================================================


class TestInjectProfilingEdgeCases:
    """Edge case tests for inject_profiling_into_existing_test with used_frameworks."""

    def test_torch_dotted_import_detected(self, tmp_path: Path) -> None:
        """Test that torch.cuda import is correctly detected as torch."""
        code = """import torch.cuda
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code is present (uses 'torch' as the alias)
        assert "_codeflash_should_sync_cuda = torch.cuda.is_available()" in result
        assert "torch.cuda.synchronize()" in result

    def test_unittest_class_with_frameworks(self, tmp_path: Path) -> None:
        """Test instrumentation of unittest.TestCase class with GPU framework imports."""
        code = """import unittest
import torch
from mymodule import my_function


class TestMyFunction(unittest.TestCase):
    def test_basic(self):
        result = my_function(1, 2)
        self.assertEqual(result, 3)
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(8, 17)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "torch.cuda.synchronize()" in result

    def test_multiple_function_calls_with_frameworks(self, tmp_path: Path) -> None:
        """Test instrumentation with multiple function calls and GPU framework imports."""
        code = """import torch
from mymodule import my_function

def test_multiple_calls():
    result1 = my_function(1, 2)
    result2 = my_function(3, 4)
    assert result1 == 3
    assert result2 == 7
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 14), CodePosition(6, 14)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify PyTorch sync code is present
        assert "_codeflash_should_sync_cuda" in result
        assert "torch.cuda.synchronize()" in result

    def test_jax_no_pre_sync_only_post_sync(self, tmp_path: Path) -> None:
        """Test that JAX only has post-sync (block_until_ready on return value), not pre-sync."""
        code = """import jax
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # JAX should only appear in post-sync context (with return_value)
        # The pre-computed condition should be present
        assert "_codeflash_should_sync_jax = hasattr(jax, 'block_until_ready')" in result
        # The actual sync call should be on return_value (post-sync only)
        assert "jax.block_until_ready(return_value)" in result

    def test_precomputed_sync_conditions_before_gc_disable(self, tmp_path: Path) -> None:
        """Test that sync conditions are precomputed before gc.disable() call."""
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

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(7, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None

        # Find positions of key elements
        gc_disable_pos = result.find("gc.disable()")
        cuda_precompute_pos = result.find("_codeflash_should_sync_cuda =")
        mps_precompute_pos = result.find("_codeflash_should_sync_mps =")
        jax_precompute_pos = result.find("_codeflash_should_sync_jax =")
        tf_precompute_pos = result.find("_codeflash_should_sync_tf =")

        # All precomputed conditions should come before gc.disable()
        assert gc_disable_pos > 0
        assert cuda_precompute_pos > 0
        assert cuda_precompute_pos < gc_disable_pos
        assert mps_precompute_pos > 0
        assert mps_precompute_pos < gc_disable_pos
        assert jax_precompute_pos > 0
        assert jax_precompute_pos < gc_disable_pos
        assert tf_precompute_pos > 0
        assert tf_precompute_pos < gc_disable_pos

    def test_framework_imports_added_to_output(self, tmp_path: Path) -> None:
        """Test that framework imports are properly added to the output."""
        code = """from torch.nn import Linear
from mymodule import my_function

def test_my_function():
    result = my_function(1, 2)
    assert result == 3
"""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(code)

        func = FunctionToOptimize(
            function_name="my_function",
            parents=[],
            file_path=Path("mymodule.py"),
        )

        success, result = inject_profiling_into_existing_test(
            test_path=test_file,
            call_positions=[CodePosition(5, 13)],
            function_to_optimize=func,
            tests_project_root=tmp_path,
            mode=TestingMode.BEHAVIOR,
        )

        assert success
        assert result is not None
        # Verify that 'import torch' was added (needed for GPU sync code)
        assert "import torch" in result
        # The sync code should use 'torch' (not 'torch.nn')
        assert "_codeflash_should_sync_cuda = torch.cuda.is_available()" in result
