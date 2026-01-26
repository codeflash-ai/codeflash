"""
Unit tests for complex_activation function.

Tests run on CUDA device with a single tensor shape.
"""

import pytest
import torch

from code_to_optimize.complex_activation import complex_activation


@pytest.fixture
def cuda_device():
    """Return CUDA device, skip if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def input_tensor(cuda_device):
    """Create a fixed-shape input tensor on CUDA."""
    torch.manual_seed(42)
    return torch.randn(32, 64, device=cuda_device, dtype=torch.float32)


class TestComplexActivation:
    """Tests for the complex_activation function."""

    def test_output_shape(self, input_tensor):
        """Test that output shape matches input shape."""
        result = complex_activation(input_tensor)
        assert result.shape == input_tensor.shape

    def test_output_dtype(self, input_tensor):
        """Test that output dtype matches input dtype."""
        result = complex_activation(input_tensor)
        assert result.dtype == input_tensor.dtype

    def test_output_device(self, input_tensor, cuda_device):
        """Test that output is on the same device as input."""
        result = complex_activation(input_tensor)
        assert result.device.type == cuda_device.type

    def test_deterministic(self, input_tensor):
        """Test that the function produces deterministic results."""
        result1 = complex_activation(input_tensor.clone())
        result2 = complex_activation(input_tensor.clone())
        torch.testing.assert_close(result1, result2)

    def test_output_is_finite(self, input_tensor):
        """Test that output contains no NaN or Inf values."""
        result = complex_activation(input_tensor)
        assert torch.isfinite(result).all()

    def test_output_bounded(self, input_tensor):
        """Test that output values are bounded (activation should not explode)."""
        result = complex_activation(input_tensor)
        assert result.abs().max() < 10.0

    def test_zero_input(self, cuda_device):
        """Test behavior with zero input."""
        x = torch.zeros(32, 64, device=cuda_device, dtype=torch.float32)
        result = complex_activation(x)
        assert torch.isfinite(result).all()
        assert result.shape == x.shape

    def test_positive_input(self, cuda_device):
        """Test behavior with all positive inputs."""
        x = torch.abs(torch.randn(32, 64, device=cuda_device, dtype=torch.float32)) + 0.1
        result = complex_activation(x)
        assert torch.isfinite(result).all()

    def test_negative_input(self, cuda_device):
        """Test behavior with all negative inputs."""
        x = -torch.abs(torch.randn(32, 64, device=cuda_device, dtype=torch.float32)) - 0.1
        result = complex_activation(x)
        assert torch.isfinite(result).all()

    def test_gradient_flow(self, cuda_device):
        """Test that gradients can flow through the activation."""
        x = torch.randn(32, 64, device=cuda_device, dtype=torch.float32, requires_grad=True)
        result = complex_activation(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()