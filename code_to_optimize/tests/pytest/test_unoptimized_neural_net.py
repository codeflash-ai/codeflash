"""
Unit tests for unoptimized PyTorch neural network implementation.

Tests run on CUDA devices only.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available - tests require CUDA"
)

from code_to_optimize.unoptimized_neural_net import UnoptimizedNeuralNet


def get_available_devices():
    """Return list of available PyTorch devices for testing."""
    return ["cuda"]


DEVICES = get_available_devices()


def get_dtype(device):
    """Get the appropriate dtype for a device."""
    return torch.float32


FIXED_INPUT_SIZE = 784
FIXED_HIDDEN_SIZE = 128
FIXED_NUM_CLASSES = 10
FIXED_BATCH_SIZE = 4


class TestUnoptimizedNeuralNet:
    """Tests for the UnoptimizedNeuralNet class.

    All tests use a single fixed shape: (4, 784)
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_shape(self, device):
        """Test that output shape is correct."""
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = model.forward(x)
        assert output.shape == (FIXED_BATCH_SIZE, FIXED_NUM_CLASSES)

    @pytest.mark.parametrize("device", DEVICES)
    def test_softmax_normalization(self, device):
        """Test that output probabilities sum to 1."""
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = model.forward(x)
        sums = output.sum(dim=1)
        np.testing.assert_array_almost_equal(sums.cpu().numpy(), np.ones(FIXED_BATCH_SIZE), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_range(self, device):
        """Test that output values are in valid probability range [0, 1]."""
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = model.forward(x)

        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_zeros_input(self, device):
        """Test with zero input."""
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        x = torch.zeros(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = model.forward(x)
        assert output.shape == (FIXED_BATCH_SIZE, FIXED_NUM_CLASSES)
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize("device", DEVICES)
    def test_deterministic_output(self, device):
        """Test that same input produces same output."""
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        model.eval()

        torch.manual_seed(100)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)

        output1 = model.forward(x)
        output2 = model.forward(x)

        np.testing.assert_array_almost_equal(
            output1.detach().cpu().numpy(),
            output2.detach().cpu().numpy(),
            decimal=6
            )

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_requires_grad_false(self, device):
        """Test that output has requires_grad=False."""
        torch.manual_seed(42)
        model = UnoptimizedNeuralNet(
            input_size=FIXED_INPUT_SIZE,
            hidden_size=FIXED_HIDDEN_SIZE,
            num_classes=FIXED_NUM_CLASSES
        )
        model = model.to(device)
        x = torch.randn(FIXED_BATCH_SIZE, FIXED_INPUT_SIZE, dtype=get_dtype(device), device=device, requires_grad=False)
        output = model.forward(x)
        assert output.requires_grad is False
