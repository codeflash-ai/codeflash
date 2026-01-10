import pytest
import torch

from code_to_optimize.discrete_riccati import _gridmake2_torch


class TestGridmake2TorchCPU:
    """Tests for _gridmake2_torch with CPU tensors."""

    def test_both_1d_simple(self):
        """Test with two simple 1D tensors."""
        x1 = torch.tensor([1, 2, 3])
        x2 = torch.tensor([10, 20])

        result = _gridmake2_torch(x1, x2)

        # Expected: x1 tiled x2.shape[0] times, x2 repeat_interleaved x1.shape[0]
        # x1 tiled: [1, 2, 3, 1, 2, 3]
        # x2 repeated: [10, 10, 10, 20, 20, 20]
        expected = torch.tensor([
            [1, 10],
            [2, 10],
            [3, 10],
            [1, 20],
            [2, 20],
            [3, 20],
        ])
        assert torch.equal(result, expected)

    def test_both_1d_single_element(self):
        """Test with single element tensors."""
        x1 = torch.tensor([5])
        x2 = torch.tensor([10])

        result = _gridmake2_torch(x1, x2)

        expected = torch.tensor([[5, 10]])
        assert torch.equal(result, expected)

    def test_both_1d_float_tensors(self):
        """Test with float tensors."""
        x1 = torch.tensor([1.5, 2.5])
        x2 = torch.tensor([0.1, 0.2, 0.3])

        result = _gridmake2_torch(x1, x2)

        assert result.shape == (6, 2)
        assert result.dtype == torch.float32

    def test_2d_and_1d_simple(self):
        """Test with 2D x1 and 1D x2."""
        x1 = torch.tensor([[1, 2], [3, 4]])
        x2 = torch.tensor([10, 20])

        result = _gridmake2_torch(x1, x2)

        # x1 tiled along first dim: [[1, 2], [3, 4], [1, 2], [3, 4]]
        # x2 repeated: [10, 10, 20, 20]
        # column_stack: [[1, 2, 10], [3, 4, 10], [1, 2, 20], [3, 4, 20]]
        expected = torch.tensor([
            [1, 2, 10],
            [3, 4, 10],
            [1, 2, 20],
            [3, 4, 20],
        ])
        assert torch.equal(result, expected)

    def test_2d_and_1d_single_column(self):
        """Test with 2D x1 having a single column and 1D x2."""
        x1 = torch.tensor([[1], [2], [3]])
        x2 = torch.tensor([10, 20])

        result = _gridmake2_torch(x1, x2)

        expected = torch.tensor([
            [1, 10],
            [2, 10],
            [3, 10],
            [1, 20],
            [2, 20],
            [3, 20],
        ])
        assert torch.equal(result, expected)

    def test_output_shape_1d_1d(self):
        """Test output shape for two 1D tensors."""
        x1 = torch.tensor([1, 2, 3, 4, 5])
        x2 = torch.tensor([10, 20, 30])

        result = _gridmake2_torch(x1, x2)

        # Shape should be (len(x1) * len(x2), 2)
        assert result.shape == (15, 2)

    def test_output_shape_2d_1d(self):
        """Test output shape for 2D and 1D tensors."""
        x1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
        x2 = torch.tensor([10, 20, 30, 40])  # Shape (4,)

        result = _gridmake2_torch(x1, x2)

        # Shape should be (2 * 4, 3 + 1) = (8, 4)
        assert result.shape == (8, 4)

    def test_not_implemented_for_2d_2d(self):
        """Test that NotImplementedError is raised for two 2D tensors."""
        x1 = torch.tensor([[1, 2], [3, 4]])
        x2 = torch.tensor([[10, 20], [30, 40]])

        with pytest.raises(NotImplementedError, match="Come back here"):
            _gridmake2_torch(x1, x2)

    def test_not_implemented_for_1d_2d(self):
        """Test that NotImplementedError is raised for 1D and 2D tensors."""
        x1 = torch.tensor([1, 2, 3])
        x2 = torch.tensor([[10, 20], [30, 40]])

        with pytest.raises(NotImplementedError, match="Come back here"):
            _gridmake2_torch(x1, x2)

    def test_preserves_dtype_int(self):
        """Test that integer dtype is preserved."""
        x1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        x2 = torch.tensor([10, 20], dtype=torch.int32)

        result = _gridmake2_torch(x1, x2)

        assert result.dtype == torch.int32

    def test_preserves_dtype_float64(self):
        """Test that float64 dtype is preserved."""
        x1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        x2 = torch.tensor([10.0, 20.0], dtype=torch.float64)

        result = _gridmake2_torch(x1, x2)

        assert result.dtype == torch.float64

    def test_large_tensors(self):
        """Test with larger tensors."""
        x1 = torch.arange(100)
        x2 = torch.arange(50)

        result = _gridmake2_torch(x1, x2)

        assert result.shape == (5000, 2)
        # Verify first and last elements
        assert result[0, 0] == 0 and result[0, 1] == 0
        assert result[-1, 0] == 99 and result[-1, 1] == 49


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGridmake2TorchCUDA:
    """Tests for _gridmake2_torch with CUDA tensors."""

    def test_both_1d_simple_cuda(self):
        """Test with two simple 1D CUDA tensors."""
        x1 = torch.tensor([1, 2, 3], device="cuda")
        x2 = torch.tensor([10, 20], device="cuda")

        result = _gridmake2_torch(x1, x2)

        expected = torch.tensor([
            [1, 10],
            [2, 10],
            [3, 10],
            [1, 20],
            [2, 20],
            [3, 20],
        ], device="cuda")
        assert result.device.type == "cuda"
        assert torch.equal(result, expected)

    def test_both_1d_matches_cpu(self):
        """Test that CUDA version matches CPU version."""
        x1_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x2_cpu = torch.tensor([10.0, 20.0, 30.0])

        x1_cuda = x1_cpu.cuda()
        x2_cuda = x2_cpu.cuda()

        result_cpu = _gridmake2_torch(x1_cpu, x2_cpu)
        result_cuda = _gridmake2_torch(x1_cuda, x2_cuda)

        assert result_cuda.device.type == "cuda"
        torch.testing.assert_close(result_cpu, result_cuda.cpu())

    def test_2d_and_1d_cuda(self):
        """Test with 2D x1 and 1D x2 on CUDA."""
        x1 = torch.tensor([[1, 2], [3, 4]], device="cuda")
        x2 = torch.tensor([10, 20], device="cuda")

        result = _gridmake2_torch(x1, x2)

        expected = torch.tensor([
            [1, 2, 10],
            [3, 4, 10],
            [1, 2, 20],
            [3, 4, 20],
        ], device="cuda")
        assert result.device.type == "cuda"
        assert torch.equal(result, expected)

    def test_2d_and_1d_matches_cpu(self):
        """Test that CUDA version matches CPU version for 2D, 1D inputs."""
        x1_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x2_cpu = torch.tensor([10.0, 20.0])

        x1_cuda = x1_cpu.cuda()
        x2_cuda = x2_cpu.cuda()

        result_cpu = _gridmake2_torch(x1_cpu, x2_cpu)
        result_cuda = _gridmake2_torch(x1_cuda, x2_cuda)

        assert result_cuda.device.type == "cuda"
        torch.testing.assert_close(result_cpu, result_cuda.cpu())

    def test_output_stays_on_cuda(self):
        """Test that output tensor stays on CUDA device."""
        x1 = torch.tensor([1, 2, 3], device="cuda")
        x2 = torch.tensor([10, 20], device="cuda")

        result = _gridmake2_torch(x1, x2)

        assert result.is_cuda

    def test_preserves_dtype_float32_cuda(self):
        """Test that float32 dtype is preserved on CUDA."""
        x1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="cuda")
        x2 = torch.tensor([10.0, 20.0], dtype=torch.float32, device="cuda")

        result = _gridmake2_torch(x1, x2)

        assert result.dtype == torch.float32
        assert result.device.type == "cuda"

    def test_preserves_dtype_float64_cuda(self):
        """Test that float64 dtype is preserved on CUDA."""
        x1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
        x2 = torch.tensor([10.0, 20.0], dtype=torch.float64, device="cuda")

        result = _gridmake2_torch(x1, x2)

        assert result.dtype == torch.float64
        assert result.device.type == "cuda"

    def test_large_tensors_cuda(self):
        """Test with larger tensors on CUDA."""
        x1 = torch.arange(100, device="cuda")
        x2 = torch.arange(50, device="cuda")

        result = _gridmake2_torch(x1, x2)

        assert result.shape == (5000, 2)
        assert result.device.type == "cuda"
        # Verify first and last elements
        assert result[0, 0].item() == 0 and result[0, 1].item() == 0
        assert result[-1, 0].item() == 99 and result[-1, 1].item() == 49

    def test_not_implemented_for_2d_2d_cuda(self):
        """Test that NotImplementedError is raised for two 2D CUDA tensors."""
        x1 = torch.tensor([[1, 2], [3, 4]], device="cuda")
        x2 = torch.tensor([[10, 20], [30, 40]], device="cuda")

        with pytest.raises(NotImplementedError, match="Come back here"):
            _gridmake2_torch(x1, x2)

