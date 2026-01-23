"""
Unit tests for PyTorch implementations of JIT-suitable functions.

Tests run on CUDA devices only.
"""

import numpy as np
import pytest

import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available - tests require CUDA"
)

from code_to_optimize.sample_code import (
    leapfrog_integration_torch,
    longest_increasing_subsequence_length_torch,
    tridiagonal_solve_torch,
)


def get_available_devices():
    """Return list of available PyTorch devices for testing."""
    # Only test on CUDA devices
    return ["cuda"]


DEVICES = get_available_devices()


def get_dtype(device):
    """Get the appropriate dtype for a device."""
    return torch.float64


def to_device(arr, device):
    """Move a tensor to the specified device."""
    dtype = get_dtype(device)
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr).to(dtype)
    return arr.to(device)


class TestTridiagonalSolveTorch:
    """Tests for the PyTorch tridiagonal_solve function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_simple_system(self, device):
        """Test a simple 3x3 tridiagonal system with known solution."""
        a = torch.tensor([-1.0, -1.0], dtype=get_dtype(device), device=device)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=get_dtype(device), device=device)
        c = torch.tensor([-1.0, -1.0], dtype=get_dtype(device), device=device)
        d = torch.tensor([1.0, 0.0, 1.0], dtype=get_dtype(device), device=device)

        x = tridiagonal_solve_torch(a, b, c, d)

        # Verify solution by multiplying back
        result = torch.zeros(3, dtype=get_dtype(device), device=device)
        result[0] = b[0] * x[0] + c[0] * x[1]
        result[1] = a[0] * x[0] + b[1] * x[1] + c[1] * x[2]
        result[2] = a[1] * x[1] + b[2] * x[2]

        np.testing.assert_array_almost_equal(result.cpu().numpy(), d.cpu().numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_diagonal_system(self, device):
        """Test a purely diagonal system."""
        a = torch.tensor([0.0, 0.0], dtype=get_dtype(device), device=device)
        b = torch.tensor([2.0, 3.0, 4.0], dtype=get_dtype(device), device=device)
        c = torch.tensor([0.0, 0.0], dtype=get_dtype(device), device=device)
        d = torch.tensor([4.0, 9.0, 16.0], dtype=get_dtype(device), device=device)

        x = tridiagonal_solve_torch(a, b, c, d)

        expected = torch.tensor([2.0, 3.0, 4.0], dtype=get_dtype(device))
        np.testing.assert_array_almost_equal(x.cpu().numpy(), expected.numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_larger_system(self, device):
        """Test a larger tridiagonal system."""
        n = 100
        a = -torch.ones(n - 1, dtype=get_dtype(device), device=device)
        b = 2.0 * torch.ones(n, dtype=get_dtype(device), device=device)
        c = -torch.ones(n - 1, dtype=get_dtype(device), device=device)
        d = torch.zeros(n, dtype=get_dtype(device), device=device)
        d[0] = 1.0
        d[-1] = 1.0

        x = tridiagonal_solve_torch(a, b, c, d)

        # Verify by reconstructing Ax
        result = torch.zeros(n, dtype=get_dtype(device), device=device)
        result[0] = b[0] * x[0] + c[0] * x[1]
        for i in range(1, n - 1):
            result[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
        result[-1] = a[-1] * x[-2] + b[-1] * x[-1]

        np.testing.assert_array_almost_equal(result.cpu().numpy(), d.cpu().numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_element_system(self, device):
        """Test minimal 2x2 tridiagonal system."""
        a = torch.tensor([1.0], dtype=get_dtype(device), device=device)
        b = torch.tensor([4.0, 4.0], dtype=get_dtype(device), device=device)
        c = torch.tensor([1.0], dtype=get_dtype(device), device=device)
        d = torch.tensor([5.0, 5.0], dtype=get_dtype(device), device=device)

        x = tridiagonal_solve_torch(a, b, c, d)

        result = torch.tensor([
            b[0] * x[0] + c[0] * x[1],
            a[0] * x[0] + b[1] * x[1]
        ], device=device)
        np.testing.assert_array_almost_equal(result.cpu().numpy(), d.cpu().numpy(), decimal=5)


class TestLeapfrogIntegrationTorch:
    """Tests for the PyTorch leapfrog_integration function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_stationary_particle(self, device):
        """A single particle with no velocity should remain stationary."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        velocities = torch.tensor([[0.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        masses = torch.tensor([1.0], dtype=get_dtype(device), device=device)

        final_pos, final_vel = leapfrog_integration_torch(
            positions, velocities, masses, dt=0.01, n_steps=100
        )

        np.testing.assert_array_almost_equal(final_pos.cpu().numpy(), positions.cpu().numpy(), decimal=5)
        np.testing.assert_array_almost_equal(final_vel.cpu().numpy(), velocities.cpu().numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_moving_particle(self, device):
        """A single moving particle should move in a straight line."""
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        velocities = torch.tensor([[1.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        masses = torch.tensor([1.0], dtype=get_dtype(device), device=device)

        dt = 0.01
        n_steps = 100

        final_pos, final_vel = leapfrog_integration_torch(
            positions, velocities, masses, dt=dt, n_steps=n_steps
        )

        np.testing.assert_array_almost_equal(final_vel.cpu().numpy(), velocities.cpu().numpy(), decimal=5)
        expected_pos = torch.tensor([[dt * n_steps, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(final_pos.cpu().numpy(), expected_pos.numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_particles_approach(self, device):
        """Two particles should attract each other gravitationally."""
        positions = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        velocities = torch.zeros((2, 3), dtype=get_dtype(device), device=device)
        masses = torch.tensor([1.0, 1.0], dtype=get_dtype(device), device=device)

        final_pos, _ = leapfrog_integration_torch(
            positions, velocities, masses, dt=0.01, n_steps=50, softening=0.1
        )

        initial_distance = 2.0
        final_distance = torch.linalg.norm(final_pos[1] - final_pos[0]).item()
        assert final_distance < initial_distance

    @pytest.mark.parametrize("device", DEVICES)
    def test_momentum_conservation(self, device):
        """Total momentum should be approximately conserved."""
        np.random.seed(42)
        n_particles = 5
        positions = torch.tensor(np.random.randn(n_particles, 3), dtype=get_dtype(device), device=device)
        velocities = torch.tensor(np.random.randn(n_particles, 3), dtype=get_dtype(device), device=device)
        masses = torch.tensor(np.abs(np.random.randn(n_particles)) + 0.1, dtype=get_dtype(device), device=device)

        initial_momentum = torch.sum(masses[:, None] * velocities, dim=0)

        final_pos, final_vel = leapfrog_integration_torch(
            positions, velocities, masses, dt=0.001, n_steps=100, softening=0.5
        )

        final_momentum = torch.sum(masses[:, None] * final_vel, dim=0)

        np.testing.assert_array_almost_equal(
            initial_momentum.cpu().numpy(), final_momentum.cpu().numpy(), decimal=4
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_does_not_modify_input(self, device):
        """Input arrays should not be modified."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        velocities = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=get_dtype(device), device=device)
        masses = torch.tensor([1.0, 1.0], dtype=get_dtype(device), device=device)

        pos_copy = positions.clone()
        vel_copy = velocities.clone()

        leapfrog_integration_torch(positions, velocities, masses, dt=0.01, n_steps=10)

        np.testing.assert_array_equal(positions.cpu().numpy(), pos_copy.cpu().numpy())
        np.testing.assert_array_equal(velocities.cpu().numpy(), vel_copy.cpu().numpy())


class TestLongestIncreasingSubsequenceLengthTorch:
    """Tests for the PyTorch longest_increasing_subsequence_length function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_array(self, device):
        """Empty array should return 0."""
        arr = torch.tensor([], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_element(self, device):
        """Single element array should return 1."""
        arr = torch.tensor([5.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_increasing(self, device):
        """Strictly increasing array - LIS is the whole array."""
        arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_decreasing(self, device):
        """Strictly decreasing array - LIS is length 1."""
        arr = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_classic_example(self, device):
        """Classic LIS example."""
        arr = torch.tensor([10.0, 9.0, 2.0, 5.0, 3.0, 7.0, 101.0, 18.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 4

    @pytest.mark.parametrize("device", DEVICES)
    def test_all_same_elements(self, device):
        """All same elements - LIS is length 1."""
        arr = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_alternating_sequence(self, device):
        """Alternating high-low sequence."""
        arr = torch.tensor([1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_elements_increasing(self, device):
        """Two elements in increasing order."""
        arr = torch.tensor([1.0, 2.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 2

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_elements_decreasing(self, device):
        """Two elements in decreasing order."""
        arr = torch.tensor([2.0, 1.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_longer_sequence(self, device):
        """Test with a longer sequence."""
        arr = torch.tensor([0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 6

    @pytest.mark.parametrize("device", DEVICES)
    def test_negative_numbers(self, device):
        """Test with negative numbers."""
        arr = torch.tensor([-5.0, -2.0, -8.0, -1.0, -6.0, 0.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 4

    @pytest.mark.parametrize("device", DEVICES)
    def test_float_values(self, device):
        """Test with floating point values."""
        arr = torch.tensor([1.5, 2.3, 1.8, 3.1, 2.9, 4.0], dtype=get_dtype(device), device=device)
        assert longest_increasing_subsequence_length_torch(arr) == 4