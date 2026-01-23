"""
Unit tests for TensorFlow implementations of JIT-suitable functions.

Tests run on CUDA devices only.
"""

import platform

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

pytestmark = pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"),
    reason="GPU not available - tests require CUDA"
)

from code_to_optimize.sample_code import (
    leapfrog_integration_tf,
    longest_increasing_subsequence_length_tf,
    tridiagonal_solve_tf,
)


def get_available_devices():
    """Return list of available TensorFlow devices for testing."""
    # Only test on CUDA devices
    return ["cuda"]


DEVICES = get_available_devices()


def run_on_device(func, device, *args, **kwargs):
    """Run a function on the specified device."""
    if device == "cpu":
        device_name = "/CPU:0"
    elif device in ("cuda", "metal"):
        device_name = "/GPU:0"
    else:
        device_name = "/CPU:0"

    with tf.device(device_name):
        return func(*args, **kwargs)


def to_tensor(arr, device, dtype=tf.float64):
    """Create a tensor on the specified device."""
    if device == "cpu":
        device_name = "/CPU:0"
    elif device in ("cuda", "metal"):
        device_name = "/GPU:0"
    else:
        device_name = "/CPU:0"

    with tf.device(device_name):
        return tf.constant(arr, dtype=dtype)


class TestTridiagonalSolveTf:
    """Tests for the TensorFlow tridiagonal_solve function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_simple_system(self, device):
        """Test a simple 3x3 tridiagonal system with known solution."""
        a = to_tensor([-1.0, -1.0], device)
        b = to_tensor([2.0, 2.0, 2.0], device)
        c = to_tensor([-1.0, -1.0], device)
        d = to_tensor([1.0, 0.0, 1.0], device)

        x = run_on_device(tridiagonal_solve_tf, device, a, b, c, d)

        # Verify solution by multiplying back
        result = np.zeros(3)
        x_np = x.numpy()
        b_np = b.numpy()
        c_np = c.numpy()
        a_np = a.numpy()
        result[0] = b_np[0] * x_np[0] + c_np[0] * x_np[1]
        result[1] = a_np[0] * x_np[0] + b_np[1] * x_np[1] + c_np[1] * x_np[2]
        result[2] = a_np[1] * x_np[1] + b_np[2] * x_np[2]

        np.testing.assert_array_almost_equal(result, d.numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_diagonal_system(self, device):
        """Test a purely diagonal system."""
        a = to_tensor([0.0, 0.0], device)
        b = to_tensor([2.0, 3.0, 4.0], device)
        c = to_tensor([0.0, 0.0], device)
        d = to_tensor([4.0, 9.0, 16.0], device)

        x = run_on_device(tridiagonal_solve_tf, device, a, b, c, d)

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(x.numpy(), expected, decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_larger_system(self, device):
        """Test a larger tridiagonal system."""
        n = 50
        a_np = -np.ones(n - 1)
        b_np = 2.0 * np.ones(n)
        c_np = -np.ones(n - 1)
        d_np = np.zeros(n)
        d_np[0] = 1.0
        d_np[-1] = 1.0

        a = to_tensor(a_np, device)
        b = to_tensor(b_np, device)
        c = to_tensor(c_np, device)
        d = to_tensor(d_np, device)

        x = run_on_device(tridiagonal_solve_tf, device, a, b, c, d)
        x_np = x.numpy()

        # Verify by reconstructing Ax
        result = np.zeros(n)
        result[0] = b_np[0] * x_np[0] + c_np[0] * x_np[1]
        for i in range(1, n - 1):
            result[i] = a_np[i - 1] * x_np[i - 1] + b_np[i] * x_np[i] + c_np[i] * x_np[i + 1]
        result[-1] = a_np[-1] * x_np[-2] + b_np[-1] * x_np[-1]

        np.testing.assert_array_almost_equal(result, d_np, decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_element_system(self, device):
        """Test minimal 2x2 tridiagonal system."""
        a = to_tensor([1.0], device)
        b = to_tensor([4.0, 4.0], device)
        c = to_tensor([1.0], device)
        d = to_tensor([5.0, 5.0], device)

        x = run_on_device(tridiagonal_solve_tf, device, a, b, c, d)
        x_np = x.numpy()
        b_np = b.numpy()
        c_np = c.numpy()
        a_np = a.numpy()

        result = np.array([
            b_np[0] * x_np[0] + c_np[0] * x_np[1],
            a_np[0] * x_np[0] + b_np[1] * x_np[1]
        ])
        np.testing.assert_array_almost_equal(result, d.numpy(), decimal=5)


class TestLeapfrogIntegrationTf:
    """Tests for the TensorFlow leapfrog_integration function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_stationary_particle(self, device):
        """A single particle with no velocity should remain stationary."""
        positions = to_tensor([[0.0, 0.0, 0.0]], device)
        velocities = to_tensor([[0.0, 0.0, 0.0]], device)
        masses = to_tensor([1.0], device)

        final_pos, final_vel = run_on_device(
            leapfrog_integration_tf, device,
            positions, velocities, masses, dt=0.01, n_steps=100
        )

        np.testing.assert_array_almost_equal(final_pos.numpy(), positions.numpy(), decimal=5)
        np.testing.assert_array_almost_equal(final_vel.numpy(), velocities.numpy(), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_moving_particle(self, device):
        """A single moving particle should move in a straight line."""
        positions = to_tensor([[0.0, 0.0, 0.0]], device)
        velocities = to_tensor([[1.0, 0.0, 0.0]], device)
        masses = to_tensor([1.0], device)

        dt = 0.01
        n_steps = 100

        final_pos, final_vel = run_on_device(
            leapfrog_integration_tf, device,
            positions, velocities, masses, dt=dt, n_steps=n_steps
        )

        np.testing.assert_array_almost_equal(final_vel.numpy(), velocities.numpy(), decimal=5)
        expected_pos = np.array([[dt * n_steps, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(final_pos.numpy(), expected_pos, decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_particles_approach(self, device):
        """Two particles should attract each other gravitationally."""
        positions = to_tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device)
        velocities = to_tensor(np.zeros((2, 3)), device)
        masses = to_tensor([1.0, 1.0], device)

        final_pos, _ = run_on_device(
            leapfrog_integration_tf, device,
            positions, velocities, masses, dt=0.01, n_steps=50, softening=0.1
        )

        initial_distance = 2.0
        final_distance = np.linalg.norm(final_pos.numpy()[1] - final_pos.numpy()[0])
        assert final_distance < initial_distance

    @pytest.mark.parametrize("device", DEVICES)
    def test_momentum_conservation(self, device):
        """Total momentum should be approximately conserved."""
        np.random.seed(42)
        n_particles = 5
        positions_np = np.random.randn(n_particles, 3)
        velocities_np = np.random.randn(n_particles, 3)
        masses_np = np.abs(np.random.randn(n_particles)) + 0.1

        positions = to_tensor(positions_np, device)
        velocities = to_tensor(velocities_np, device)
        masses = to_tensor(masses_np, device)

        initial_momentum = np.sum(masses_np[:, np.newaxis] * velocities_np, axis=0)

        final_pos, final_vel = run_on_device(
            leapfrog_integration_tf, device,
            positions, velocities, masses, dt=0.001, n_steps=100, softening=0.5
        )

        final_momentum = np.sum(masses_np[:, np.newaxis] * final_vel.numpy(), axis=0)

        np.testing.assert_array_almost_equal(initial_momentum, final_momentum, decimal=4)


class TestLongestIncreasingSubsequenceLengthTf:
    """Tests for the TensorFlow longest_increasing_subsequence_length function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_element(self, device):
        """Single element array should return 1."""
        arr = to_tensor([5.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_increasing(self, device):
        """Strictly increasing array - LIS is the whole array."""
        arr = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_decreasing(self, device):
        """Strictly decreasing array - LIS is length 1."""
        arr = to_tensor([5.0, 4.0, 3.0, 2.0, 1.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_classic_example(self, device):
        """Classic LIS example."""
        arr = to_tensor([10.0, 9.0, 2.0, 5.0, 3.0, 7.0, 101.0, 18.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 4

    @pytest.mark.parametrize("device", DEVICES)
    def test_all_same_elements(self, device):
        """All same elements - LIS is length 1."""
        arr = to_tensor([5.0, 5.0, 5.0, 5.0, 5.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_alternating_sequence(self, device):
        """Alternating high-low sequence."""
        arr = to_tensor([1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_elements_increasing(self, device):
        """Two elements in increasing order."""
        arr = to_tensor([1.0, 2.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 2

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_elements_decreasing(self, device):
        """Two elements in decreasing order."""
        arr = to_tensor([2.0, 1.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_longer_sequence(self, device):
        """Test with a longer sequence."""
        arr = to_tensor([0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 6

    @pytest.mark.parametrize("device", DEVICES)
    def test_negative_numbers(self, device):
        """Test with negative numbers."""
        arr = to_tensor([-5.0, -2.0, -8.0, -1.0, -6.0, 0.0], device, dtype=tf.float32)
        result = run_on_device(longest_increasing_subsequence_length_tf, device, arr)
        assert result == 4
