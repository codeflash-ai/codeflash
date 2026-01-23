"""
Unit tests for JAX implementations of JIT-suitable functions.

Tests run on CUDA devices only.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

try:
    gpu_devices = jax.devices("gpu")
    has_gpu = bool(gpu_devices)
except RuntimeError:
    has_gpu = False

pytestmark = pytest.mark.skipif(
    not has_gpu,
    reason="GPU not available - tests require CUDA"
)

from code_to_optimize.sample_code import (
    leapfrog_integration_jax,
    longest_increasing_subsequence_length_jax,
    tridiagonal_solve_jax,
)


def get_available_devices():
    """Return list of available JAX devices for testing."""
    # Only test on CUDA devices
    return ["cuda"]


DEVICES = get_available_devices()


def to_device(arr, device):
    """Move a JAX array to the specified device."""
    if device == "cpu":
        return jax.device_put(arr, jax.devices("cpu")[0])
    elif device == "cuda":
        return jax.device_put(arr, jax.devices("gpu")[0])
    elif device == "metal":
        return jax.device_put(arr, jax.devices("METAL")[0])
    return arr


class TestTridiagonalSolveJax:
    """Tests for the JAX tridiagonal_solve function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_simple_system(self, device):
        """Test a simple 3x3 tridiagonal system with known solution."""
        a = jnp.array([-1.0, -1.0])
        b = jnp.array([2.0, 2.0, 2.0])
        c = jnp.array([-1.0, -1.0])
        d = jnp.array([1.0, 0.0, 1.0])

        a, b, c, d = to_device(a, device), to_device(b, device), to_device(c, device), to_device(d, device)

        x = tridiagonal_solve_jax(a, b, c, d)

        # Verify solution by multiplying back
        result = jnp.zeros(3)
        result = result.at[0].set(b[0] * x[0] + c[0] * x[1])
        result = result.at[1].set(a[0] * x[0] + b[1] * x[1] + c[1] * x[2])
        result = result.at[2].set(a[1] * x[1] + b[2] * x[2])

        np.testing.assert_array_almost_equal(np.array(result), np.array(d), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_diagonal_system(self, device):
        """Test a purely diagonal system."""
        a = jnp.array([0.0, 0.0])
        b = jnp.array([2.0, 3.0, 4.0])
        c = jnp.array([0.0, 0.0])
        d = jnp.array([4.0, 9.0, 16.0])

        a, b, c, d = to_device(a, device), to_device(b, device), to_device(c, device), to_device(d, device)

        x = tridiagonal_solve_jax(a, b, c, d)

        expected = jnp.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(np.array(x), np.array(expected), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_larger_system(self, device):
        """Test a larger tridiagonal system."""
        n = 50
        a = -jnp.ones(n - 1)
        b = 2.0 * jnp.ones(n)
        c = -jnp.ones(n - 1)
        d = jnp.zeros(n).at[0].set(1.0).at[-1].set(1.0)

        a, b, c, d = to_device(a, device), to_device(b, device), to_device(c, device), to_device(d, device)

        x = tridiagonal_solve_jax(a, b, c, d)

        # Verify by reconstructing Ax
        result = jnp.zeros(n)
        result = result.at[0].set(b[0] * x[0] + c[0] * x[1])
        for i in range(1, n - 1):
            result = result.at[i].set(a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1])
        result = result.at[-1].set(a[-1] * x[-2] + b[-1] * x[-1])

        np.testing.assert_array_almost_equal(np.array(result), np.array(d), decimal=5)


class TestLeapfrogIntegrationJax:
    """Tests for the JAX leapfrog_integration function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_stationary_particle(self, device):
        """A single particle with no velocity should remain stationary."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 0.0, 0.0]])
        masses = jnp.array([1.0])

        positions = to_device(positions, device)
        velocities = to_device(velocities, device)
        masses = to_device(masses, device)

        final_pos, final_vel = leapfrog_integration_jax(
            positions, velocities, masses, dt=0.01, n_steps=100
        )

        np.testing.assert_array_almost_equal(np.array(final_pos), np.array(positions), decimal=5)
        np.testing.assert_array_almost_equal(np.array(final_vel), np.array(velocities), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_moving_particle(self, device):
        """A single moving particle should move in a straight line."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array([[1.0, 0.0, 0.0]])
        masses = jnp.array([1.0])

        positions = to_device(positions, device)
        velocities = to_device(velocities, device)
        masses = to_device(masses, device)

        dt = 0.01
        n_steps = 100

        final_pos, final_vel = leapfrog_integration_jax(
            positions, velocities, masses, dt=dt, n_steps=n_steps
        )

        np.testing.assert_array_almost_equal(np.array(final_vel), np.array(velocities), decimal=5)
        expected_pos = jnp.array([[dt * n_steps, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(np.array(final_pos), np.array(expected_pos), decimal=5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_particles_approach(self, device):
        """Two particles should attract each other gravitationally."""
        positions = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = jnp.zeros((2, 3))
        masses = jnp.array([1.0, 1.0])

        positions = to_device(positions, device)
        velocities = to_device(velocities, device)
        masses = to_device(masses, device)

        final_pos, _ = leapfrog_integration_jax(
            positions, velocities, masses, dt=0.01, n_steps=50, softening=0.1
        )

        initial_distance = 2.0
        final_distance = float(jnp.linalg.norm(final_pos[1] - final_pos[0]))
        assert final_distance < initial_distance

    @pytest.mark.parametrize("device", DEVICES)
    def test_momentum_conservation(self, device):
        """Total momentum should be approximately conserved."""
        np.random.seed(42)
        n_particles = 5
        positions = jnp.array(np.random.randn(n_particles, 3))
        velocities = jnp.array(np.random.randn(n_particles, 3))
        masses = jnp.array(np.abs(np.random.randn(n_particles)) + 0.1)

        positions = to_device(positions, device)
        velocities = to_device(velocities, device)
        masses = to_device(masses, device)

        initial_momentum = jnp.sum(masses[:, jnp.newaxis] * velocities, axis=0)

        final_pos, final_vel = leapfrog_integration_jax(
            positions, velocities, masses, dt=0.001, n_steps=100, softening=0.5
        )

        final_momentum = jnp.sum(masses[:, jnp.newaxis] * final_vel, axis=0)

        np.testing.assert_array_almost_equal(
            np.array(initial_momentum), np.array(final_momentum), decimal=4
        )


class TestLongestIncreasingSubsequenceLengthJax:
    """Tests for the JAX longest_increasing_subsequence_length function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_empty_array(self, device):
        """Empty array should return 0."""
        arr = jnp.array([], dtype=jnp.float32)
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_single_element(self, device):
        """Single element array should return 1."""
        arr = jnp.array([5.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_increasing(self, device):
        """Strictly increasing array - LIS is the whole array."""
        arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_strictly_decreasing(self, device):
        """Strictly decreasing array - LIS is length 1."""
        arr = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_classic_example(self, device):
        """Classic LIS example."""
        arr = jnp.array([10.0, 9.0, 2.0, 5.0, 3.0, 7.0, 101.0, 18.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 4

    @pytest.mark.parametrize("device", DEVICES)
    def test_all_same_elements(self, device):
        """All same elements - LIS is length 1."""
        arr = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 1

    @pytest.mark.parametrize("device", DEVICES)
    def test_alternating_sequence(self, device):
        """Alternating high-low sequence."""
        arr = jnp.array([1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 5

    @pytest.mark.parametrize("device", DEVICES)
    def test_longer_sequence(self, device):
        """Test with a longer sequence."""
        arr = jnp.array([0.0, 8.0, 4.0, 12.0, 2.0, 10.0, 6.0, 14.0, 1.0, 9.0, 5.0, 13.0, 3.0, 11.0, 7.0, 15.0])
        arr = to_device(arr, device)
        assert longest_increasing_subsequence_length_jax(arr) == 6