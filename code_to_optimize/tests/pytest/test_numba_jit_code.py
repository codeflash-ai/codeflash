import numpy as np
import pytest

from code_to_optimize.sample_code import (
    leapfrog_integration,
    longest_increasing_subsequence_length,
    tridiagonal_solve,
)


class TestTridiagonalSolve:
    """Tests for the tridiagonal_solve function (Thomas algorithm)."""

    def test_simple_system(self):
        """Test a simple 3x3 tridiagonal system with known solution."""
        # System: [2 -1 0] [x0]   [1]
        #         [-1 2 -1] [x1] = [0]
        #         [0 -1 2] [x2]   [1]
        a = np.array([-1.0, -1.0])  # lower diagonal
        b = np.array([2.0, 2.0, 2.0])  # main diagonal
        c = np.array([-1.0, -1.0])  # upper diagonal
        d = np.array([1.0, 0.0, 1.0])  # right-hand side

        x = tridiagonal_solve(a, b, c, d)

        # Verify solution by multiplying back
        # Ax should equal d
        result = np.zeros(3)
        result[0] = b[0] * x[0] + c[0] * x[1]
        result[1] = a[0] * x[0] + b[1] * x[1] + c[1] * x[2]
        result[2] = a[1] * x[1] + b[2] * x[2]

        np.testing.assert_array_almost_equal(result, d)

    def test_diagonal_system(self):
        """Test a purely diagonal system (a and c are zero)."""
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 3.0, 4.0])
        c = np.array([0.0, 0.0])
        d = np.array([4.0, 9.0, 16.0])

        x = tridiagonal_solve(a, b, c, d)

        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(x, expected)

    def test_larger_system(self):
        """Test a larger tridiagonal system."""
        n = 100
        a = -np.ones(n - 1)
        b = 2.0 * np.ones(n)
        c = -np.ones(n - 1)
        d = np.zeros(n)
        d[0] = 1.0
        d[-1] = 1.0

        x = tridiagonal_solve(a, b, c, d)

        # Verify by reconstructing Ax
        result = np.zeros(n)
        result[0] = b[0] * x[0] + c[0] * x[1]
        for i in range(1, n - 1):
            result[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
        result[-1] = a[-1] * x[-2] + b[-1] * x[-1]

        np.testing.assert_array_almost_equal(result, d, decimal=10)

    def test_two_element_system(self):
        """Test minimal 2x2 tridiagonal system."""
        a = np.array([1.0])
        b = np.array([4.0, 4.0])
        c = np.array([1.0])
        d = np.array([5.0, 5.0])

        x = tridiagonal_solve(a, b, c, d)

        # Verify: [4 1] [x0] = [5]
        #         [1 4] [x1]   [5]
        result = np.array([
            b[0] * x[0] + c[0] * x[1],
            a[0] * x[0] + b[1] * x[1]
        ])
        np.testing.assert_array_almost_equal(result, d)


class TestLeapfrogIntegration:
    """Tests for the leapfrog_integration function (N-body simulation)."""

    def test_single_stationary_particle(self):
        """A single particle with no velocity should remain stationary."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([1.0])

        final_pos, final_vel = leapfrog_integration(
            positions, velocities, masses, dt=0.01, n_steps=100
        )

        np.testing.assert_array_almost_equal(final_pos, positions)
        np.testing.assert_array_almost_equal(final_vel, velocities)

    def test_single_moving_particle(self):
        """A single moving particle should move in a straight line."""
        positions = np.array([[0.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])
        masses = np.array([1.0])

        dt = 0.01
        n_steps = 100

        final_pos, final_vel = leapfrog_integration(
            positions, velocities, masses, dt=dt, n_steps=n_steps
        )

        # With no other particles, velocity should remain constant
        np.testing.assert_array_almost_equal(final_vel, velocities)

        # Position should be initial + velocity * time
        expected_pos = np.array([[dt * n_steps, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(final_pos, expected_pos)

    def test_two_particles_approach(self):
        """Two particles should attract each other gravitationally."""
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        velocities = np.zeros((2, 3))
        masses = np.array([1.0, 1.0])

        final_pos, final_vel = leapfrog_integration(
            positions, velocities, masses, dt=0.01, n_steps=50, softening=0.1
        )

        # Particles should move closer together
        initial_distance = 2.0
        final_distance = np.linalg.norm(final_pos[1] - final_pos[0])
        assert final_distance < initial_distance

    def test_momentum_conservation(self):
        """Total momentum should be approximately conserved."""
        np.random.seed(42)
        n_particles = 5
        positions = np.random.randn(n_particles, 3)
        velocities = np.random.randn(n_particles, 3)
        masses = np.abs(np.random.randn(n_particles)) + 0.1

        initial_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)

        final_pos, final_vel = leapfrog_integration(
            positions, velocities, masses, dt=0.001, n_steps=100, softening=0.5
        )

        final_momentum = np.sum(masses[:, np.newaxis] * final_vel, axis=0)

        # Momentum should be conserved to good precision
        np.testing.assert_array_almost_equal(
            initial_momentum, final_momentum, decimal=5
        )

    def test_does_not_modify_input(self):
        """Input arrays should not be modified."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        velocities = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])

        pos_copy = positions.copy()
        vel_copy = velocities.copy()

        leapfrog_integration(positions, velocities, masses, dt=0.01, n_steps=10)

        np.testing.assert_array_equal(positions, pos_copy)
        np.testing.assert_array_equal(velocities, vel_copy)


class TestLongestIncreasingSubsequenceLength:
    """Tests for the longest_increasing_subsequence_length function."""

    def test_empty_array(self):
        """Empty array should return 0."""
        arr = np.array([], dtype=np.float64)
        assert longest_increasing_subsequence_length(arr) == 0

    def test_single_element(self):
        """Single element array should return 1."""
        arr = np.array([5])
        assert longest_increasing_subsequence_length(arr) == 1

    def test_strictly_increasing(self):
        """Strictly increasing array - LIS is the whole array."""
        arr = np.array([1, 2, 3, 4, 5])
        assert longest_increasing_subsequence_length(arr) == 5

    def test_strictly_decreasing(self):
        """Strictly decreasing array - LIS is length 1."""
        arr = np.array([5, 4, 3, 2, 1])
        assert longest_increasing_subsequence_length(arr) == 1

    def test_classic_example(self):
        """Classic LIS example: [10, 9, 2, 5, 3, 7, 101, 18]."""
        arr = np.array([10, 9, 2, 5, 3, 7, 101, 18])
        # LIS: [2, 3, 7, 101] or [2, 5, 7, 101] or [2, 3, 7, 18] etc.
        assert longest_increasing_subsequence_length(arr) == 4

    def test_all_same_elements(self):
        """All same elements - LIS is length 1 (strictly increasing)."""
        arr = np.array([5, 5, 5, 5, 5])
        assert longest_increasing_subsequence_length(arr) == 1

    def test_alternating_sequence(self):
        """Alternating high-low sequence."""
        arr = np.array([1, 10, 2, 9, 3, 8, 4, 7])
        # LIS: [1, 2, 3, 4] or [1, 2, 3, 4, 7] - length 5
        assert longest_increasing_subsequence_length(arr) == 5

    def test_two_elements_increasing(self):
        """Two elements in increasing order."""
        arr = np.array([1, 2])
        assert longest_increasing_subsequence_length(arr) == 2

    def test_two_elements_decreasing(self):
        """Two elements in decreasing order."""
        arr = np.array([2, 1])
        assert longest_increasing_subsequence_length(arr) == 1

    def test_longer_sequence(self):
        """Test with a longer sequence."""
        arr = np.array([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
        # Known LIS length for this sequence is 6
        assert longest_increasing_subsequence_length(arr) == 6

    def test_negative_numbers(self):
        """Test with negative numbers."""
        arr = np.array([-5, -2, -8, -1, -6, 0])
        # LIS: [-5, -2, -1, 0] or [-8, -6, 0] etc. - length 4
        assert longest_increasing_subsequence_length(arr) == 4

    def test_float_values(self):
        """Test with floating point values."""
        arr = np.array([1.5, 2.3, 1.8, 3.1, 2.9, 4.0])
        # LIS: [1.5, 2.3, 3.1, 4.0] or [1.5, 1.8, 2.9, 4.0] - length 4
        assert longest_increasing_subsequence_length(arr) == 4