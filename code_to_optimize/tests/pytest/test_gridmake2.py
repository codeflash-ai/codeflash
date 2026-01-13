import numpy as np
import pytest
from numpy.testing import assert_array_equal

from code_to_optimize.discrete_riccati import _gridmake2


class TestGridmake2With1DArrays:
    """Tests for _gridmake2 with two 1D arrays."""

    def test_basic_two_element_arrays(self):
        """Test basic cartesian product of two 2-element arrays."""
        x1 = np.array([1, 2])
        x2 = np.array([3, 4])
        result = _gridmake2(x1, x2)

        # Expected: x1 is tiled len(x2) times, x2 is repeated len(x1) times
        expected = np.array([
            [1, 3],
            [2, 3],
            [1, 4],
            [2, 4]
        ])
        assert_array_equal(result, expected)

    def test_different_length_arrays(self):
        """Test cartesian product with arrays of different lengths."""
        x1 = np.array([1, 2, 3])
        x2 = np.array([10, 20])
        result = _gridmake2(x1, x2)

        # Result should have len(x1) * len(x2) = 6 rows
        expected = np.array([
            [1, 10],
            [2, 10],
            [3, 10],
            [1, 20],
            [2, 20],
            [3, 20]
        ])
        assert_array_equal(result, expected)
        assert result.shape == (6, 2)

    def test_single_element_arrays(self):
        """Test with single-element arrays."""
        x1 = np.array([5])
        x2 = np.array([7])
        result = _gridmake2(x1, x2)

        expected = np.array([[5, 7]])
        assert_array_equal(result, expected)
        assert result.shape == (1, 2)

    def test_single_element_with_multi_element(self):
        """Test single-element array with multi-element array."""
        x1 = np.array([1])
        x2 = np.array([10, 20, 30])
        result = _gridmake2(x1, x2)

        expected = np.array([
            [1, 10],
            [1, 20],
            [1, 30]
        ])
        assert_array_equal(result, expected)

    def test_float_arrays(self):
        """Test with float arrays."""
        x1 = np.array([1.5, 2.5])
        x2 = np.array([0.1, 0.2])
        result = _gridmake2(x1, x2)

        expected = np.array([
            [1.5, 0.1],
            [2.5, 0.1],
            [1.5, 0.2],
            [2.5, 0.2]
        ])
        assert_array_equal(result, expected)

    def test_negative_values(self):
        """Test with negative values."""
        x1 = np.array([-1, 0, 1])
        x2 = np.array([-10, 10])
        result = _gridmake2(x1, x2)

        expected = np.array([
            [-1, -10],
            [0, -10],
            [1, -10],
            [-1, 10],
            [0, 10],
            [1, 10]
        ])
        assert_array_equal(result, expected)

    def test_result_shape(self):
        """Test that result shape is (len(x1)*len(x2), 2)."""
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array([5, 6, 7])
        result = _gridmake2(x1, x2)

        assert result.shape == (12, 2)

    def test_larger_arrays(self):
        """Test with larger arrays."""
        x1 = np.arange(10)
        x2 = np.arange(5)
        result = _gridmake2(x1, x2)

        assert result.shape == (50, 2)
        # Verify first column is x1 tiled 5 times
        assert_array_equal(result[:10, 0], x1)
        assert_array_equal(result[10:20, 0], x1)
        # Verify second column is x2 repeated 10 times each
        assert all(result[:10, 1] == 0)
        assert all(result[10:20, 1] == 1)


class TestGridmake2With2DFirst:
    """Tests for _gridmake2 when x1 is 2D and x2 is 1D."""

    def test_2d_first_1d_second(self):
        """Test with 2D first array and 1D second array."""
        x1 = np.array([[1, 2], [3, 4]])  # 2 rows, 2 cols
        x2 = np.array([10, 20])
        result = _gridmake2(x1, x2)

        # x1 is tiled len(x2) times vertically
        # x2 is repeated len(x1) times (2 rows)
        expected = np.array([
            [1, 2, 10],
            [3, 4, 10],
            [1, 2, 20],
            [3, 4, 20]
        ])
        assert_array_equal(result, expected)

    def test_2d_single_column(self):
        """Test with 2D array having single column."""
        x1 = np.array([[1], [2], [3]])  # 3 rows, 1 col
        x2 = np.array([10, 20])
        result = _gridmake2(x1, x2)

        expected = np.array([
            [1, 10],
            [2, 10],
            [3, 10],
            [1, 20],
            [2, 20],
            [3, 20]
        ])
        assert_array_equal(result, expected)

    def test_2d_multiple_columns(self):
        """Test with 2D array having multiple columns."""
        x1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2 rows, 3 cols
        x2 = np.array([100])
        result = _gridmake2(x1, x2)

        expected = np.array([
            [1, 2, 3, 100],
            [4, 5, 6, 100]
        ])
        assert_array_equal(result, expected)


class TestGridmake2EdgeCases:
    """Edge case tests for _gridmake2."""

    def test_empty_arrays_raise_or_return_empty(self):
        """Test behavior with empty arrays."""
        x1 = np.array([])
        x2 = np.array([1, 2])
        result = _gridmake2(x1, x2)
        # Empty x1 should result in empty output
        assert result.shape[0] == 0

    def test_both_empty_arrays(self):
        """Test with both empty arrays."""
        x1 = np.array([])
        x2 = np.array([])
        result = _gridmake2(x1, x2)
        assert result.shape[0] == 0

    def test_integer_dtype_preserved(self):
        """Test that integer dtype is handled correctly."""
        x1 = np.array([1, 2], dtype=np.int64)
        x2 = np.array([3, 4], dtype=np.int64)
        result = _gridmake2(x1, x2)
        assert result.dtype == np.int64

    def test_float_dtype_preserved(self):
        """Test that float dtype is handled correctly."""
        x1 = np.array([1.0, 2.0], dtype=np.float64)
        x2 = np.array([3.0, 4.0], dtype=np.float64)
        result = _gridmake2(x1, x2)
        assert result.dtype == np.float64


class TestGridmake2NotImplemented:
    """Tests for NotImplementedError cases."""

    def test_both_2d_raises(self):
        """Test that two 2D arrays raises NotImplementedError."""
        x1 = np.array([[1, 2], [3, 4]])
        x2 = np.array([[5, 6], [7, 8]])
        with pytest.raises(NotImplementedError):
            _gridmake2(x1, x2)

    def test_1d_first_2d_second_raises(self):
        """Test that 1D first and 2D second raises NotImplementedError."""
        x1 = np.array([1, 2])
        x2 = np.array([[5, 6], [7, 8]])
        with pytest.raises(NotImplementedError):
            _gridmake2(x1, x2)
