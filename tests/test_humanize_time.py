from codeflash.code_utils.time_utils import humanize_runtime, format_time
from codeflash.code_utils.time_utils import format_perf
import pytest


def test_humanize_runtime():
    assert humanize_runtime(0) == "0.00 nanoseconds"
    assert humanize_runtime(1000) == "1.00 microsecond"
    assert humanize_runtime(1000000) == "1.00 millisecond"
    assert humanize_runtime(1000000000) == "1.00 second"
    assert humanize_runtime(60000000000) == "1.00 minute"
    assert humanize_runtime(3600000000000) == "1.00 hour"
    assert humanize_runtime(86400000000000) == "1.00 day"

    assert humanize_runtime(1) == "1.00 nanosecond"
    assert humanize_runtime(12) == "12.0 nanoseconds"
    assert humanize_runtime(123) == "123 nanoseconds"
    assert humanize_runtime(999) == "999 nanoseconds"
    assert humanize_runtime(1234) == "1.23 microseconds"
    assert humanize_runtime(12345) == "12.3 microseconds"
    assert humanize_runtime(123456) == "123 microseconds"
    assert humanize_runtime(1234567) == "1.23 milliseconds"
    assert humanize_runtime(12345678) == "12.3 milliseconds"
    assert humanize_runtime(123456789) == "123 milliseconds"

    assert humanize_runtime(1234567891) == "1.23 seconds"
    assert humanize_runtime(12345678912) == "12.3 seconds"
    assert humanize_runtime(123456789123) == "2.06 minutes"
    assert humanize_runtime(1234567891234) == "20.6 minutes"
    assert humanize_runtime(12345678912345) == "3.43 hours"
    assert humanize_runtime(98765431298760) == "1.14 days"
    assert humanize_runtime(197530862597520) == "2.29 days"


class TestFormatTime:
    """Test cases for the format_time function."""

    def test_nanoseconds_range(self):
        """Test formatting for nanoseconds (< 1,000 ns)."""
        assert format_time(0) == "0ns"
        assert format_time(1) == "1ns"
        assert format_time(500) == "500ns"
        assert format_time(999) == "999ns"

    def test_microseconds_range(self):
        """Test formatting for microseconds (1,000 ns to 999,999 ns)."""
        # Integer microseconds >= 100
        # assert format_time(100_000) == "100μs"
        # assert format_time(500_000) == "500μs"
        # assert format_time(999_000) == "999μs"

        # Decimal microseconds with varying precision
        assert format_time(1_000) == "1.00μs"  # 1.0 μs, 2 decimal places
        assert format_time(1_500) == "1.50μs"  # 1.5 μs, 2 decimal places
        assert format_time(9_999) == "10.00μs"  # 9.999 μs rounds to 10.00
        assert format_time(10_000) == "10.0μs"  # 10.0 μs, 1 decimal place
        assert format_time(15_500) == "15.5μs"  # 15.5 μs, 1 decimal place
        assert format_time(99_900) == "99.9μs"  # 99.9 μs, 1 decimal place

    def test_milliseconds_range(self):
        """Test formatting for milliseconds (1,000,000 ns to 999,999,999 ns)."""
        # Integer milliseconds >= 100
        assert format_time(100_000_000) == "100ms"
        assert format_time(500_000_000) == "500ms"
        assert format_time(999_000_000) == "999ms"

        # Decimal milliseconds with varying precision
        assert format_time(1_000_000) == "1.00ms"  # 1.0 ms, 2 decimal places
        assert format_time(1_500_000) == "1.50ms"  # 1.5 ms, 2 decimal places
        assert format_time(9_999_000) == "10.00ms"  # 9.999 ms rounds to 10.00
        assert format_time(10_000_000) == "10.0ms"  # 10.0 ms, 1 decimal place
        assert format_time(15_500_000) == "15.5ms"  # 15.5 ms, 1 decimal place
        assert format_time(99_900_000) == "99.9ms"  # 99.9 ms, 1 decimal place

    def test_seconds_range(self):
        """Test formatting for seconds (>= 1,000,000,000 ns)."""
        # Integer seconds >= 100
        assert format_time(100_000_000_000) == "100s"
        assert format_time(500_000_000_000) == "500s"
        assert format_time(999_000_000_000) == "999s"

        # Decimal seconds with varying precision
        assert format_time(1_000_000_000) == "1.00s"  # 1.0 s, 2 decimal places
        assert format_time(1_500_000_000) == "1.50s"  # 1.5 s, 2 decimal places
        assert format_time(9_999_000_000) == "10.00s"  # 9.999 s rounds to 10.00
        assert format_time(10_000_000_000) == "10.0s"  # 10.0 s, 1 decimal place
        assert format_time(15_500_000_000) == "15.5s"  # 15.5 s, 1 decimal place
        assert format_time(99_900_000_000) == "99.9s"  # 99.9 s, 1 decimal place

    def test_boundary_values(self):
        """Test exact boundary values between units."""
        # Boundaries between nanoseconds and microseconds
        assert format_time(999) == "999ns"
        assert format_time(1_000) == "1.00μs"

        # Boundaries between microseconds and milliseconds
        assert format_time(999_999) == "999μs"  # This might round to 1000.00μs
        assert format_time(1_000_000) == "1.00ms"

        # Boundaries between milliseconds and seconds
        assert format_time(999_999_999) == "999ms"  # This might round to 1000.00ms
        assert format_time(1_000_000_000) == "1.00s"

    def test_precision_boundaries(self):
        """Test precision changes at significant digit boundaries."""
        # Microseconds precision changes
        assert format_time(9_950) == "9.95μs"  # 2 decimal places
        assert format_time(10_000) == "10.0μs"  # 1 decimal place
        assert format_time(99_900) == "99.9μs"  # 1 decimal place
        assert format_time(100_000) == "100μs"  # No decimal places

        # Milliseconds precision changes
        assert format_time(9_950_000) == "9.95ms"  # 2 decimal places
        assert format_time(10_000_000) == "10.0ms"  # 1 decimal place
        assert format_time(99_900_000) == "99.9ms"  # 1 decimal place
        assert format_time(100_000_000) == "100ms"  # No decimal places

        # Seconds precision changes
        assert format_time(9_950_000_000) == "9.95s"  # 2 decimal places
        assert format_time(10_000_000_000) == "10.0s"  # 1 decimal place
        assert format_time(99_900_000_000) == "99.9s"  # 1 decimal place
        assert format_time(100_000_000_000) == "100s"  # No decimal places

    def test_rounding_behavior(self):
        """Test rounding behavior for edge cases."""
        # Test rounding in microseconds
        assert format_time(1_234) == "1.23μs"
        assert format_time(1_235) == "1.24μs"  # Should round up
        assert format_time(12_345) == "12.3μs"
        assert format_time(12_350) == "12.3μs"  # Should round up

        # Test rounding in milliseconds
        assert format_time(1_234_000) == "1.23ms"
        assert format_time(1_235_000) == "1.24ms"  # Should round up
        assert format_time(12_345_000) == "12.3ms"
        assert format_time(12_350_000) == "12.3ms"  # Should round up

    def test_large_values(self):
        """Test very large nanosecond values."""
        assert format_time(3_600_000_000_000) == "3600s"  # 1 hour
        assert format_time(86_400_000_000_000) == "86400s"  # 1 day

    @pytest.mark.parametrize("nanoseconds,expected", [
        (0, "0ns"),
        (42, "42ns"),
        (1_500, "1.50μs"),
        (25_000, "25.0μs"),
        (150_000, "150μs"),
        (2_500_000, "2.50ms"),
        (45_000_000, "45.0ms"),
        (200_000_000, "200ms"),
        (3_500_000_000, "3.50s"),
        (75_000_000_000, "75.0s"),
        (300_000_000_000, "300s"),
    ])
    def test_parametrized_examples(self, nanoseconds, expected):
        """Parametrized test with various input/output combinations."""
        assert format_time(nanoseconds) == expected

    def test_invalid_input_types(self):
        """Test that function handles invalid input types appropriately."""
        with pytest.raises(TypeError):
            format_time("1000")

        with pytest.raises(TypeError):
            format_time(1000.5)

        with pytest.raises(TypeError):
            format_time(None)

    def test_negative_values(self):
        """Test behavior with negative values (if applicable)."""
        # This test depends on whether your function should handle negative values
        # You might want to modify based on expected behavior
        with pytest.raises((ValueError, TypeError)) or pytest.warns():
            format_time(-1000)


class TestFormatPerf:
    """Test cases for the format_perf function."""

    def test_format_perf_large_values_above_100(self):
        """Test formatting for values above 100 (no decimal places)."""
        assert format_perf(150.789) == "151"
        assert format_perf(999.999) == "1000"
        assert format_perf(100.1) == "100"
        assert format_perf(500) == "500"
        assert format_perf(1000.5) == "1000"

    def test_format_perf_medium_values_10_to_100(self):
        """Test formatting for values between 10 and 100 (1 decimal place)."""
        assert format_perf(99.99) == "100.0"
        assert format_perf(50.789) == "50.8"
        assert format_perf(10.1) == "10.1"
        assert format_perf(25.0) == "25.0"
        assert format_perf(33.333) == "33.3"

    def test_format_perf_small_values_1_to_10(self):
        """Test formatting for values between 1 and 10 (2 decimal places)."""
        assert format_perf(9.999) == "10.00"
        assert format_perf(5.789) == "5.79"
        assert format_perf(1.1) == "1.10"
        assert format_perf(2.0) == "2.00"
        assert format_perf(7.123) == "7.12"

    def test_format_perf_very_small_values_below_1(self):
        """Test formatting for values below 1 (3 decimal places)."""
        assert format_perf(0.999) == "0.999"
        assert format_perf(0.5) == "0.500"
        assert format_perf(0.123) == "0.123"
        assert format_perf(0.001) == "0.001"
        assert format_perf(0.0) == "0.000"

    def test_format_perf_negative_values(self):
        """Test formatting for negative values (uses absolute value for comparison)."""
        assert format_perf(-150.789) == "-151"
        assert format_perf(-50.789) == "-50.8"
        assert format_perf(-5.789) == "-5.79"
        assert format_perf(-0.999) == "-0.999"
        assert format_perf(-0.0) == "-0.000"

    def test_format_perf_boundary_values(self):
        """Test formatting for exact boundary values."""
        assert format_perf(100.0) == "100"
        assert format_perf(10.0) == "10.0"
        assert format_perf(1.0) == "1.00"
        assert format_perf(-100.0) == "-100"
        assert format_perf(-10.0) == "-10.0"
        assert format_perf(-1.0) == "-1.00"

    def test_format_perf_integer_inputs(self):
        """Test formatting with integer inputs."""
        assert format_perf(150) == "150"
        assert format_perf(50) == "50.0"
        assert format_perf(5) == "5.00"
        assert format_perf(0) == "0.000"
        assert format_perf(-150) == "-150"
        assert format_perf(-50) == "-50.0"
        assert format_perf(-5) == "-5.00"

    def test_format_perf_float_inputs(self):
        """Test formatting with float inputs."""
        assert format_perf(123.456) == "123"
        assert format_perf(12.3456) == "12.3"
        assert format_perf(1.23456) == "1.23"
        assert format_perf(0.123456) == "0.123"

    def test_format_perf_edge_cases(self):
        """Test formatting for edge cases and special values."""
        # Very large numbers
        assert format_perf(999999.99) == "1000000"
        assert format_perf(1000000) == "1000000"

        # Very small positive numbers
        assert format_perf(0.0001) == "0.000"
        assert format_perf(0.00001) == "0.000"

        # Numbers very close to boundaries
        assert format_perf(99.9999) == "100.0"
        assert format_perf(9.9999) == "10.00"
        assert format_perf(0.9999) == "1.000"

    def test_format_perf_rounding_behavior(self):
        """Test that rounding behavior is consistent."""
        # Test rounding up
        assert format_perf(100.5) == "100"
        assert format_perf(10.55) == "10.6"
        assert format_perf(1.555) == "1.55"
        assert format_perf(0.1555) == "0.155"

        # Test rounding down
        assert format_perf(100.4) == "100"
        assert format_perf(10.54) == "10.5"
        assert format_perf(1.554) == "1.55"
        assert format_perf(0.1554) == "0.155"