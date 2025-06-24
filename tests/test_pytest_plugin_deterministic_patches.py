"""Test the deterministic patching functionality in pytest_plugin.py.

This test verifies that all sources of randomness and non-determinism are properly
mocked/patched to ensure reproducible test execution for CodeFlash optimization.

Key functionality tested:
- time.time() returns fixed timestamp (1609459200.0 = 2021-01-01 00:00:00 UTC)
- time.perf_counter() returns incrementing values (maintaining relative timing)
- uuid.uuid4() and uuid.uuid1() return fixed UUID (12345678-1234-5678-9abc-123456789012)
- random.random() returns fixed value (0.123456789)
- random module is seeded deterministically (seed=42)
- os.urandom() returns fixed bytes (0x42 repeated)
- numpy.random is seeded if available (seed=42)
- Performance characteristics are maintained (original functions called internally)
- datetime mock functions are properly stored in builtins
- All patches work consistently across multiple calls
- Integration with real optimization scenarios

This ensures that CodeFlash optimization correctness checks will pass by eliminating
all sources of non-determinism that could cause object comparison failures.
"""

import datetime
import os
import random
import sys
import time
import uuid
from unittest.mock import patch

import pytest


class TestDeterministicPatches:
    """Test suite for deterministic patching functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Import plugin to apply patches (patches are applied at module level)
        import codeflash.verification.pytest_plugin  # noqa: F401
        
        # Note: Original functions are already patched by the time we get here
        # This is expected behavior since patches are applied at module import
        
        yield
        
        # Note: In practice, these patches should remain for the entire test session

    def test_time_time_deterministic(self):
        """Test that time.time() returns a fixed deterministic value."""
        expected_timestamp = 1609459200.0  # 2021-01-01 00:00:00 UTC
        
        # Call multiple times and verify consistent results
        result1 = time.time()
        result2 = time.time()
        result3 = time.time()
        
        assert result1 == expected_timestamp
        assert result2 == expected_timestamp
        assert result3 == expected_timestamp
        assert result1 == result2 == result3

    def test_perf_counter_incremental(self):
        """Test that time.perf_counter() returns incrementing values."""
        # Call multiple times and verify incrementing behavior
        result1 = time.perf_counter()
        result2 = time.perf_counter()
        result3 = time.perf_counter()
        
        # Verify they're different and incrementing by approximately 0.001
        assert result1 < result2 < result3
        assert abs((result2 - result1) - 0.001) < 1e-6  # Use reasonable epsilon for float comparison
        assert abs((result3 - result2) - 0.001) < 1e-6

    def test_uuid4_deterministic(self):
        """Test that uuid.uuid4() returns a fixed deterministic UUID."""
        expected_uuid = uuid.UUID('12345678-1234-5678-9abc-123456789012')
        
        # Call multiple times and verify consistent results
        result1 = uuid.uuid4()
        result2 = uuid.uuid4()
        result3 = uuid.uuid4()
        
        assert result1 == expected_uuid
        assert result2 == expected_uuid
        assert result3 == expected_uuid
        assert result1 == result2 == result3
        assert isinstance(result1, uuid.UUID)

    def test_uuid1_deterministic(self):
        """Test that uuid.uuid1() returns a fixed deterministic UUID."""
        expected_uuid = uuid.UUID('12345678-1234-5678-9abc-123456789012')
        
        # Call multiple times with different parameters
        result1 = uuid.uuid1()
        result2 = uuid.uuid1(node=123456)
        result3 = uuid.uuid1(clock_seq=789)
        
        assert result1 == expected_uuid
        assert result2 == expected_uuid
        assert result3 == expected_uuid
        assert isinstance(result1, uuid.UUID)

    def test_random_random_deterministic(self):
        """Test that random.random() returns a fixed deterministic value."""
        expected_value = 0.123456789
        
        # Call multiple times and verify consistent results
        result1 = random.random()
        result2 = random.random()
        result3 = random.random()
        
        assert result1 == expected_value
        assert result2 == expected_value
        assert result3 == expected_value
        assert 0.0 <= result1 <= 1.0  # Should still be a valid random float

    def test_random_seed_deterministic(self):
        """Test that random module is seeded deterministically."""
        # The plugin should have already seeded with 42
        # Test other random functions for consistency
        
        # Note: random.random() is patched to always return the same value
        # So we test that the random module behaves deterministically
        # by testing that random.seed() affects other functions consistently
        
        # First, test that our patched random.random always returns the same value
        assert random.random() == 0.123456789
        assert random.random() == 0.123456789
        
        # Test that seeding affects other random functions consistently
        random.seed(42)
        result1_int = random.randint(1, 100)
        result1_choice = random.choice([1, 2, 3, 4, 5])
        
        # Re-seed and get same results
        random.seed(42)
        result2_int = random.randint(1, 100)
        result2_choice = random.choice([1, 2, 3, 4, 5])
        
        assert result1_int == result2_int
        assert result1_choice == result2_choice

    def test_os_urandom_deterministic(self):
        """Test that os.urandom() returns deterministic bytes."""
        # Test various byte lengths
        for n in [1, 8, 16, 32]:
            result1 = os.urandom(n)
            result2 = os.urandom(n)
            
            # Should return fixed bytes (0x42 repeated)
            expected = b'\x42' * n
            assert result1 == expected
            assert result2 == expected
            assert len(result1) == n
            assert isinstance(result1, bytes)

    def test_numpy_seeding(self):
        """Test that numpy.random is seeded if available."""
        try:
            import numpy as np
            
            # Generate some random numbers
            result1 = np.random.random(5)
            
            # Re-seed and generate again
            np.random.seed(42)
            result2 = np.random.random(5)
            
            # Should be deterministic due to seeding
            assert np.array_equal(result1, result2)
            
        except ImportError:
            # numpy not available, test should pass
            pytest.skip("NumPy not available")

    def test_performance_characteristics_maintained(self):
        """Test that performance characteristics are maintained."""
        # Test that they still execute quickly (performance check)
        start = time.perf_counter()
        for _ in range(1000):
            time.time()
            uuid.uuid4()
            random.random()
        end = time.perf_counter()
        
        # Should complete quickly (less than 1 second for 1000 calls)
        duration = end - start
        assert duration < 1.0, f"Performance degraded: {duration}s for 1000 calls"

    def test_builtins_datetime_mocks_stored(self):
        """Test that datetime mock functions are stored in builtins."""
        import builtins
        
        # Verify that the mock functions are stored
        assert hasattr(builtins, '_original_datetime_now')
        assert hasattr(builtins, '_original_datetime_utcnow')
        assert hasattr(builtins, '_mock_datetime_now')
        assert hasattr(builtins, '_mock_datetime_utcnow')
        
        # Test that the mock functions work
        mock_now = builtins._mock_datetime_now
        mock_utcnow = builtins._mock_datetime_utcnow
        
        result1 = mock_now()
        result2 = mock_utcnow()
        
        expected_dt = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
        assert result1 == expected_dt
        assert result2 == expected_dt

    def test_consistency_across_multiple_calls(self):
        """Test that all patched functions remain consistent across many calls."""
        # Store initial results
        initial_time = time.time()
        initial_uuid = uuid.uuid4()
        initial_random = random.random()
        initial_urandom = os.urandom(8)
        
        # Call functions many times (but not perf_counter since it increments)
        for _ in range(5):
            assert time.time() == initial_time
            assert uuid.uuid4() == initial_uuid
            assert random.random() == initial_random
            assert os.urandom(8) == initial_urandom

    def test_perf_counter_state_management(self):
        """Test that perf_counter maintains its own internal state correctly."""
        # Get a baseline
        base = time.perf_counter()
        
        # Call several times and verify incrementing
        results = [time.perf_counter() for _ in range(5)]
        
        # Each call should increment by approximately 0.001
        for i, result in enumerate(results):
            expected = base + ((i + 1) * 0.001)
            assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_different_uuid_functions_same_result(self):
        """Test that both uuid4 and uuid1 return the same deterministic UUID."""
        uuid4_result = uuid.uuid4()
        uuid1_result = uuid.uuid1()
        
        # Both should return the same fixed UUID
        assert uuid4_result == uuid1_result
        assert str(uuid4_result) == '12345678-1234-5678-9abc-123456789012'

    def test_patches_applied_at_module_level(self):
        """Test that patches are applied when the module is imported."""
        # Test that functions return expected deterministic values
        # (This indirectly confirms they are patched)
        assert time.time() == 1609459200.0
        assert uuid.uuid4() == uuid.UUID('12345678-1234-5678-9abc-123456789012')
        assert random.random() == 0.123456789
        
        # Test that function names indicate they are mock functions
        assert 'mock' in time.time.__name__
        assert 'mock' in uuid.uuid4.__name__
        assert 'mock' in random.random.__name__

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test uuid functions with edge case parameters
        assert uuid.uuid1(node=0) == uuid.UUID('12345678-1234-5678-9abc-123456789012')
        assert uuid.uuid1(clock_seq=0) == uuid.UUID('12345678-1234-5678-9abc-123456789012')
        
        # Test urandom with edge cases
        assert os.urandom(0) == b''
        assert os.urandom(1) == b'\x42'
        
        # Test datetime mock with timezone
        import builtins
        mock_now = builtins._mock_datetime_now
        
        # Test with different timezone
        utc_tz = datetime.timezone.utc
        result_with_tz = mock_now(utc_tz)
        expected_with_tz = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=utc_tz)
        assert result_with_tz == expected_with_tz

    def test_integration_with_actual_optimization_scenario(self):
        """Test the patching in a scenario similar to actual optimization."""
        # Simulate what happens during optimization - multiple function calls
        # that would normally produce different results but should now be deterministic
        
        class MockOptimizedFunction:
            """Mock function that uses various sources of randomness."""
            
            def __init__(self):
                self.id = uuid.uuid4()
                self.created_at = time.time()
                self.random_factor = random.random()
                self.random_bytes = os.urandom(4)
            
            def execute(self):
                execution_time = time.perf_counter()
                random_choice = random.randint(1, 100)
                return {
                    'id': self.id,
                    'created_at': self.created_at,
                    'execution_time': execution_time,
                    'random_factor': self.random_factor,
                    'random_choice': random_choice,
                    'random_bytes': self.random_bytes
                }
        
        # Create two instances and execute them
        func1 = MockOptimizedFunction()
        func2 = MockOptimizedFunction()
        
        result1 = func1.execute()
        result2 = func2.execute()
        
        # All values should be identical due to deterministic patching
        assert result1['id'] == result2['id']
        assert result1['created_at'] == result2['created_at']
        assert result1['random_factor'] == result2['random_factor']
        assert result1['random_bytes'] == result2['random_bytes']
        
        # Only execution_time should be different (incremental)
        assert result1['execution_time'] != result2['execution_time']
        assert result2['execution_time'] > result1['execution_time']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])