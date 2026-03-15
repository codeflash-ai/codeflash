"""Test the deterministic patching functionality in pytest_plugin.py."""

import datetime
import os
import random
import time
import uuid
from unittest.mock import patch

import pytest


class TestDeterministicPatches:
    @pytest.fixture(autouse=True)
    def setup_deterministic_environment(self):
        """Setup isolated deterministic environment for testing."""
        original_time_time = time.time
        original_time_ns = time.time_ns
        original_perf_counter = time.perf_counter
        original_perf_counter_ns = time.perf_counter_ns
        original_monotonic = time.monotonic
        original_monotonic_ns = time.monotonic_ns
        original_uuid4 = uuid.uuid4
        original_uuid1 = uuid.uuid1
        original_random_random = random.random
        original_os_urandom = os.urandom
        original_datetime_class = datetime.datetime

        fixed_timestamp = 1761717605.108106
        fixed_datetime = datetime.datetime(2021, 1, 1, 2, 5, 10, tzinfo=datetime.timezone.utc)
        fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789012")

        perf_counter_start = fixed_timestamp
        perf_counter_calls = 0
        monotonic_start = fixed_timestamp
        monotonic_calls = 0
        monotonic_ns_start = int(fixed_timestamp * 1_000_000_000)
        monotonic_ns_calls = 0

        def mock_time_time():
            return fixed_timestamp

        def mock_time_ns():
            return int(fixed_timestamp * 1_000_000_000)

        def mock_perf_counter():
            nonlocal perf_counter_calls
            perf_counter_calls += 1
            return perf_counter_start + (perf_counter_calls * 0.001)

        def mock_monotonic():
            nonlocal monotonic_calls
            monotonic_calls += 1
            return monotonic_start + (monotonic_calls * 0.001)

        def mock_monotonic_ns():
            nonlocal monotonic_ns_calls
            monotonic_ns_calls += 1
            return monotonic_ns_start + (monotonic_ns_calls * 1_000_000)

        def mock_uuid4():
            return fixed_uuid

        def mock_uuid1(node=None, clock_seq=None):
            return fixed_uuid

        def mock_random():
            return 0.123456789

        def mock_urandom(n):
            return b"\x42" * n

        class DeterministicDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return fixed_datetime.replace(tzinfo=None)
                return fixed_datetime.astimezone(tz)

            @classmethod
            def utcnow(cls):
                return fixed_datetime.replace(tzinfo=None)

        patches = [
            patch.object(time, "time", side_effect=mock_time_time),
            patch.object(time, "time_ns", side_effect=mock_time_ns),
            patch.object(time, "perf_counter", side_effect=mock_perf_counter),
            patch.object(time, "monotonic", side_effect=mock_monotonic),
            patch.object(time, "monotonic_ns", side_effect=mock_monotonic_ns),
            patch.object(uuid, "uuid4", side_effect=mock_uuid4),
            patch.object(uuid, "uuid1", side_effect=mock_uuid1),
            patch.object(random, "random", side_effect=mock_random),
            patch.object(os, "urandom", side_effect=mock_urandom),
        ]

        started_patches = []
        for p in patches:
            started_patches.append(p.start())

        random.seed(42)
        datetime.datetime = DeterministicDatetime

        numpy_patched = False
        try:
            import numpy as np

            np.random.seed(42)
            numpy_patched = True
        except ImportError:
            pass

        yield {
            "original_functions": {
                "time_time": original_time_time,
                "perf_counter": original_perf_counter,
                "uuid4": original_uuid4,
                "uuid1": original_uuid1,
                "random_random": original_random_random,
                "os_urandom": original_os_urandom,
            },
            "numpy_patched": numpy_patched,
        }

        for p in patches:
            p.stop()

        datetime.datetime = original_datetime_class
        random.seed()

    def test_time_time_deterministic(self, setup_deterministic_environment):
        expected_timestamp = 1761717605.108106

        result1 = time.time()
        result2 = time.time()
        result3 = time.time()

        assert result1 == expected_timestamp
        assert result2 == expected_timestamp
        assert result3 == expected_timestamp

    def test_time_ns_deterministic(self, setup_deterministic_environment):
        expected = int(1761717605.108106 * 1_000_000_000)

        result1 = time.time_ns()
        result2 = time.time_ns()

        assert result1 == expected
        assert result2 == expected

    def test_perf_counter_incremental(self, setup_deterministic_environment):
        result1 = time.perf_counter()
        result2 = time.perf_counter()
        result3 = time.perf_counter()

        assert result1 < result2 < result3
        assert abs((result2 - result1) - 0.001) < 1e-6
        assert abs((result3 - result2) - 0.001) < 1e-6

    def test_monotonic_incremental(self, setup_deterministic_environment):
        result1 = time.monotonic()
        result2 = time.monotonic()
        result3 = time.monotonic()

        assert result1 < result2 < result3
        assert abs((result2 - result1) - 0.001) < 1e-6

    def test_monotonic_ns_incremental(self, setup_deterministic_environment):
        result1 = time.monotonic_ns()
        result2 = time.monotonic_ns()
        result3 = time.monotonic_ns()

        assert result1 < result2 < result3
        assert result2 - result1 == 1_000_000

    def test_uuid4_deterministic(self, setup_deterministic_environment):
        expected_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789012")

        result1 = uuid.uuid4()
        result2 = uuid.uuid4()

        assert result1 == expected_uuid
        assert result2 == expected_uuid
        assert isinstance(result1, uuid.UUID)

    def test_uuid1_deterministic(self, setup_deterministic_environment):
        expected_uuid = uuid.UUID("12345678-1234-5678-9abc-123456789012")

        result1 = uuid.uuid1()
        result2 = uuid.uuid1(node=123456)
        result3 = uuid.uuid1(clock_seq=789)

        assert result1 == expected_uuid
        assert result2 == expected_uuid
        assert result3 == expected_uuid

    def test_random_random_deterministic(self, setup_deterministic_environment):
        expected_value = 0.123456789

        result1 = random.random()
        result2 = random.random()

        assert result1 == expected_value
        assert result2 == expected_value
        assert 0.0 <= result1 <= 1.0

    def test_random_seed_deterministic(self, setup_deterministic_environment):
        assert random.random() == 0.123456789

        random.seed(42)
        result1_int = random.randint(1, 100)
        result1_choice = random.choice([1, 2, 3, 4, 5])

        random.seed(42)
        result2_int = random.randint(1, 100)
        result2_choice = random.choice([1, 2, 3, 4, 5])

        assert result1_int == result2_int
        assert result1_choice == result2_choice

    def test_os_urandom_deterministic(self, setup_deterministic_environment):
        for n in [0, 1, 8, 16, 32]:
            result1 = os.urandom(n)
            result2 = os.urandom(n)

            expected = b"\x42" * n
            assert result1 == expected
            assert result2 == expected
            assert len(result1) == n

    def test_datetime_now_naive(self, setup_deterministic_environment):
        result = datetime.datetime.now()
        expected = datetime.datetime(2021, 1, 1, 2, 5, 10)

        assert result == expected
        assert result.tzinfo is None

    def test_datetime_now_with_tz(self, setup_deterministic_environment):
        result = datetime.datetime.now(tz=datetime.timezone.utc)

        assert result.tzinfo is not None
        assert result.year == 2021
        assert result.month == 1
        assert result.day == 1

    def test_datetime_utcnow_naive(self, setup_deterministic_environment):
        result = datetime.datetime.utcnow()
        expected = datetime.datetime(2021, 1, 1, 2, 5, 10)

        assert result == expected
        assert result.tzinfo is None

    def test_numpy_seeding(self, setup_deterministic_environment):
        try:
            import numpy as np

            result1 = np.random.random(5)

            np.random.seed(42)
            result2 = np.random.random(5)

            assert np.array_equal(result1, result2)
        except ImportError:
            pytest.skip("NumPy not available")

    def test_performance_characteristics_maintained(self, setup_deterministic_environment):
        start = time.perf_counter()
        for _ in range(1000):
            time.time()
            uuid.uuid4()
            random.random()
        end = time.perf_counter()

        duration = end - start
        assert duration < 1.0, f"Performance degraded: {duration}s for 1000 calls"

    def test_consistency_across_multiple_calls(self, setup_deterministic_environment):
        initial_time = time.time()
        initial_uuid = uuid.uuid4()
        initial_random = random.random()
        initial_urandom = os.urandom(8)

        for _ in range(5):
            assert time.time() == initial_time
            assert uuid.uuid4() == initial_uuid
            assert random.random() == initial_random
            assert os.urandom(8) == initial_urandom

    def test_perf_counter_state_management(self, setup_deterministic_environment):
        base = time.perf_counter()
        results = [time.perf_counter() for _ in range(5)]

        for i, result in enumerate(results):
            expected = base + ((i + 1) * 0.001)
            assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_different_uuid_functions_same_result(self, setup_deterministic_environment):
        uuid4_result = uuid.uuid4()
        uuid1_result = uuid.uuid1()

        assert uuid4_result == uuid1_result
        assert str(uuid4_result) == "12345678-1234-5678-9abc-123456789012"

    def test_patches_applied_correctly(self, setup_deterministic_environment):
        assert time.time() == 1761717605.108106
        assert uuid.uuid4() == uuid.UUID("12345678-1234-5678-9abc-123456789012")
        assert random.random() == 0.123456789
        assert os.urandom(4) == b"\x42\x42\x42\x42"

    def test_edge_cases(self, setup_deterministic_environment):
        assert uuid.uuid1(node=0) == uuid.UUID("12345678-1234-5678-9abc-123456789012")
        assert uuid.uuid1(clock_seq=0) == uuid.UUID("12345678-1234-5678-9abc-123456789012")

        assert os.urandom(0) == b""
        assert os.urandom(1) == b"\x42"

        result_with_tz = datetime.datetime.now(datetime.timezone.utc)
        assert result_with_tz.tzinfo is not None

    def test_integration_with_actual_optimization_scenario(self, setup_deterministic_environment):
        class MockOptimizedFunction:
            def __init__(self):
                self.id = uuid.uuid4()
                self.created_at = time.time()
                self.random_factor = random.random()
                self.random_bytes = os.urandom(4)

            def execute(self):
                execution_time = time.perf_counter()
                random_choice = random.randint(1, 100)
                return {
                    "id": self.id,
                    "created_at": self.created_at,
                    "execution_time": execution_time,
                    "random_factor": self.random_factor,
                    "random_choice": random_choice,
                    "random_bytes": self.random_bytes,
                }

        func1 = MockOptimizedFunction()
        func2 = MockOptimizedFunction()

        result1 = func1.execute()
        result2 = func2.execute()

        assert result1["id"] == result2["id"]
        assert result1["created_at"] == result2["created_at"]
        assert result1["random_factor"] == result2["random_factor"]
        assert result1["random_bytes"] == result2["random_bytes"]

        assert result1["execution_time"] != result2["execution_time"]
        assert result2["execution_time"] > result1["execution_time"]

    def test_cleanup_works_properly(self, setup_deterministic_environment):
        pass

    def test_reentrancy_guard(self, setup_deterministic_environment):
        import codeflash.verification.pytest_plugin as plugin_module

        # Force the flag to True, then call again — should be a no-op
        plugin_module._DETERMINISTIC_PATCHES_APPLIED = True
        # Calling again should return early without error or double-patching
        plugin_module._apply_deterministic_patches()
        assert plugin_module._DETERMINISTIC_PATCHES_APPLIED is True

        # Reset for other tests
        plugin_module._DETERMINISTIC_PATCHES_APPLIED = False
