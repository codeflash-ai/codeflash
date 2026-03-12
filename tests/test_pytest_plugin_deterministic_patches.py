"""Test the deterministic patching functionality in pytest_plugin.py.

Verifies that _apply_deterministic_patches() seeds RNGs for reproducibility
and does NOT patch time, uuid, or os.urandom (which caused false failures).
"""

import os
import random
import time
import uuid

import pytest


class TestDeterministicPatches:
    """Test suite for deterministic patching functionality."""

    @pytest.fixture(autouse=True)
    def apply_patches(self):
        """Apply deterministic patches and clean up afterward."""
        from codeflash.verification.pytest_plugin import _apply_deterministic_patches

        # Save random state so we don't affect other tests
        old_state = random.getstate()
        _apply_deterministic_patches()
        yield
        random.setstate(old_state)

    def test_random_seed_deterministic(self):
        """random.seed(42) produces a deterministic sequence."""
        random.seed(42)
        seq1 = [random.random() for _ in range(5)]

        random.seed(42)
        seq2 = [random.random() for _ in range(5)]

        assert seq1 == seq2

    def test_random_produces_distinct_values(self):
        """Regression: random.random() must NOT return the same value every call."""
        random.seed(42)
        values = [random.random() for _ in range(5)]
        assert len(set(values)) == 5, f"Expected 5 distinct values, got {values}"

    def test_numpy_seed_deterministic(self):
        """np.random.seed(42) produces a deterministic sequence."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("NumPy not available")

        np.random.seed(42)
        seq1 = np.random.random(5).tolist()

        np.random.seed(42)
        seq2 = np.random.random(5).tolist()

        assert seq1 == seq2

    def test_time_not_patched(self):
        """Regression: time.time must NOT be replaced with a mock."""
        import types

        # Real time.time is a C builtin_function_or_method, not a Python function
        assert not isinstance(time.time, types.FunctionType), "time.time appears to be mocked"

    def test_uuid4_not_patched(self):
        """Regression: uuid.uuid4 must NOT be replaced with a mock."""
        u1 = uuid.uuid4()
        u2 = uuid.uuid4()
        assert u1 != u2, "uuid.uuid4 appears to be mocked to a fixed value"

    def test_os_urandom_not_patched(self):
        """Regression: os.urandom must NOT be replaced with a mock."""
        b1 = os.urandom(16)
        b2 = os.urandom(16)
        assert b1 != b2, "os.urandom appears to be mocked to fixed bytes"
