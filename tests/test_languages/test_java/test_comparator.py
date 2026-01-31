"""Tests for Java test result comparison."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.java.comparator import (
    compare_invocations_directly,
    compare_test_results,
)
from codeflash.models.models import TestDiffScope


class TestDirectComparison:
    """Tests for direct Python-based comparison."""

    def test_identical_results(self):
        """Test comparing identical results."""
        original = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
            "2": {"result_json": '{"value": 100}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
            "2": {"result_json": '{"value": 100}', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is True
        assert len(diffs) == 0

    def test_different_return_values(self):
        """Test detecting different return values."""
        original = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{"value": 99}', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE
        assert diffs[0].original_value == '{"value": 42}'
        assert diffs[0].candidate_value == '{"value": 99}'

    def test_missing_invocation_in_candidate(self):
        """Test detecting missing invocation in candidate."""
        original = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
            "2": {"result_json": '{"value": 100}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
            # Missing invocation 2
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].candidate_pass is False

    def test_extra_invocation_in_candidate(self):
        """Test detecting extra invocation in candidate."""
        original = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{"value": 42}', "error_json": None},
            "2": {"result_json": '{"value": 100}', "error_json": None},  # Extra
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        # Having extra invocations is noted but doesn't necessarily fail
        assert len(diffs) == 1

    def test_exception_differences(self):
        """Test detecting exception differences."""
        original = {
            "1": {"result_json": None, "error_json": '{"type": "NullPointerException"}'},
        }
        candidate = {
            "1": {"result_json": '{"value": 42}', "error_json": None},  # No exception
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.DID_PASS

    def test_empty_results(self):
        """Test comparing empty results."""
        original = {}
        candidate = {}

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is True
        assert len(diffs) == 0


class TestSqliteComparison:
    """Tests for SQLite-based comparison (requires Java runtime)."""

    @pytest.fixture
    def create_test_db(self):
        """Create a test SQLite database with invocations table."""

        def _create(path: Path, invocations: list[dict]):
            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE invocations (
                    call_id INTEGER PRIMARY KEY,
                    method_id TEXT NOT NULL,
                    args_json TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    start_time INTEGER,
                    end_time INTEGER
                )
            """
            )

            for inv in invocations:
                cursor.execute(
                    """
                    INSERT INTO invocations (call_id, method_id, args_json, result_json, error_json)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        inv.get("call_id"),
                        inv.get("method_id", "test.method"),
                        inv.get("args_json"),
                        inv.get("result_json"),
                        inv.get("error_json"),
                    ),
                )

            conn.commit()
            conn.close()
            return path

        return _create

    def test_compare_test_results_missing_original(self, tmp_path: Path):
        """Test comparison when original DB is missing."""
        original_path = tmp_path / "original.db"  # Doesn't exist
        candidate_path = tmp_path / "candidate.db"
        candidate_path.touch()

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) == 0

    def test_compare_test_results_missing_candidate(self, tmp_path: Path):
        """Test comparison when candidate DB is missing."""
        original_path = tmp_path / "original.db"
        original_path.touch()
        candidate_path = tmp_path / "candidate.db"  # Doesn't exist

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) == 0


class TestComparisonWithRealData:
    """Tests simulating real comparison scenarios."""

    def test_string_result_comparison(self):
        """Test comparing string results."""
        original = {
            "1": {"result_json": '"Hello World"', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '"Hello World"', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_array_result_comparison(self):
        """Test comparing array results."""
        original = {
            "1": {"result_json": "[1, 2, 3, 4, 5]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[1, 2, 3, 4, 5]", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_array_order_matters(self):
        """Test that array order matters for comparison."""
        original = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[3, 2, 1]", "error_json": None},  # Different order
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False

    def test_object_result_comparison(self):
        """Test comparing object results."""
        original = {
            "1": {"result_json": '{"name": "John", "age": 30}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{"name": "John", "age": 30}', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_null_result(self):
        """Test comparing null results."""
        original = {
            "1": {"result_json": "null", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "null", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_multiple_invocations_mixed(self):
        """Test multiple invocations with mixed results."""
        original = {
            "1": {"result_json": "42", "error_json": None},
            "2": {"result_json": '"hello"', "error_json": None},
            "3": {"result_json": None, "error_json": '{"type": "Exception"}'},
        }
        candidate = {
            "1": {"result_json": "42", "error_json": None},
            "2": {"result_json": '"hello"', "error_json": None},
            "3": {"result_json": None, "error_json": '{"type": "Exception"}'},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_whitespace_in_json(self):
        """Test that whitespace differences in JSON don't cause issues."""
        original = {
            "1": {"result_json": '{"a":1,"b":2}', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '{ "a": 1, "b": 2 }', "error_json": None},  # With spaces
        }

        # Note: Direct string comparison will see these as different
        # The Java comparator would handle this correctly by parsing JSON
        equivalent, diffs = compare_invocations_directly(original, candidate)
        # This will fail with direct comparison - expected behavior
        assert equivalent is False  # String comparison doesn't normalize whitespace

    def test_large_number_of_invocations(self):
        """Test handling large number of invocations."""
        original = {str(i): {"result_json": str(i), "error_json": None} for i in range(1000)}
        candidate = {str(i): {"result_json": str(i), "error_json": None} for i in range(1000)}

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_unicode_in_results(self):
        """Test handling unicode in results."""
        original = {
            "1": {"result_json": '"Hello 世界 🌍"', "error_json": None},
        }
        candidate = {
            "1": {"result_json": '"Hello 世界 🌍"', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_deeply_nested_objects(self):
        """Test handling deeply nested objects."""
        nested = '{"a": {"b": {"c": {"d": {"e": 1}}}}}'
        original = {
            "1": {"result_json": nested, "error_json": None},
        }
        candidate = {
            "1": {"result_json": nested, "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
