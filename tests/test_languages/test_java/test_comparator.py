"""Tests for Java test result comparison."""

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from codeflash.languages.java.comparator import (
    compare_invocations_directly,
    compare_test_results,
    values_equal,
)
from codeflash.models.models import TestDiffScope

# Skip tests that require Java runtime if Java is not available
requires_java = pytest.mark.skipif(
    shutil.which("java") is None,
    reason="Java not found - skipping Comparator integration tests",
)

# Kryo-serialized bytes for common test values.
# Generated via com.codeflash.Serializer.serialize() from codeflash-java-runtime.
KRYO_INT_1 = bytes([0x02, 0x02])
KRYO_INT_2 = bytes([0x02, 0x04])
KRYO_INT_3 = bytes([0x02, 0x06])
KRYO_INT_4 = bytes([0x02, 0x08])
KRYO_INT_6 = bytes([0x02, 0x0C])
KRYO_INT_42 = bytes([0x02, 0x54])
KRYO_INT_100 = bytes([0x02, 0xC8, 0x01])
KRYO_STR_OLLEH = bytes([0x03, 0x01, 0x6F, 0x6C, 0x6C, 0x65, 0xE8])
KRYO_STR_WRONG = bytes([0x03, 0x01, 0x77, 0x72, 0x6F, 0x6E, 0xE7])
KRYO_STR_RESULT1 = bytes([0x03, 0x01, 0x7B, 0x22, 0x72, 0x65, 0x73, 0x75, 0x6C, 0x74, 0x22, 0x3A, 0x20, 0x31, 0xFD])
KRYO_STR_RESULT2 = bytes([0x03, 0x01, 0x7B, 0x22, 0x72, 0x65, 0x73, 0x75, 0x6C, 0x74, 0x22, 0x3A, 0x20, 0x32, 0xFD])
KRYO_STR_RESULT3 = bytes([0x03, 0x01, 0x7B, 0x22, 0x72, 0x65, 0x73, 0x75, 0x6C, 0x74, 0x22, 0x3A, 0x20, 0x33, 0xFD])
KRYO_STR_VALUE1 = bytes([0x03, 0x01, 0x7B, 0x22, 0x76, 0x61, 0x6C, 0x75, 0x65, 0x22, 0x3A, 0x20, 0x31, 0xFD])
KRYO_STR_VALUE2 = bytes([0x03, 0x01, 0x7B, 0x22, 0x76, 0x61, 0x6C, 0x75, 0x65, 0x22, 0x3A, 0x20, 0x32, 0xFD])
KRYO_STR_VALUE42 = bytes([0x03, 0x01, 0x7B, 0x22, 0x76, 0x61, 0x6C, 0x75, 0x65, 0x22, 0x3A, 0x20, 0x34, 0x32, 0xFD])
KRYO_STR_VALUE100 = bytes([0x03, 0x01, 0x7B, 0x22, 0x76, 0x61, 0x6C, 0x75, 0x65, 0x22, 0x3A, 0x20, 0x31, 0x30, 0x30, 0xFD])
KRYO_DOUBLE_1_0000000001 = bytes([0x0A, 0x38, 0xDF, 0x06, 0x00, 0x00, 0x00, 0xF0, 0x3F])
KRYO_DOUBLE_1_0000000002 = bytes([0x0A, 0x70, 0xBE, 0x0D, 0x00, 0x00, 0x00, 0xF0, 0x3F])
KRYO_NAN = bytes([0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F])
KRYO_INFINITY = bytes([0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F])
KRYO_NEG_INFINITY = bytes([0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xFF])


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


class TestNumericValueEquality:
    """Tests for numeric-aware value comparison."""

    def test_identical_strings(self):
        assert values_equal("0", "0") is True
        assert values_equal("42", "42") is True
        assert values_equal("hello", "hello") is True

    def test_integer_long_equivalence(self):
        assert values_equal("0", "0.0") is True
        assert values_equal("42", "42.0") is True
        assert values_equal("-5", "-5.0") is True

    def test_float_double_equivalence(self):
        assert values_equal("3.14", "3.14") is True
        assert values_equal("3.14", "3.1400000000000001") is True

    def test_nan_handling(self):
        assert values_equal("NaN", "NaN") is True

    def test_infinity_handling(self):
        assert values_equal("Infinity", "Infinity") is True
        assert values_equal("-Infinity", "-Infinity") is True
        assert values_equal("Infinity", "-Infinity") is False

    def test_none_handling(self):
        assert values_equal(None, None) is True
        assert values_equal(None, "0") is False
        assert values_equal("0", None) is False

    def test_non_numeric_strings_differ(self):
        assert values_equal("hello", "world") is False
        assert values_equal("abc", "123") is False

    def test_numeric_comparison_in_direct_invocation(self):
        """Test that compare_invocations_directly uses numeric-aware comparison."""
        original = {
            "1": {"result_json": "0", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "0.0", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_integer_long_mismatch_resolved(self):
        """Test that Integer(42) vs Long(42) serialized differently are still equal."""
        original = {
            "1": {"result_json": "42", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "42.0", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_boolean_string_equality(self):
        """Test that boolean serialized strings compare correctly."""
        assert values_equal("true", "true") is True
        assert values_equal("false", "false") is True
        assert values_equal("true", "false") is False

    def test_boolean_not_numeric(self):
        """Test that boolean strings are not treated as numeric values."""
        assert values_equal("true", "1") is False
        assert values_equal("false", "0") is False

    def test_character_as_int_equality(self):
        """Test that characters serialized as int codepoints compare correctly.

        _cfSerialize converts Character('A') to "65", so both sides should match.
        """
        assert values_equal("65", "65") is True
        assert values_equal("65", "65.0") is True  # int vs float representation
        assert values_equal("65", "66") is False

    def test_array_string_equality(self):
        """Test that array serialized strings compare correctly.

        Arrays.toString produces strings like '[1, 2, 3]' which are compared as strings.
        """
        assert values_equal("[1, 2, 3]", "[1, 2, 3]") is True
        assert values_equal("[1, 2, 3]", "[3, 2, 1]") is False
        assert values_equal("[true, false]", "[true, false]") is True

    def test_array_string_not_numeric(self):
        """Test that array strings are not treated as numeric."""
        assert values_equal("[1, 2]", "12") is False
        assert values_equal("[]", "0") is False

    def test_null_string_equality(self):
        """Test that 'null' strings compare correctly."""
        assert values_equal("null", "null") is True
        assert values_equal("null", "0") is False

    def test_byte_short_int_long_all_equivalent(self):
        """Test that Byte(5), Short(5), Integer(5), Long(5) all serialize equivalently.

        _cfSerialize normalizes all integer Number types to long representation.
        """
        assert values_equal("5", "5") is True
        assert values_equal("5", "5.0") is True
        assert values_equal("-128", "-128.0") is True

    def test_float_double_precision(self):
        """Test float vs double precision differences are handled."""
        assert values_equal("3.14", "3.14") is True
        # Float(3.14f).doubleValue() may give 3.140000104904175
        assert values_equal("3.140000104904175", "3.14") is False  # too far apart
        # But very close values should match
        assert values_equal("1.0000000001", "1.0") is True

    def test_negative_zero(self):
        """Test that -0.0 and 0.0 are treated as equal."""
        assert values_equal("0.0", "-0.0") is True
        assert values_equal("0", "-0.0") is True

    def test_boolean_invocation_comparison(self):
        """Test boolean return values in full invocation comparison."""
        original = {
            "1": {"result_json": "true", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "true", "error_json": None},
        }
        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_boolean_mismatch_invocation_comparison(self):
        """Test boolean mismatch is correctly detected."""
        original = {
            "1": {"result_json": "true", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "false", "error_json": None},
        }
        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1

    def test_array_invocation_comparison(self):
        """Test array return values in full invocation comparison."""
        original = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_array_mismatch_invocation_comparison(self):
        """Test array mismatch is correctly detected."""
        original = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[1, 2, 4]", "error_json": None},
        }
        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1


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


@requires_java
class TestTestResultsTableSchema:
    """Tests for Java Comparator reading from test_results table schema.

    This validates the schema integration between instrumentation (which writes
    to test_results) and the Comparator (which reads from test_results).

    These tests require Java to be installed to run the actual Comparator.jar.
    """

    @pytest.fixture
    def create_test_results_db(self):
        """Create a test SQLite database with test_results table (actual schema used by instrumentation)."""

        def _create(path: Path, results: list[dict]):
            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            # Create test_results table matching instrumentation schema
            cursor.execute(
                """
                CREATE TABLE test_results (
                    test_module_path TEXT,
                    test_class_name TEXT,
                    test_function_name TEXT,
                    function_getting_tested TEXT,
                    loop_index INTEGER,
                    iteration_id TEXT,
                    runtime INTEGER,
                    return_value BLOB,
                    verification_type TEXT
                )
            """
            )

            for result in results:
                cursor.execute(
                    """
                    INSERT INTO test_results
                    (test_module_path, test_class_name, test_function_name,
                     function_getting_tested, loop_index, iteration_id,
                     runtime, return_value, verification_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.get("test_module_path", "TestModule"),
                        result.get("test_class_name", "TestClass"),
                        result.get("test_function_name", "testMethod"),
                        result.get("function_getting_tested", "targetMethod"),
                        result.get("loop_index", 1),
                        result.get("iteration_id", "1_0"),
                        result.get("runtime", 1000000),
                        result.get("return_value"),
                        result.get("verification_type", "function_call"),
                    ),
                )

            conn.commit()
            conn.close()
            return path

        return _create

    def test_comparator_reads_test_results_table_identical(
        self, tmp_path: Path, create_test_results_db
    ):
        """Test that Comparator correctly reads test_results table with identical results."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        # Create databases with identical Kryo-serialized results
        results = [
            {
                "test_class_name": "CalculatorTest",
                "function_getting_tested": "add",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_INT_42,
            },
            {
                "test_class_name": "CalculatorTest",
                "function_getting_tested": "add",
                "loop_index": 1,
                "iteration_id": "2_0",
                "return_value": KRYO_INT_100,
            },
        ]

        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        # Compare using Java Comparator
        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_reads_test_results_table_different_values(
        self, tmp_path: Path, create_test_results_db
    ):
        """Test that Comparator detects different return values from test_results table."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        original_results = [
            {
                "test_class_name": "StringUtilsTest",
                "function_getting_tested": "reverse",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_STR_OLLEH,
            },
        ]

        candidate_results = [
            {
                "test_class_name": "StringUtilsTest",
                "function_getting_tested": "reverse",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_STR_WRONG,  # Different result
            },
        ]

        create_test_results_db(original_path, original_results)
        create_test_results_db(candidate_path, candidate_results)

        # Compare using Java Comparator
        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE

    def test_comparator_handles_multiple_loop_iterations(
        self, tmp_path: Path, create_test_results_db
    ):
        """Test that Comparator correctly handles multiple loop iterations."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        # Simulate multiple benchmark loops with Kryo-serialized integers
        # loop*iteration: 1*1=1, 1*2=2, 2*1=2, 2*2=4, 3*1=3, 3*2=6
        kryo_ints = {1: KRYO_INT_1, 2: KRYO_INT_2, 3: KRYO_INT_3, 4: KRYO_INT_4, 6: KRYO_INT_6}
        results = []
        for loop in range(1, 4):  # 3 loops
            for iteration in range(1, 3):  # 2 iterations per loop
                results.append(
                    {
                        "test_class_name": "AlgorithmTest",
                        "function_getting_tested": "fibonacci",
                        "loop_index": loop,
                        "iteration_id": f"{iteration}_0",
                        "return_value": kryo_ints[loop * iteration],
                    }
                )

        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        # Compare using Java Comparator
        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_iteration_id_parsing(
        self, tmp_path: Path, create_test_results_db
    ):
        """Test that Comparator correctly parses iteration_id format 'iter_testIteration'."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        # Test various iteration_id formats with Kryo-serialized values
        results = [
            {
                "loop_index": 1,
                "iteration_id": "1_0",  # Standard format
                "return_value": KRYO_INT_1,
            },
            {
                "loop_index": 1,
                "iteration_id": "2_5",  # With test iteration
                "return_value": KRYO_INT_2,
            },
            {
                "loop_index": 2,
                "iteration_id": "1_0",  # Different loop
                "return_value": KRYO_INT_3,
            },
        ]

        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        # Compare using Java Comparator
        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_missing_result_in_candidate(
        self, tmp_path: Path, create_test_results_db
    ):
        """Test that Comparator detects missing results in candidate."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        original_results = [
            {
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_INT_1,
            },
            {
                "loop_index": 1,
                "iteration_id": "2_0",
                "return_value": KRYO_INT_2,
            },
        ]

        candidate_results = [
            {
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_INT_1,
            },
            # Missing second iteration
        ]

        create_test_results_db(original_path, original_results)
        create_test_results_db(candidate_path, candidate_results)

        # Compare using Java Comparator
        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) >= 1  # Should detect missing invocation


class TestComparatorEdgeCases:
    """Tests for edge case data types in direct Python comparison path."""

    def test_float_values_identical(self):
        """Float return values that are string-identical should be equivalent."""
        original = {
            "1": {"result_json": "3.14159", "error_json": None},
            "2": {"result_json": "2.71828", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "3.14159", "error_json": None},
            "2": {"result_json": "2.71828", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_float_values_slightly_different(self):
        """Float strings within epsilon tolerance should be considered equivalent.

        The Python comparison uses math.isclose() with rel_tol=1e-9 for numeric values,
        matching the Java Comparator's EPSILON-based tolerance. Values like "3.14159"
        and "3.141590001" differ by ~3e-10, which is within the tolerance and thus
        considered equivalent.

        For truly different values, the difference must exceed the epsilon threshold.
        """
        # These values differ by ~3e-10, which is within epsilon tolerance (1e-9)
        original = {
            "1": {"result_json": "3.14159", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "3.141590001", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True  # Within epsilon tolerance
        assert len(diffs) == 0

    def test_float_values_significantly_different(self):
        """Float strings outside epsilon tolerance should be detected as different."""
        original = {
            "1": {"result_json": "3.14159", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "3.14160", "error_json": None},  # Differs by ~1e-5
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE

    def test_nan_string_comparison(self):
        """NaN as a string return value should be comparable."""
        original = {
            "1": {"result_json": "NaN", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "NaN", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_nan_vs_number(self):
        """NaN vs a normal number should be detected as different."""
        original = {
            "1": {"result_json": "NaN", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "0.0", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1

    def test_infinity_string_comparison(self):
        """Infinity as a string return value should be comparable."""
        original = {
            "1": {"result_json": "Infinity", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "Infinity", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_negative_infinity(self):
        """-Infinity as a string return value should be comparable."""
        original = {
            "1": {"result_json": "-Infinity", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "-Infinity", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_infinity_vs_negative_infinity(self):
        """Infinity and -Infinity should be detected as different."""
        original = {
            "1": {"result_json": "Infinity", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "-Infinity", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1

    def test_empty_collection_results(self):
        """Empty array '[]' as return value should be comparable."""
        original = {
            "1": {"result_json": "[]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[]", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_empty_object_results(self):
        """Empty object '{}' as return value should be comparable."""
        original = {
            "1": {"result_json": "{}", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "{}", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_large_number_comparison(self):
        """Very large integers should compare correctly as strings."""
        original = {
            "1": {"result_json": "99999999999999999", "error_json": None},
            "2": {"result_json": "123456789012345678901234567890", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "99999999999999999", "error_json": None},
            "2": {"result_json": "123456789012345678901234567890", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_large_number_different(self):
        """Very large numbers may lose precision when compared as floats.

        Numbers like 99999999999999999 and 99999999999999998 both convert to
        1e+17 as floats due to precision limits, making them indistinguishable.
        This is a known limitation of floating-point comparison for very large integers.
        """
        original = {
            "1": {"result_json": "99999999999999999", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "99999999999999998", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        # Due to float precision limits, these are considered equal
        assert equivalent is True
        assert len(diffs) == 0

    def test_large_number_significantly_different(self):
        """Large numbers with significant differences should be detected."""
        original = {
            "1": {"result_json": "100000000000000000", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "200000000000000000", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1

    def test_null_vs_empty_string(self):
        """'null' and '""' should NOT be equivalent."""
        original = {
            "1": {"result_json": "null", "error_json": None},
        }
        candidate = {
            "1": {"result_json": '""', "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE

    def test_boolean_string_comparison(self):
        """Boolean strings 'true'/'false' should compare correctly."""
        original = {
            "1": {"result_json": "true", "error_json": None},
            "2": {"result_json": "false", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "true", "error_json": None},
            "2": {"result_json": "false", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True

    def test_boolean_true_vs_false(self):
        """'true' vs 'false' should be detected as different."""
        original = {
            "1": {"result_json": "true", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "false", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1


class TestComparatorErrorHandling:
    """Tests for error handling in comparison paths."""

    def test_compare_empty_databases_both_missing(self, tmp_path: Path):
        """When both SQLite files don't exist, compare_test_results returns (False, [])."""
        original_path = tmp_path / "nonexistent_original.db"
        candidate_path = tmp_path / "nonexistent_candidate.db"

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) == 0

    def test_compare_schema_mismatch_db(self, tmp_path: Path):
        """DB with wrong table name should be handled gracefully (not crash).

        The Java Comparator expects a test_results table. A DB with a different
        schema should result in a (False, []) or error response, not a crash.
        """
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        # Create DBs with wrong table name
        for db_path in [original_path, candidate_path]:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE wrong_table (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("INSERT INTO wrong_table VALUES (1, 'test')")
            conn.commit()
            conn.close()

        # This should not crash -- it either returns (False, []) because Java
        # comparator reports error, or (True, []) if it sees empty test_results.
        # The key assertion is that it doesn't raise an exception.
        equivalent, diffs = compare_test_results(original_path, candidate_path)
        assert isinstance(equivalent, bool)
        assert isinstance(diffs, list)

    def test_compare_with_none_return_values_direct(self):
        """Rows where result_json is None should be handled in direct comparison."""
        original = {
            "1": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": None, "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_compare_one_none_one_value_direct(self):
        """One None result vs a real value should detect the difference."""
        original = {
            "1": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": "42", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1

    def test_compare_both_errors_identical(self):
        """Identical errors in both original and candidate should be equivalent."""
        original = {
            "1": {"result_json": None, "error_json": '{"type": "IOException", "message": "file not found"}'},
        }
        candidate = {
            "1": {"result_json": None, "error_json": '{"type": "IOException", "message": "file not found"}'},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is True
        assert len(diffs) == 0

    def test_compare_different_error_types(self):
        """Different error types should be detected."""
        original = {
            "1": {"result_json": None, "error_json": '{"type": "IOException"}'},
        }
        candidate = {
            "1": {"result_json": None, "error_json": '{"type": "NullPointerException"}'},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)
        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.DID_PASS


@requires_java
class TestComparatorJavaEdgeCases(TestTestResultsTableSchema):
    """Tests for Java Comparator edge cases that require Java runtime.

    Extends TestTestResultsTableSchema to reuse the create_test_results_db fixture.
    """

    def test_comparator_float_epsilon_tolerance(
        self, tmp_path: Path, create_test_results_db
    ):
        """Values differing by less than EPSILON (1e-9) should be treated as equivalent.

        The Java Comparator uses EPSILON=1e-9 for float comparison.
        Values must be Kryo-serialized Double bytes for the Comparator to deserialize and
        apply epsilon-based comparison.
        """
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        original_results = [
            {
                "test_class_name": "MathTest",
                "function_getting_tested": "compute",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_DOUBLE_1_0000000001,
            },
        ]

        candidate_results = [
            {
                "test_class_name": "MathTest",
                "function_getting_tested": "compute",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_DOUBLE_1_0000000002,
            },
        ]

        create_test_results_db(original_path, original_results)
        create_test_results_db(candidate_path, candidate_results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        # The Java Comparator should treat these as equivalent (diff < EPSILON)
        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_nan_handling(
        self, tmp_path: Path, create_test_results_db
    ):
        """Java Comparator should handle NaN return values."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        results = [
            {
                "test_class_name": "MathTest",
                "function_getting_tested": "divide",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_NAN,
            },
        ]

        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        # NaN == NaN should be true in the comparator (special case)
        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_empty_table(
        self, tmp_path: Path, create_test_results_db
    ):
        """Empty test_results tables should result in equivalent=True."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        # Create databases with empty tables (no rows)
        create_test_results_db(original_path, [])
        create_test_results_db(candidate_path, [])

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        # No rows to compare, so they should be equivalent
        assert equivalent is True
        assert len(diffs) == 0

    def test_comparator_infinity_handling(
        self, tmp_path: Path, create_test_results_db
    ):
        """Java Comparator should handle Infinity return values correctly."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        results = [
            {
                "test_class_name": "MathTest",
                "function_getting_tested": "overflow",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": KRYO_INFINITY,
            },
            {
                "test_class_name": "MathTest",
                "function_getting_tested": "underflow",
                "loop_index": 1,
                "iteration_id": "2_0",
                "return_value": KRYO_NEG_INFINITY,
            },
        ]

        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0


class TestVoidFunctionComparison:
    """Tests for void function comparison using compare_invocations_directly."""

    def test_void_both_null_result_equivalent(self):
        """Both original and candidate have None result_json (void success)."""
        original = {
            "1": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": None, "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is True
        assert len(diffs) == 0

    def test_void_null_vs_non_null_result(self):
        """Original void (None) vs candidate with return value should differ."""
        original = {
            "1": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": "42", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE

    def test_void_non_null_vs_null_result(self):
        """Original with return value vs candidate void (None) should differ."""
        original = {
            "1": {"result_json": "42", "error_json": None},
        }
        candidate = {
            "1": {"result_json": None, "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE

    def test_void_same_serialized_side_effects(self):
        """Identical side-effect serializations (Object[] arrays) should be equivalent."""
        original = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is True
        assert len(diffs) == 0

    def test_void_different_serialized_side_effects(self):
        """Different side-effect serializations should be detected."""
        original = {
            "1": {"result_json": "[1, 2, 3]", "error_json": None},
        }
        candidate = {
            "1": {"result_json": "[1, 2, 99]", "error_json": None},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.RETURN_VALUE
        assert diffs[0].original_value == "[1, 2, 3]"
        assert diffs[0].candidate_value == "[1, 2, 99]"

    def test_void_exception_in_candidate(self):
        """Void success in original vs exception in candidate should differ."""
        original = {
            "1": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": None, "error_json": '{"type": "NullPointerException"}'},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 1
        assert diffs[0].scope == TestDiffScope.DID_PASS

    def test_void_multiple_invocations_mixed(self):
        """Multiple void invocations: some matching, some differing."""
        original = {
            "1": {"result_json": None, "error_json": None},
            "2": {"result_json": "[10, 20]", "error_json": None},
            "3": {"result_json": None, "error_json": None},
        }
        candidate = {
            "1": {"result_json": None, "error_json": None},
            "2": {"result_json": "[10, 99]", "error_json": None},
            "3": {"result_json": None, "error_json": '{"type": "RuntimeException"}'},
        }

        equivalent, diffs = compare_invocations_directly(original, candidate)

        assert equivalent is False
        assert len(diffs) == 2
        sorted_diffs = sorted(diffs, key=lambda d: d.scope.value)
        assert sorted_diffs[0].scope == TestDiffScope.DID_PASS
        assert sorted_diffs[1].scope == TestDiffScope.RETURN_VALUE
        assert sorted_diffs[1].original_value == "[10, 20]"
        assert sorted_diffs[1].candidate_value == "[10, 99]"


@requires_java
class TestVoidSqliteComparison:
    """Tests for void function comparison via Java Comparator with 10-column SQLite schema."""

    @pytest.fixture
    def create_void_test_results_db(self):
        """Create a test SQLite database with 10-column schema (including stdout)."""

        def _create(path: Path, results: list[dict]):
            conn = sqlite3.connect(path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE test_results (
                    test_module_path TEXT,
                    test_class_name TEXT,
                    test_function_name TEXT,
                    function_getting_tested TEXT,
                    loop_index INTEGER,
                    iteration_id TEXT,
                    runtime INTEGER,
                    return_value BLOB,
                    verification_type TEXT,
                    stdout TEXT
                )
            """
            )

            for result in results:
                cursor.execute(
                    """
                    INSERT INTO test_results
                    (test_module_path, test_class_name, test_function_name,
                     function_getting_tested, loop_index, iteration_id,
                     runtime, return_value, verification_type, stdout)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.get("test_module_path", "TestModule"),
                        result.get("test_class_name", "TestClass"),
                        result.get("test_function_name", "testMethod"),
                        result.get("function_getting_tested", "targetMethod"),
                        result.get("loop_index", 1),
                        result.get("iteration_id", "1_0"),
                        result.get("runtime", 1000000),
                        result.get("return_value"),
                        result.get("verification_type", "function_call"),
                        result.get("stdout"),
                    ),
                )

            conn.commit()
            conn.close()
            return path

        return _create

    def test_void_sqlite_both_null_return_same_stdout(
        self, tmp_path: Path, create_void_test_results_db
    ):
        """Both DBs have NULL return_value and same stdout — equivalent."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        results = [
            {
                "test_class_name": "PrinterTest",
                "function_getting_tested": "printMessage",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": None,
                "stdout": "Hello World\n",
            },
        ]

        create_void_test_results_db(original_path, results)
        create_void_test_results_db(candidate_path, results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0

    def test_void_sqlite_different_stdout(
        self, tmp_path: Path, create_void_test_results_db
    ):
        """Both DBs have NULL return_value but different stdout — not equivalent."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        original_results = [
            {
                "test_class_name": "LoggerTest",
                "function_getting_tested": "log",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": None,
                "stdout": "INFO: Starting\n",
            },
        ]

        candidate_results = [
            {
                "test_class_name": "LoggerTest",
                "function_getting_tested": "log",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": None,
                "stdout": "DEBUG: Starting\n",
            },
        ]

        create_void_test_results_db(original_path, original_results)
        create_void_test_results_db(candidate_path, candidate_results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert len(diffs) == 1

    def test_void_sqlite_null_stdout_both(
        self, tmp_path: Path, create_void_test_results_db
    ):
        """Both DBs have NULL return_value and NULL stdout — equivalent."""
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        results = [
            {
                "test_class_name": "WorkerTest",
                "function_getting_tested": "doWork",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": None,
                "stdout": None,
            },
        ]

        create_void_test_results_db(original_path, results)
        create_void_test_results_db(candidate_path, results)

        equivalent, diffs = compare_test_results(original_path, candidate_path)

        assert equivalent is True
        assert len(diffs) == 0
