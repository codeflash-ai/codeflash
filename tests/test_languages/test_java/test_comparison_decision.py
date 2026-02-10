"""Tests for the comparison decision logic in function_optimizer.py.

Validates SQLite-based comparison (via language_support.compare_test_results) when both
original and candidate SQLite files exist. If SQLite files are missing, optimization will
fail with an error to maintain strict correctness guarantees.
"""

import inspect
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from codeflash.languages.java.comparator import (
    compare_test_results as java_compare_test_results,
)
from codeflash.models.models import (
    FunctionTestInvocation,
    InvocationId,
    TestDiffScope,
    TestResults,
    TestType,
    VerificationType,
)
from codeflash.verification.equivalence import (
    compare_test_results as python_compare_test_results,
)


def make_invocation(
    test_module_path: str = "test_module",
    test_class_name: str = "TestClass",
    test_function_name: str = "test_method",
    function_getting_tested: str = "target_method",
    iteration_id: str = "1_0",
    loop_index: int = 1,
    did_pass: bool = True,
    return_value: object = 42,
    runtime: int = 1000,
    timed_out: bool = False,
) -> FunctionTestInvocation:
    """Helper to create a FunctionTestInvocation for testing."""
    return FunctionTestInvocation(
        loop_index=loop_index,
        id=InvocationId(
            test_module_path=test_module_path,
            test_class_name=test_class_name,
            test_function_name=test_function_name,
            function_getting_tested=function_getting_tested,
            iteration_id=iteration_id,
        ),
        file_name=Path("test_file.py"),
        did_pass=did_pass,
        runtime=runtime,
        test_framework="pytest",
        test_type=TestType.EXISTING_UNIT_TEST,
        return_value=return_value,
        timed_out=timed_out,
        verification_type=VerificationType.FUNCTION_CALL,
    )


def make_test_results(invocations: list[FunctionTestInvocation]) -> TestResults:
    """Helper to create a TestResults object from a list of invocations."""
    results = TestResults()
    for inv in invocations:
        results.add(inv)
    return results


class TestSqlitePathSelection:
    """Tests for SQLite file existence checks in the Java comparison path.

    These validate that compare_test_results from codeflash.languages.java.comparator
    handles file existence correctly, which is the precondition for the SQLite
    comparison path at function_optimizer.py:2822.
    """

    @pytest.fixture
    def create_test_results_db(self):
        """Create a test SQLite database with test_results table."""

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
                    return_value TEXT,
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

    def test_sqlite_files_exist_returns_tuple(self, tmp_path: Path, create_test_results_db):
        """When both SQLite files exist with valid schema, compare_test_results returns (bool, list) tuple.

        This validates the precondition for the SQLite comparison path at
        function_optimizer.py:2822-2828.
        """
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "candidate.db"

        results = [
            {
                "test_class_name": "DecisionTest",
                "function_getting_tested": "compute",
                "loop_index": 1,
                "iteration_id": "1_0",
                "return_value": '{"value": 42}',
            },
        ]
        create_test_results_db(original_path, results)
        create_test_results_db(candidate_path, results)

        result = java_compare_test_results(original_path, candidate_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        equivalent, diffs = result
        assert isinstance(equivalent, bool)
        assert isinstance(diffs, list)

    def test_sqlite_file_missing_original_returns_false(self, tmp_path: Path, create_test_results_db):
        """When original SQLite file doesn't exist, returns (False, []).

        This confirms the guard at comparator.py:129-130. In the decision logic,
        this would mean the code falls through because original_sqlite.exists()
        returns False at function_optimizer.py:2822.
        """
        original_path = tmp_path / "nonexistent_original.db"
        candidate_path = tmp_path / "candidate.db"
        create_test_results_db(candidate_path, [{"return_value": "42"}])

        equivalent, diffs = java_compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert diffs == []

    def test_sqlite_file_missing_candidate_returns_false(self, tmp_path: Path, create_test_results_db):
        """When candidate SQLite file doesn't exist, returns (False, []).

        This confirms the guard at comparator.py:133-134.
        """
        original_path = tmp_path / "original.db"
        candidate_path = tmp_path / "nonexistent_candidate.db"
        create_test_results_db(original_path, [{"return_value": "42"}])

        equivalent, diffs = java_compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert diffs == []

    def test_sqlite_file_missing_both_returns_false(self, tmp_path: Path):
        """When neither SQLite file exists, returns (False, []).

        Both guards fire: original check at comparator.py:129, so candidate
        check is never reached.
        """
        original_path = tmp_path / "nonexistent_original.db"
        candidate_path = tmp_path / "nonexistent_candidate.db"

        equivalent, diffs = java_compare_test_results(original_path, candidate_path)

        assert equivalent is False
        assert diffs == []


class TestDecisionPointDocumentation:
    """Canary tests that validate the decision logic code pattern exists.

    If someone refactors the comparison decision point in function_optimizer.py,
    these tests will alert us so we can update our understanding.
    """

    def test_decision_point_exists_in_function_optimizer(self):
        """Verify the decision logic pattern exists in function_optimizer.py source.

        The comparison decision at lines ~2816-2836 checks:
        1. if not is_python() -> enters non-Python path
        2. original_sqlite.exists() and candidate_sqlite.exists() -> SQLite path
        3. else -> fail with error (strict correctness)

        This is a canary test: if the pattern is refactored, this test fails
        to alert that the routing logic has changed.
        """
        import codeflash.optimization.function_optimizer as fo_module

        source = inspect.getsource(fo_module)

        # Verify the non-Python branch exists
        assert "if not is_python():" in source, (
            "Decision point 'if not is_python():' not found in function_optimizer.py. "
            "The comparison routing logic may have been refactored."
        )

        # Verify SQLite file existence check
        assert "original_sqlite.exists()" in source, (
            "SQLite existence check 'original_sqlite.exists()' not found. "
            "The SQLite comparison routing may have been refactored."
        )

        # Verify the SQLite file naming pattern
        assert "test_return_values_0.sqlite" in source, (
            "SQLite file naming pattern 'test_return_values_0.sqlite' not found. "
            "The SQLite file naming convention may have changed."
        )

    def test_java_comparator_import_path(self):
        """Verify the Java comparator module is importable at the expected path.

        The language_support.compare_test_results call at function_optimizer.py:2826
        resolves to codeflash.languages.java.comparator.compare_test_results for Java.
        """
        from codeflash.languages.java.comparator import compare_test_results

        assert callable(compare_test_results)

    def test_python_equivalence_import_path(self):
        """Verify the Python equivalence module is importable.

        Python uses equivalence.compare_test_results for behavioral verification.
        """
        from codeflash.verification.equivalence import compare_test_results

        assert callable(compare_test_results)
