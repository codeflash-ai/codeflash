"""Tests for the comparison decision logic in function_optimizer.py.

Validates the routing between:
1. SQLite-based comparison (via language_support.compare_test_results) when both
   original and candidate SQLite files exist
2. pass_fail_only fallback (via equivalence.compare_test_results with pass_fail_only=True)
   when SQLite files are missing

Also validates the Python equivalence.compare_test_results behavior with pass_fail_only
flag to ensure the fallback path works correctly.
"""

import inspect
import logging
import sqlite3
from dataclasses import replace
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


class TestPassFailOnlyWarningLogging:
    """Tests that pass_fail_only=True logs warnings when differences are silently ignored."""

    def test_pass_fail_only_logs_warning_on_return_value_difference(self, caplog):
        """When pass_fail_only=True and return values differ, a warning is logged."""
        original = make_test_results([
            make_invocation(iteration_id="1_0", did_pass=True, return_value=42),
        ])
        candidate = make_test_results([
            make_invocation(iteration_id="1_0", did_pass=True, return_value=999),
        ])

        with caplog.at_level(logging.WARNING, logger="rich"):
            match, diffs = python_compare_test_results(original, candidate, pass_fail_only=True)

        assert match is True
        assert len(diffs) == 0
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("pass_fail_only mode" in msg and "return value" in msg for msg in warning_messages), (
            f"Expected warning about pass_fail_only ignoring return value difference, got: {warning_messages}"
        )

    def test_pass_fail_only_no_warning_when_values_match(self, caplog):
        """When pass_fail_only=True and return values are the same, no warning is logged."""
        original = make_test_results([
            make_invocation(iteration_id="1_0", did_pass=True, return_value=42),
        ])
        candidate = make_test_results([
            make_invocation(iteration_id="1_0", did_pass=True, return_value=42),
        ])

        with caplog.at_level(logging.WARNING, logger="rich"):
            match, diffs = python_compare_test_results(original, candidate, pass_fail_only=True)

        assert match is True
        assert len(diffs) == 0
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert not any("pass_fail_only mode" in msg for msg in warning_messages), (
            f"No warning expected when values match, got: {warning_messages}"
        )

    def test_pass_fail_only_logs_warning_on_stdout_difference(self, caplog):
        """When pass_fail_only=True and stdout differs, a warning is logged."""
        orig_inv = make_invocation(iteration_id="1_0", did_pass=True, return_value=42)
        cand_inv = make_invocation(iteration_id="1_0", did_pass=True, return_value=42)
        original = make_test_results([replace(orig_inv, stdout="original output")])
        candidate = make_test_results([replace(cand_inv, stdout="candidate output")])

        with caplog.at_level(logging.WARNING, logger="rich"):
            match, diffs = python_compare_test_results(original, candidate, pass_fail_only=True)

        assert match is True
        assert len(diffs) == 0
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("pass_fail_only mode" in msg and "stdout" in msg for msg in warning_messages), (
            f"Expected warning about pass_fail_only ignoring stdout difference, got: {warning_messages}"
        )
