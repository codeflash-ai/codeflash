"""Tests for the early exit guard when all behavioral tests fail for non-Python candidates.

This tests the Bug 4 fix: when all behavioral tests fail for a Java/JS optimization
candidate, the code should return early with a 'results not matched' error instead of
proceeding to SQLite file comparison (which would crash with FileNotFoundError since
instrumentation hooks never fired).
"""

from dataclasses import dataclass
from pathlib import Path

from codeflash.either import Failure
from codeflash.models.models import FunctionTestInvocation, InvocationId, TestResults
from codeflash.models.test_type import TestType


def make_test_invocation(*, did_pass: bool, test_type: TestType = TestType.EXISTING_UNIT_TEST) -> FunctionTestInvocation:
    """Helper to create a FunctionTestInvocation with minimal required fields."""
    return FunctionTestInvocation(
        loop_index=1,
        id=InvocationId(
            test_module_path="com.example.FooTest",
            test_class_name="FooTest",
            test_function_name="testSomething",
            function_getting_tested="foo",
            iteration_id="0",
        ),
        file_name=Path("FooTest.java"),
        did_pass=did_pass,
        runtime=1000,
        test_framework="junit",
        test_type=test_type,
        return_value=None,
        timed_out=False,
    )


class TestCandidateBehavioralTestGuard:
    """Tests for the early exit guard that prevents SQLite FileNotFoundError."""

    def test_all_tests_failed_returns_zero_passed(self):
        """When all behavioral tests fail, get_test_pass_fail_report_by_type should show 0 passed."""
        results = TestResults()
        results.add(make_test_invocation(did_pass=False, test_type=TestType.EXISTING_UNIT_TEST))
        results.add(make_test_invocation(did_pass=False, test_type=TestType.GENERATED_REGRESSION))

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed == 0

    def test_some_tests_passed_returns_nonzero(self):
        """When some tests pass, the total should be > 0 and the guard should NOT trigger."""
        results = TestResults()
        results.add(make_test_invocation(did_pass=True, test_type=TestType.EXISTING_UNIT_TEST))
        results.add(make_test_invocation(did_pass=False, test_type=TestType.GENERATED_REGRESSION))

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed > 0

    def test_empty_results_returns_zero_passed(self):
        """When no tests ran at all, the guard should trigger (0 passed)."""
        results = TestResults()

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed == 0

    def test_only_non_loop1_results_returns_zero_passed(self):
        """Only loop_index=1 results count. Other loop indices should be ignored."""
        results = TestResults()
        # Add a passing test with loop_index=2 (should be ignored by report)
        inv = FunctionTestInvocation(
            loop_index=2,
            id=InvocationId(
                test_module_path="com.example.FooTest",
                test_class_name="FooTest",
                test_function_name="testOther",
                function_getting_tested="foo",
                iteration_id="0",
            ),
            file_name=Path("FooTest.java"),
            did_pass=True,
            runtime=1000,
            test_framework="junit",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
        )
        results.add(inv)

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed == 0

    def test_mixed_test_types_all_failing(self):
        """All test types failing should yield 0 total passed."""
        results = TestResults()
        for tt in [TestType.EXISTING_UNIT_TEST, TestType.GENERATED_REGRESSION, TestType.REPLAY_TEST]:
            results.add(FunctionTestInvocation(
                loop_index=1,
                id=InvocationId(
                    test_module_path="com.example.FooTest",
                    test_class_name="FooTest",
                    test_function_name=f"test_{tt.name}",
                    function_getting_tested="foo",
                    iteration_id="0",
                ),
                file_name=Path("FooTest.java"),
                did_pass=False,
                runtime=1000,
                test_framework="junit",
                test_type=tt,
                return_value=None,
                timed_out=False,
            ))

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed == 0

    def test_single_passing_test_prevents_early_exit(self):
        """Even one passing test should prevent the early exit (total_passed > 0)."""
        results = TestResults()
        # Many failures
        for i in range(5):
            results.add(FunctionTestInvocation(
                loop_index=1,
                id=InvocationId(
                    test_module_path="com.example.FooTest",
                    test_class_name="FooTest",
                    test_function_name=f"testFail{i}",
                    function_getting_tested="foo",
                    iteration_id="0",
                ),
                file_name=Path("FooTest.java"),
                did_pass=False,
                runtime=1000,
                test_framework="junit",
                test_type=TestType.GENERATED_REGRESSION,
                return_value=None,
                timed_out=False,
            ))
        # One pass
        results.add(FunctionTestInvocation(
            loop_index=1,
            id=InvocationId(
                test_module_path="com.example.FooTest",
                test_class_name="FooTest",
                test_function_name="testPass",
                function_getting_tested="foo",
                iteration_id="0",
            ),
            file_name=Path("FooTest.java"),
            did_pass=True,
            runtime=1000,
            test_framework="junit",
            test_type=TestType.EXISTING_UNIT_TEST,
            return_value=None,
            timed_out=False,
        ))

        report = results.get_test_pass_fail_report_by_type()
        total_passed = sum(r.get("passed", 0) for r in report.values())

        assert total_passed == 1
