from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from codeflash.cli_cmds.console import logger
from codeflash.models.models import TestResults, TestType, VerificationType
from codeflash.verification.comparator import comparator

if TYPE_CHECKING:
    from codeflash.models.models import TestResults

INCREASED_RECURSION_LIMIT = 5000


class TestDiffScope(Enum):
    RETURN_VALUE = "return_value"
    STDOUT = "stdout"
    TIMED_OUT = "timed_out"
    DID_PASS = "did_pass"  # noqa: S105


@dataclass
class TestDiff:
    scope: TestDiffScope
    pytest_error: str
    original_value: any
    candidate_value: any
    test_src_code: Optional[str] = None


def compare_test_results(original_results: TestResults, candidate_results: TestResults) -> tuple[bool, list[TestDiff]]:
    # This is meant to be only called with test results for the first loop index
    if len(original_results) == 0 or len(candidate_results) == 0:
        return False, []  # empty test results are not equal
    original_recursion_limit = sys.getrecursionlimit()
    if original_recursion_limit < INCREASED_RECURSION_LIMIT:
        sys.setrecursionlimit(INCREASED_RECURSION_LIMIT)  # Increase recursion limit to avoid RecursionError
    test_ids_superset = original_results.get_all_unique_invocation_loop_ids().union(
        set(candidate_results.get_all_unique_invocation_loop_ids())
    )
    test_diffs: list[TestDiff] = []
    did_all_timeout: bool = True
    for test_id in test_ids_superset:
        original_test_result = original_results.get_by_unique_invocation_loop_id(test_id)
        cdd_test_result = candidate_results.get_by_unique_invocation_loop_id(test_id)
        candidate_test_failures = candidate_results.test_failures
        # original_test_failures = original_results.test_failures
        cdd_pytest_error = (
            candidate_test_failures.get(original_test_result.id.test_function_name, "")
            if candidate_test_failures
            else ""
        )
        # original_pytest_error = (
        #     original_test_failures.get(original_test_result.id.test_function_name, "") if original_test_failures else ""
        # )

        if cdd_test_result is not None and original_test_result is None:
            continue
        # If helper function instance_state verification is not present, that's ok. continue
        if (
            original_test_result.verification_type
            and original_test_result.verification_type == VerificationType.INIT_STATE_HELPER
            and cdd_test_result is None
        ):
            continue
        if original_test_result is None or cdd_test_result is None:
            return False, []
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        superset_obj = False
        if original_test_result.verification_type and (
            original_test_result.verification_type
            in {VerificationType.INIT_STATE_HELPER, VerificationType.INIT_STATE_FTO}
        ):
            superset_obj = True
        test_src_code = original_test_result.id.get_src_code(original_test_result.file_name)
        if not comparator(original_test_result.return_value, cdd_test_result.return_value, superset_obj=superset_obj):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.RETURN_VALUE,
                    test_src_code=test_src_code,
                    original_value=original_test_result.return_value,
                    candidate_value=cdd_test_result.return_value,
                    pytest_error=cdd_pytest_error,
                )
            )

            try:
                print(
                    f"File Name: {original_test_result.file_name}\n"
                    f"Test Type: {original_test_result.test_type}\n"
                    f"Verification Type: {original_test_result.verification_type}\n"
                    f"Invocation ID: {original_test_result.id}\n"
                    f"Original return value: {original_test_result.return_value}\n"
                    f"Candidate return value: {cdd_test_result.return_value}\n"
                )
            except Exception as e:
                logger.error(e)
            break
        if (original_test_result.stdout and cdd_test_result.stdout) and not comparator(
            original_test_result.stdout, cdd_test_result.stdout
        ):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.STDOUT,
                    test_src_code=test_src_code,
                    original_value=original_test_result.stdout,
                    candidate_value=cdd_test_result.stdout,
                    pytest_error=cdd_pytest_error,
                )
            )
            break

        if original_test_result.test_type in {
            TestType.EXISTING_UNIT_TEST,
            TestType.CONCOLIC_COVERAGE_TEST,
            TestType.GENERATED_REGRESSION,
            TestType.REPLAY_TEST,
        } and (cdd_test_result.did_pass != original_test_result.did_pass):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.DID_PASS,
                    test_src_code=test_src_code,
                    original_value=original_test_result.did_pass,
                    candidate_value=cdd_test_result.did_pass,
                    pytest_error=cdd_pytest_error,
                )
            )
            break
    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        return False, test_diffs
    return len(test_diffs) == 0, test_diffs
