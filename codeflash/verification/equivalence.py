import sys

from codeflash.cli_cmds.console import logger
from codeflash.models.models import TestResults, TestType, VerificationType
from codeflash.verification.comparator import comparator

INCREASED_RECURSION_LIMIT = 5000


def compare_test_results(original_results: TestResults, candidate_results: TestResults) -> bool:
    # This is meant to be only called with test results for the first loop index
    if len(original_results) == 0 or len(candidate_results) == 0:
        return False  # empty test results are not equal
    original_recursion_limit = sys.getrecursionlimit()
    if original_recursion_limit < INCREASED_RECURSION_LIMIT:
        sys.setrecursionlimit(INCREASED_RECURSION_LIMIT)  # Increase recursion limit to avoid RecursionError
    test_ids_superset = original_results.get_all_unique_invocation_loop_ids().union(
        set(candidate_results.get_all_unique_invocation_loop_ids())
    )
    are_equal: bool = True
    did_all_timeout: bool = True
    for test_id in test_ids_superset:
        original_test_result = original_results.get_by_unique_invocation_loop_id(test_id)
        cdd_test_result = candidate_results.get_by_unique_invocation_loop_id(test_id)
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
            are_equal = False
            break
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        superset_obj = False
        if original_test_result.verification_type and (
            original_test_result.verification_type
            in (VerificationType.INIT_STATE_HELPER, VerificationType.INIT_STATE_FTO)
        ):
            superset_obj = True
        if not comparator(original_test_result.return_value, cdd_test_result.return_value, superset_obj=superset_obj):
            are_equal = False
            logger.debug(
                "File Name: %s\n"
                "Test Type: %s\n"
                "Verification Type: %s\n"
                "Invocation ID: %s\n"
                "Original return value: %s\n"
                "Candidate return value: %s\n"
                "-------------------",
                original_test_result.file_name,
                original_test_result.test_type,
                original_test_result.verification_type,
                original_test_result.id,
                original_test_result.return_value,
                cdd_test_result.return_value,
            )
            break
        if (original_test_result.stdout and cdd_test_result.stdout) and not comparator(
            original_test_result.stdout, cdd_test_result.stdout
        ):
            are_equal = False
            break

        if original_test_result.test_type in [TestType.EXISTING_UNIT_TEST, TestType.CONCOLIC_COVERAGE_TEST] and (
            cdd_test_result.did_pass != original_test_result.did_pass
        ):
            are_equal = False
            break
    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        return False
    return are_equal
