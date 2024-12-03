import sys

from codeflash.verification.comparator import comparator
from codeflash.verification.test_results import TestResults, TestType

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

        if original_test_result is None or cdd_test_result is None:
            are_equal = False
            break
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        if not comparator(original_test_result.return_value, cdd_test_result.return_value):
            are_equal = False
            break
        if original_test_result.test_type == TestType.EXISTING_UNIT_TEST and (
            cdd_test_result.did_pass != original_test_result.did_pass
        ):
            are_equal = False
            break
    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        return False
    return are_equal
