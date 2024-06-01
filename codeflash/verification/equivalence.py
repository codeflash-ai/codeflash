import sys

from codeflash.verification.comparator import comparator
from codeflash.verification.test_results import TestResults

INCREASED_RECURSION_LIMIT = 5000


def compare_test_results(original_results: TestResults, candidate_results: TestResults) -> bool:
    if len(original_results) == 0 or len(candidate_results) == 0:
        return False  # empty test results are not equal
    original_recursion_limit = sys.getrecursionlimit()
    if original_recursion_limit < INCREASED_RECURSION_LIMIT:
        sys.setrecursionlimit(INCREASED_RECURSION_LIMIT)  # Increase recursion limit to avoid RecursionError
    test_ids_superset = set(original_results.get_all_ids()).union(set(candidate_results.get_all_ids()))
    are_equal: bool = True
    did_all_timeout: bool = True
    for test_id in test_ids_superset:
        original_test_result = original_results.get_by_id(test_id)
        cdd_test_results = candidate_results.get_by_id(test_id)
        if cdd_test_results is not None and original_test_result is None:
            continue

        if original_test_result is None or cdd_test_results is None:
            are_equal = False
            break
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        if not comparator(original_test_result.return_value, cdd_test_results.return_value):
            are_equal = False
            break
    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        return False
    return are_equal
