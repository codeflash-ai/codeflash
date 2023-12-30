from codeflash.verification.comparator import comparator

from codeflash.verification.test_results import TestResults


def compare_results(original_result: TestResults, test_result: TestResults) -> bool:
    if len(original_result) == 0 or len(test_result) == 0:
        return False
    test_ids_superset = set(original_result.get_all_ids()).union(set(test_result.get_all_ids()))
    for test_id in test_ids_superset:
        original_test_result = original_result.get_by_id(test_id)
        test_test_result = test_result.get_by_id(test_id)
        if original_test_result is None or test_test_result is None:
            return False
        if not comparator(original_test_result.return_value, test_test_result.return_value):
            return False
    return True
