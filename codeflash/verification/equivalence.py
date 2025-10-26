import sys

from codeflash.cli_cmds.console import logger
from codeflash.models.models import FunctionTestInvocation, TestResults, TestType, VerificationType
from codeflash.verification.comparator import comparator

INCREASED_RECURSION_LIMIT = 5000


def compare_test_results(original_results: TestResults, candidate_results: TestResults) -> bool:
    # This is meant to be only called with test results for the first loop index
    if len(original_results) == 0 or len(candidate_results) == 0:
        return False  # empty test results are not equal
    original_recursion_limit = sys.getrecursionlimit()
    if original_recursion_limit < INCREASED_RECURSION_LIMIT:
        sys.setrecursionlimit(INCREASED_RECURSION_LIMIT)  # Increase recursion limit to avoid RecursionError

    # Separate Hypothesis tests from other test types for semantic comparison
    # Hypothesis tests are always compared semantically (by test function, not example count)
    original_hypothesis = [
        r for r in original_results.test_results if r.test_type == TestType.HYPOTHESIS_TEST and r.loop_index == 1
    ]
    candidate_hypothesis = [
        r for r in candidate_results.test_results if r.test_type == TestType.HYPOTHESIS_TEST and r.loop_index == 1
    ]

    # Compare Hypothesis tests semantically if any are present
    if original_hypothesis or candidate_hypothesis:
        logger.debug(
            f"Comparing Hypothesis tests: original={len(original_hypothesis)} examples, "
            f"candidate={len(candidate_hypothesis)} examples"
        )
        hypothesis_equal = _compare_hypothesis_tests_semantic(original_hypothesis, candidate_hypothesis)
        if not hypothesis_equal:
            logger.info("Hypothesis comparison failed")
            sys.setrecursionlimit(original_recursion_limit)
            return False
        logger.debug("Hypothesis comparison passed")

    test_ids_superset = original_results.get_all_unique_invocation_loop_ids().union(
        set(candidate_results.get_all_unique_invocation_loop_ids())
    )
    logger.debug(f"Total test IDs in superset: {len(test_ids_superset)}")
    are_equal: bool = True
    did_all_timeout: bool = True
    for test_id in test_ids_superset:
        original_test_result = original_results.get_by_unique_invocation_loop_id(test_id)
        cdd_test_result = candidate_results.get_by_unique_invocation_loop_id(test_id)

        # Skip Hypothesis tests - already compared semantically above
        if original_test_result and original_test_result.test_type == TestType.HYPOTHESIS_TEST:
            did_all_timeout = False  # Hypothesis tests are checked separately, not timed out
            continue
        if cdd_test_result and cdd_test_result.test_type == TestType.HYPOTHESIS_TEST:
            did_all_timeout = False
            continue

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
            logger.debug(
                f"Test result mismatch: test_id={test_id}, "
                f"original_present={original_test_result is not None}, "
                f"candidate_present={cdd_test_result is not None}"
            )
            break
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        superset_obj = False
        if original_test_result.verification_type and (
            original_test_result.verification_type
            in {VerificationType.INIT_STATE_HELPER, VerificationType.INIT_STATE_FTO}
        ):
            superset_obj = True
        if not comparator(original_test_result.return_value, cdd_test_result.return_value, superset_obj=superset_obj):
            are_equal = False
            try:
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
            except Exception as e:
                logger.error(e)
            break
        if (original_test_result.stdout and cdd_test_result.stdout) and not comparator(
            original_test_result.stdout, cdd_test_result.stdout
        ):
            are_equal = False
            break

        if original_test_result.test_type in {
            TestType.EXISTING_UNIT_TEST,
            TestType.CONCOLIC_COVERAGE_TEST,
            TestType.GENERATED_REGRESSION,
            TestType.REPLAY_TEST,
        } and (cdd_test_result.did_pass != original_test_result.did_pass):
            are_equal = False
            break
    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        logger.debug("Comparison failed: all tests timed out")
        return False
    logger.debug(f"Final comparison result: are_equal={are_equal}")
    return are_equal


def _compare_hypothesis_tests_semantic(original_hypothesis: list, candidate_hypothesis: list) -> bool:
    """Compare Hypothesis tests by test function, not by example count.

    Hypothesis can generate different numbers of examples between runs due to:
    - Timing differences
    - Early stopping
    - Shrinking behavior
    - Performance differences

    What matters is whether the test functions themselves pass or fail,
    not how many examples Hypothesis generated.
    """

    def get_test_key(test_result: FunctionTestInvocation) -> tuple[str, str, str, str]:
        """Get unique key for a Hypothesis test function."""
        return (
            test_result.id.test_module_path,
            test_result.id.test_class_name,
            test_result.id.test_function_name,
            test_result.id.function_getting_tested,
        )

    # Group by test function and simultaneously collect failure flag and example count
    orig_by_func = {}
    for result in original_hypothesis:
        test_key = get_test_key(result)
        group = orig_by_func.setdefault(test_key, [0, False])  # [count, had_failure]
        group[0] += 1
        if not result.did_pass:
            group[1] = True

    cand_by_func = {}
    for result in candidate_hypothesis:
        test_key = get_test_key(result)
        group = cand_by_func.setdefault(test_key, [0, False])  # [count, had_failure]
        group[0] += 1
        if not result.did_pass:
            group[1] = True

    orig_total_examples = sum(group[0] for group in orig_by_func.values())
    cand_total_examples = sum(group[0] for group in cand_by_func.values())

    logger.debug(
        f"Hypothesis comparison: Original={len(orig_by_func)} test functions ({orig_total_examples} examples), "
        f"Candidate={len(cand_by_func)} test functions ({cand_total_examples} examples)"
    )

    # Compare only for test_keys present in original
    for test_key, (orig_count, orig_had_failure) in orig_by_func.items():
        cand_group = cand_by_func.get(test_key)
        if cand_group is None:
            continue  # Already handled above

        cand_had_failure = cand_group[1]

        if orig_had_failure != cand_had_failure:
            logger.debug(
                f"Hypothesis test function behavior mismatch: {test_key} "
                f"(original_failed={orig_had_failure}, candidate_failed={cand_had_failure})"
            )
            return False
    return True
