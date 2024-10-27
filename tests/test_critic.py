import os

from codeflash.code_utils.env_utils import get_pr_number
from codeflash.models.models import OptimizedCandidateResult
from codeflash.result.critic import quantity_of_tests_critic, speedup_critic
from codeflash.verification.test_results import FunctionTestInvocation, InvocationId, TestResults, TestType


def test_speedup_critic():
    original_code_runtime = 1000
    best_runtime_until_now = 1000
    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=800,
        best_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=12,
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now)  # 20% improvement

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=940,
        best_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now)  # 6% improvement

    original_code_runtime = 100000
    best_runtime_until_now = 100000

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=94000,
        best_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now)  # 6% improvement


def test_generated_test_critic():
    test_1 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_1",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name="test_1",
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.GENERATED_REGRESSION,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_2 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_2",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name="test_2",
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.GENERATED_REGRESSION,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_3 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_3",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name="test_3",
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.EXISTING_UNIT_TEST,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_4 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_4",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name="test_4",
        did_pass=False,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.GENERATED_REGRESSION,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_5 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_5",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name="test_5",
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.REPLAY_TEST,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_results = [test_1, test_2, test_3]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_3]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_3, test_4]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_4]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_4, test_5]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_4, test_5]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    get_pr_number.cache_clear()
    os.environ["CODEFLASH_PR_NUMBER"] = "1234"
    test_results = [test_1, test_2, test_3]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_4]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_5]

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=100,
        best_test_results=TestResults(test_results=test_results),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    del os.environ["CODEFLASH_PR_NUMBER"]
