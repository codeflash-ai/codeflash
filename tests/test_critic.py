import os
from pathlib import Path
from unittest.mock import Mock

from codeflash.code_utils.env_utils import get_pr_number
from codeflash.models.models import (
    CodeOptimizationContext,
    CoverageData,
    CoverageStatus,
    FunctionCoverage,
    FunctionTestInvocation,
    InvocationId,
    OptimizedCandidateResult,
    TestResults,
    TestType,
)
from codeflash.result.critic import (
    coverage_critic,
    performance_gain,
    quantity_of_tests_critic,
    speedup_critic,
    throughput_gain,
)


def test_performance_gain() -> None:
    assert performance_gain(original_runtime_ns=1000, optimized_runtime_ns=0) == 0.0

    assert performance_gain(original_runtime_ns=1000, optimized_runtime_ns=500) == 1.0

    assert performance_gain(original_runtime_ns=1000, optimized_runtime_ns=900) == 0.1111111111111111

    assert performance_gain(original_runtime_ns=1000, optimized_runtime_ns=1000) == 0.0

    assert performance_gain(original_runtime_ns=1000, optimized_runtime_ns=1100) == -0.09090909090909091


def test_speedup_critic() -> None:
    original_code_runtime = 1000
    best_runtime_until_now = 1000
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=800,
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=12,
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now, disable_gh_action_noise=True)  # 20% improvement

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=940,
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now, disable_gh_action_noise=True)  # 6% improvement

    original_code_runtime = 100000
    best_runtime_until_now = 100000

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=94000,
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now, disable_gh_action_noise=True)  # 6% improvement


def test_generated_test_critic() -> None:
    test_1 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_1",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name=Path("test_1"),
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
        file_name=Path("test_2"),
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
        file_name=Path("test_3"),
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
        file_name=Path("test_4"),
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
        file_name=Path("test_5"),
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.REPLAY_TEST,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )

    test_6 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_6",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name=Path("test_6"),
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.GENERATED_REGRESSION,
        return_value=None,
        timed_out=False,
        loop_index=2,
    )

    test_7 = FunctionTestInvocation(
        id=InvocationId(
            test_module_path="",
            test_class_name="",
            test_function_name="test_7",
            function_getting_tested="sorter",
            iteration_id="",
        ),
        file_name=Path("test_7"),
        did_pass=True,
        runtime=0,
        test_framework="pytest",
        test_type=TestType.EXISTING_UNIT_TEST,
        return_value=None,
        timed_out=False,
        loop_index=1,
    )
    test_results = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_6, test_7, test_1, test_4, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_3, test_4, test_2, test_7, test_1, test_6, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_4, test_5, test_1, test_1, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_4, test_6]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_4, test_5]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_4, test_5, test_1, test_1, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    get_pr_number.cache_clear()
    os.environ["CODEFLASH_PR_NUMBER"] = "1234"
    test_results = [test_1, test_2, test_3, test_6]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_4]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert not quantity_of_tests_critic(candidate_result)

    test_results = [test_1, test_2, test_3, test_5, test_1, test_1, test_1, test_1]

    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=100,
        behavior_test_results=TestResults(test_results=test_results),
        benchmarking_test_results=TestResults(),
        total_candidate_timing=12,
        optimization_candidate_index=0,
    )

    assert quantity_of_tests_critic(candidate_result)

    del os.environ["CODEFLASH_PR_NUMBER"]


def test_coverage_critic() -> None:
    mock_code_context = Mock(spec=CodeOptimizationContext)

    passing_coverage = CoverageData(
        file_path=Path("test_file.py"),
        coverage=100.0,
        function_name="test_function",
        functions_being_tested=["function1", "function2"],
        graph={},
        code_context=mock_code_context,
        main_func_coverage=FunctionCoverage(
            name="test_function",
            coverage=100.0,
            executed_lines=[10],
            unexecuted_lines=[2],
            executed_branches=[[5]],
            unexecuted_branches=[[1]],
        ),
        dependent_func_coverage=None,
        status=CoverageStatus.PARSED_SUCCESSFULLY,
    )

    assert coverage_critic(passing_coverage, "pytest") is True

    border_coverage = CoverageData(
        file_path=Path("test_file.py"),
        coverage=60.0,
        function_name="test_function",
        functions_being_tested=["function1", "function2"],
        graph={},
        code_context=mock_code_context,
        main_func_coverage=FunctionCoverage(
            name="test_function",
            coverage=50.0,
            executed_lines=[10],
            unexecuted_lines=[2],
            executed_branches=[[5]],
            unexecuted_branches=[[1]],
        ),
        dependent_func_coverage=None,
        status=CoverageStatus.PARSED_SUCCESSFULLY,
    )

    assert coverage_critic(border_coverage, "pytest") is True

    failing_coverage = CoverageData(
        file_path=Path("test_file.py"),
        coverage=30.0,
        function_name="test_function",
        functions_being_tested=["function1", "function2"],
        graph={},
        code_context=mock_code_context,
        main_func_coverage=FunctionCoverage(
            name="test_function",
            coverage=0.0,
            executed_lines=[],
            unexecuted_lines=[10],
            executed_branches=[],
            unexecuted_branches=[[5]],
        ),
        dependent_func_coverage=None,
        status=CoverageStatus.PARSED_SUCCESSFULLY,
    )

    assert coverage_critic(failing_coverage, "pytest") is False

    unittest_coverage = CoverageData(
        file_path=Path("test_file.py"),
        coverage=0,
        function_name="test_function",
        functions_being_tested=["function1", "function2"],
        graph={},
        code_context=mock_code_context,
        main_func_coverage=FunctionCoverage(
            name="test_function",
            coverage=0,
            executed_lines=[10],
            unexecuted_lines=[2],
            executed_branches=[[5]],
            unexecuted_branches=[[1]],
        ),
        dependent_func_coverage=None,
        status=CoverageStatus.PARSED_SUCCESSFULLY,
    )

    assert coverage_critic(unittest_coverage, "unittest") is True


def test_throughput_gain() -> None:
    """Test throughput_gain calculation."""
    # Test basic throughput improvement
    assert throughput_gain(original_throughput=100, optimized_throughput=150) == 0.5  # 50% improvement

    # Test no improvement
    assert throughput_gain(original_throughput=100, optimized_throughput=100) == 0.0

    # Test regression
    assert throughput_gain(original_throughput=100, optimized_throughput=80) == -0.2  # 20% regression

    # Test zero original throughput (edge case)
    assert throughput_gain(original_throughput=0, optimized_throughput=50) == 0.0

    # Test large improvement
    assert throughput_gain(original_throughput=50, optimized_throughput=200) == 3.0  # 300% improvement


def test_speedup_critic_with_async_throughput() -> None:
    """Test speedup_critic with async throughput evaluation."""
    original_code_runtime = 10000  # 10 microseconds
    original_async_throughput = 100

    # Test case 1: Both runtime and throughput improve significantly
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=8000,  # 20% runtime improvement
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=8000,
        async_throughput=120,  # 20% throughput improvement
    )

    assert speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=original_async_throughput,
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )

    # Test case 2: Runtime improves but throughput doesn't meet threshold
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=8000,  # 20% runtime improvement
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=8000,
        async_throughput=105,  # Only 5% throughput improvement (below 10% threshold)
    )

    assert not speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=original_async_throughput,
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )

    # Test case 3: Throughput improves but runtime doesn't meet threshold
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=9800,  # Only 2% runtime improvement (below 5% threshold)
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=9800,
        async_throughput=120,  # 20% throughput improvement
    )

    assert not speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=original_async_throughput,
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )

    # Test case 4: No throughput data - should fall back to runtime-only evaluation
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=8000,  # 20% runtime improvement
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=8000,
        async_throughput=None,  # No throughput data
    )

    assert speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=None,  # No original throughput data
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )

    # Test case 5: Test best_throughput_until_now comparison
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=8000,  # 20% runtime improvement
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=8000,
        async_throughput=115,  # 15% throughput improvement
    )

    # Should pass when no best throughput yet
    assert speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=original_async_throughput,
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )

    # Should fail when there's a better throughput already
    assert not speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=original_async_throughput,
        best_throughput_until_now=120,  # Better throughput already exists
        disable_gh_action_noise=True
    )

    # Test case 6: Zero original throughput (edge case)
    candidate_result = OptimizedCandidateResult(
        max_loop_count=5,
        best_test_runtime=8000,  # 20% runtime improvement
        behavior_test_results=TestResults(),
        benchmarking_test_results=TestResults(),
        optimization_candidate_index=0,
        total_candidate_timing=8000,
        async_throughput=50,
    )

    # Should pass when original throughput is 0 (throughput evaluation skipped)
    assert speedup_critic(
        candidate_result=candidate_result,
        original_code_runtime=original_code_runtime,
        best_runtime_until_now=None,
        original_async_throughput=0,  # Zero original throughput
        best_throughput_until_now=None,
        disable_gh_action_noise=True
    )
