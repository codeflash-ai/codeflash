from codeflash.models.models import OptimizedCandidateResult
from codeflash.result.critic import speedup_critic
from codeflash.verification.test_results import TestResults


def test_speedup_critic():
    original_code_runtime = 1000
    best_runtime_until_now = 1000
    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=800,
        best_test_results=TestResults(),
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now)  # 20% improvement

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=940,
        best_test_results=TestResults(),
    )

    assert not speedup_critic(
        candidate_result, original_code_runtime, best_runtime_until_now
    )  # 6% improvement

    original_code_runtime = 100000
    best_runtime_until_now = 100000

    candidate_result = OptimizedCandidateResult(
        times_run=5,
        best_test_runtime=94000,
        best_test_results=TestResults(),
    )

    assert speedup_critic(candidate_result, original_code_runtime, best_runtime_until_now)  # 6% improvement
