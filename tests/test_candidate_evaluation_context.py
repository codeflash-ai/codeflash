from unittest.mock import MagicMock, patch

from codeflash.models.models import (
    CandidateEvaluationContext,
    CodeString,
    CodeStringsMarkdown,
    OptimizedCandidate,
    OptimizedCandidateSource,
)


def make_source_code(code: str = "pass") -> CodeStringsMarkdown:
    return CodeStringsMarkdown(code_strings=[CodeString(code=code, file_path=None)])


def make_candidate(optimization_id: str, code: str = "pass") -> OptimizedCandidate:
    return OptimizedCandidate(
        source_code=make_source_code(code),
        explanation="test",
        optimization_id=optimization_id,
        source=OptimizedCandidateSource.OPTIMIZE,
    )


def make_context_with_past_candidate(past_opt_id: str, normalized_code: str, code: str = "pass") -> CandidateEvaluationContext:
    ctx = CandidateEvaluationContext()
    ctx.ast_code_to_id[normalized_code] = {
        "optimization_id": past_opt_id,
        "shorter_source_code": make_source_code(code),
        "diff_len": 10,
    }
    ctx.speedup_ratios[past_opt_id] = 2.0
    ctx.is_correct[past_opt_id] = True
    ctx.optimized_runtimes[past_opt_id] = 0.5
    return ctx


@patch("codeflash.code_utils.code_utils.diff_length", return_value=10)
def test_copy_line_profiler_results_existing_key(_mock_diff: MagicMock) -> None:
    past_opt_id = "past-123"
    normalized_code = "normalized_code_abc"
    ctx = make_context_with_past_candidate(past_opt_id, normalized_code)
    ctx.optimized_line_profiler_results[past_opt_id] = "line_profile_output_data"

    candidate = make_candidate("new-456")
    ctx.handle_duplicate_candidate(candidate, normalized_code, "original_flat")

    assert ctx.optimized_line_profiler_results["new-456"] == "line_profile_output_data"


@patch("codeflash.code_utils.code_utils.diff_length", return_value=10)
def test_copy_line_profiler_results_missing_key(_mock_diff: MagicMock) -> None:
    past_opt_id = "past-789"
    normalized_code = "normalized_code_xyz"
    ctx = make_context_with_past_candidate(past_opt_id, normalized_code)
    # No line profiler result for past_opt_id

    candidate = make_candidate("new-012")
    ctx.handle_duplicate_candidate(candidate, normalized_code, "original_flat")

    assert "new-012" not in ctx.optimized_line_profiler_results


@patch("codeflash.code_utils.code_utils.diff_length", return_value=10)
def test_copy_line_profiler_results_does_not_corrupt(_mock_diff: MagicMock) -> None:
    past_opt_id = "past-aaa"
    normalized_code = "normalized_code_zzz"
    ctx = make_context_with_past_candidate(past_opt_id, normalized_code)
    original_value = "original_profile_data"
    ctx.optimized_line_profiler_results[past_opt_id] = original_value

    candidate = make_candidate("new-bbb")
    ctx.handle_duplicate_candidate(candidate, normalized_code, "original_flat")

    # Original entry unchanged
    assert ctx.optimized_line_profiler_results[past_opt_id] == original_value
    # New entry is equal but does not affect original on mutation
    assert ctx.optimized_line_profiler_results["new-bbb"] == original_value
