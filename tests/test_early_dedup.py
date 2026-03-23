"""Tests for early candidate deduplication in CandidateProcessor."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from codeflash.languages.function_optimizer import CandidateProcessor
from codeflash.languages.python.normalizer import normalize_python_code
from codeflash.models.models import (
    CandidateEvaluationContext,
    CodeString,
    CodeStringsMarkdown,
    OptimizedCandidate,
    OptimizedCandidateSource,
)


def make_source_code(code: str) -> CodeStringsMarkdown:
    return CodeStringsMarkdown(
        code_strings=[CodeString(code=code, file_path=Path("test.py"))],
    )


def make_candidate(code: str, opt_id: str | None = None, source: OptimizedCandidateSource = OptimizedCandidateSource.OPTIMIZE) -> OptimizedCandidate:
    return OptimizedCandidate(
        source_code=make_source_code(code),
        explanation="test",
        optimization_id=opt_id or f"opt-{id(code)}",
        source=source,
    )


def normalize_fn(source: str) -> str:
    try:
        return normalize_python_code(source, remove_docstrings=True)
    except Exception:
        return source


ORIGINAL_CODE = "def foo(x):\n    return x + 1\n"
ORIGINAL_FLAT = f"# file: test.py\n{ORIGINAL_CODE}"

# Normalizes identically to ORIGINAL_CODE (docstring and comment stripped)
IDENTICAL_TO_ORIGINAL = 'def foo(x):\n    """Docstring."""\n    # comment\n    return x + 1\n'

# Different from original
CANDIDATE_A = "def foo(x):\n    return x + 2\n"
CANDIDATE_B = "def foo(x):\n    return x * 2\n"
CANDIDATE_C = "def foo(x):\n    return x << 1\n"

# Normalizes identically to CANDIDATE_A (added comment stripped by normalizer)
CANDIDATE_A_DUP = "def foo(x):\n    # optimized\n    return x + 2\n"


def make_done_future(value=None):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


def make_processor(initial_candidates, eval_ctx=None):
    if eval_ctx is None:
        eval_ctx = CandidateEvaluationContext()
    return CandidateProcessor(
        initial_candidates=initial_candidates,
        future_line_profile_results=make_done_future(None),
        eval_ctx=eval_ctx,
        effort="default",
        original_markdown_code=f"```python\n{ORIGINAL_CODE}```",
        future_all_refinements=[],
        future_all_code_repair=[],
        future_adaptive_optimizations=[],
        normalize_fn=normalize_fn,
        normalized_original=normalize_fn(ORIGINAL_CODE.strip()),
        original_flat_code=ORIGINAL_FLAT,
    )


class TestDedup:
    def test_unique_candidates_pass_through(self):
        candidates = [
            make_candidate(CANDIDATE_A, "opt-a"),
            make_candidate(CANDIDATE_B, "opt-b"),
            make_candidate(CANDIDATE_C, "opt-c"),
        ]
        proc = make_processor(candidates)
        assert proc.candidate_len == 3

    def test_identical_to_original_removed(self):
        candidates = [
            make_candidate(IDENTICAL_TO_ORIGINAL, "opt-dup-orig"),
            make_candidate(CANDIDATE_A, "opt-a"),
        ]
        proc = make_processor(candidates)
        assert proc.candidate_len == 1

    def test_intra_batch_duplicates_removed(self):
        candidates = [
            make_candidate(CANDIDATE_A, "opt-a1"),
            make_candidate(CANDIDATE_A_DUP, "opt-a2"),
            make_candidate(CANDIDATE_B, "opt-b"),
        ]
        proc = make_processor(candidates)
        assert proc.candidate_len == 2

    def test_cross_batch_duplicates_copy_results(self):
        eval_ctx = CandidateEvaluationContext()
        # Simulate a previously-benchmarked candidate
        prev_candidate = make_candidate(CANDIDATE_A, "opt-prev")
        eval_ctx.register_new_candidate(
            normalize_fn(CANDIDATE_A.strip()),
            prev_candidate,
            ORIGINAL_FLAT,
        )
        eval_ctx.record_successful_candidate("opt-prev", runtime=1000.0, speedup=2.0)

        # New batch has a duplicate of the already-benchmarked candidate
        new_candidates = [
            make_candidate(CANDIDATE_A_DUP, "opt-new-dup"),
            make_candidate(CANDIDATE_B, "opt-b"),
        ]
        proc = make_processor(new_candidates, eval_ctx=eval_ctx)
        # Only CANDIDATE_B should be queued (A_DUP is a cross-batch dup)
        assert proc.candidate_len == 1
        # Results should be copied to the duplicate
        assert eval_ctx.speedup_ratios["opt-new-dup"] == 2.0
        assert eval_ctx.optimized_runtimes["opt-new-dup"] == 1000.0
        assert eval_ctx.is_correct["opt-new-dup"] is True

    def test_empty_list(self):
        proc = make_processor([])
        assert proc.candidate_len == 0

    def test_all_duplicates_of_original(self):
        candidates = [
            make_candidate(IDENTICAL_TO_ORIGINAL, "opt-1"),
            make_candidate(ORIGINAL_CODE, "opt-2"),
        ]
        proc = make_processor(candidates)
        assert proc.candidate_len == 0

    def test_mixed_removal_types(self):
        eval_ctx = CandidateEvaluationContext()
        prev = make_candidate(CANDIDATE_C, "opt-prev-c")
        eval_ctx.register_new_candidate(normalize_fn(CANDIDATE_C.strip()), prev, ORIGINAL_FLAT)
        eval_ctx.record_successful_candidate("opt-prev-c", runtime=500.0, speedup=3.0)

        candidates = [
            make_candidate(IDENTICAL_TO_ORIGINAL, "opt-orig"),  # identical to original
            make_candidate(CANDIDATE_A, "opt-a1"),              # unique
            make_candidate(CANDIDATE_A_DUP, "opt-a2"),          # intra-batch dup of opt-a1
            make_candidate(CANDIDATE_C, "opt-c-dup"),           # cross-batch dup
            make_candidate(CANDIDATE_B, "opt-b"),               # unique
        ]
        proc = make_processor(candidates, eval_ctx=eval_ctx)
        # Only CANDIDATE_A and CANDIDATE_B should survive
        assert proc.candidate_len == 2
        # Cross-batch dup should have results copied
        assert eval_ctx.speedup_ratios["opt-c-dup"] == 3.0

    def test_dedup_in_async_batch(self):
        """Candidates arriving from line profiler are deduped against prior batches via seen_normalized."""
        candidates_initial = [make_candidate(CANDIDATE_A, "opt-a")]
        proc = make_processor(candidates_initial)
        assert proc.candidate_len == 1

        # Simulate what _process_candidates does: dedup a new batch
        async_batch = [
            make_candidate(CANDIDATE_B, "opt-b"),
            make_candidate(CANDIDATE_A_DUP, "opt-a-lp"),  # dup of initial, caught by seen_normalized
        ]
        deduped = proc.dedup_candidates(async_batch)
        assert len(deduped) == 1
        assert deduped[0].optimization_id == "opt-b"

    def test_dedup_in_async_batch_after_benchmark(self):
        """After initial candidates are benchmarked, async batch dedup catches cross-batch dups."""
        eval_ctx = CandidateEvaluationContext()
        # Simulate initial candidate already benchmarked
        prev = make_candidate(CANDIDATE_A, "opt-a")
        eval_ctx.register_new_candidate(normalize_fn(CANDIDATE_A.strip()), prev, ORIGINAL_FLAT)
        eval_ctx.record_successful_candidate("opt-a", runtime=2000.0, speedup=1.5)

        proc = make_processor([], eval_ctx=eval_ctx)

        async_batch = [
            make_candidate(CANDIDATE_A_DUP, "opt-a-lp"),
            make_candidate(CANDIDATE_B, "opt-b"),
        ]
        deduped = proc.dedup_candidates(async_batch)
        assert len(deduped) == 1
        assert deduped[0].optimization_id == "opt-b"
        assert eval_ctx.speedup_ratios["opt-a-lp"] == 1.5


class TestCandidateEvaluationContext:
    """Direct tests for register_new_candidate and handle_duplicate_candidate with original_flat_code param."""

    def test_register_new_candidate_stores_diff_len(self):
        eval_ctx = CandidateEvaluationContext()
        candidate = make_candidate(CANDIDATE_A, "opt-a")
        normalized = normalize_fn(CANDIDATE_A.strip())

        eval_ctx.register_new_candidate(normalized, candidate, ORIGINAL_FLAT)

        entry = eval_ctx.ast_code_to_id[normalized]
        assert entry["optimization_id"] == "opt-a"
        assert entry["shorter_source_code"] is candidate.source_code
        assert isinstance(entry["diff_len"], int)
        assert entry["diff_len"] > 0

    def test_handle_duplicate_copies_all_results(self):
        eval_ctx = CandidateEvaluationContext()
        first = make_candidate(CANDIDATE_A, "opt-first")
        normalized = normalize_fn(CANDIDATE_A.strip())

        eval_ctx.register_new_candidate(normalized, first, ORIGINAL_FLAT)
        eval_ctx.record_successful_candidate("opt-first", runtime=1234.0, speedup=2.5)
        eval_ctx.record_line_profiler_result("opt-first", "line profiler output")

        dup = make_candidate(CANDIDATE_A_DUP, "opt-dup")
        eval_ctx.handle_duplicate_candidate(dup, normalized, ORIGINAL_FLAT)

        assert eval_ctx.speedup_ratios["opt-dup"] == 2.5
        assert eval_ctx.optimized_runtimes["opt-dup"] == 1234.0
        assert eval_ctx.is_correct["opt-dup"] is True
        assert eval_ctx.optimized_line_profiler_results["opt-dup"] == "line profiler output"

    def test_handle_duplicate_copies_failed_results(self):
        eval_ctx = CandidateEvaluationContext()
        first = make_candidate(CANDIDATE_A, "opt-first")
        normalized = normalize_fn(CANDIDATE_A.strip())

        eval_ctx.register_new_candidate(normalized, first, ORIGINAL_FLAT)
        eval_ctx.record_failed_candidate("opt-first")

        dup = make_candidate(CANDIDATE_A_DUP, "opt-dup")
        eval_ctx.handle_duplicate_candidate(dup, normalized, ORIGINAL_FLAT)

        assert eval_ctx.speedup_ratios["opt-dup"] is None
        assert eval_ctx.optimized_runtimes["opt-dup"] is None
        assert eval_ctx.is_correct["opt-dup"] is False

    def test_handle_duplicate_tracks_shorter_source(self):
        """When a duplicate has a shorter diff, it replaces the stored shorter_source_code."""
        eval_ctx = CandidateEvaluationContext()
        # Register a candidate with longer code
        longer_code = "def foo(x):\n    # this comment makes it longer\n    # and this one too\n    return x + 2\n"
        first = make_candidate(longer_code, "opt-long")
        normalized = normalize_fn(longer_code.strip())

        eval_ctx.register_new_candidate(normalized, first, ORIGINAL_FLAT)
        eval_ctx.record_successful_candidate("opt-long", runtime=500.0, speedup=3.0)
        original_diff_len = eval_ctx.ast_code_to_id[normalized]["diff_len"]

        # Duplicate with shorter code (same normalized form)
        shorter = make_candidate(CANDIDATE_A, "opt-short")
        eval_ctx.handle_duplicate_candidate(shorter, normalized, ORIGINAL_FLAT)
        new_diff_len = eval_ctx.ast_code_to_id[normalized]["diff_len"]

        # Shorter code should have replaced the longer one
        assert new_diff_len <= original_diff_len
        assert eval_ctx.ast_code_to_id[normalized]["shorter_source_code"] is shorter.source_code

    def test_handle_duplicate_keeps_shorter_when_new_is_longer(self):
        """When a duplicate has a longer diff, the original shorter_source_code is kept."""
        eval_ctx = CandidateEvaluationContext()
        first = make_candidate(CANDIDATE_A, "opt-short")
        normalized = normalize_fn(CANDIDATE_A.strip())

        eval_ctx.register_new_candidate(normalized, first, ORIGINAL_FLAT)
        eval_ctx.record_successful_candidate("opt-short", runtime=500.0, speedup=3.0)

        longer_code = "def foo(x):\n    # this comment makes it longer\n    # and this one too\n    return x + 2\n"
        dup = make_candidate(longer_code, "opt-long")
        eval_ctx.handle_duplicate_candidate(dup, normalized, ORIGINAL_FLAT)

        assert eval_ctx.ast_code_to_id[normalized]["shorter_source_code"] is first.source_code
