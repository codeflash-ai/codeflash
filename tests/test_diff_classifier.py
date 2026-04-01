from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.discovery.diff_classifier import (
    DiffCategory,
    FunctionDiffInfo,
    _is_comment_line,
    _is_logic_line,
    get_effort_for_diff,
)


class TestIsCommentLine:
    def test_python_comment(self) -> None:
        assert _is_comment_line("    # this is a comment")
        assert _is_comment_line("# top-level comment")

    def test_js_comment(self) -> None:
        assert _is_comment_line("  // this is a comment")
        assert _is_comment_line("// top-level")

    def test_multiline_comment(self) -> None:
        assert _is_comment_line("  * middle of block comment")
        assert _is_comment_line("  /* start of block comment")
        assert _is_comment_line("  */ end of block comment")

    def test_docstring(self) -> None:
        assert _is_comment_line('    """docstring"""')
        assert _is_comment_line("    '''docstring'''")

    def test_not_a_comment(self) -> None:
        assert not _is_comment_line("    x = 1")
        assert not _is_comment_line("    return x")
        assert not _is_comment_line("    for i in range(10):")


class TestIsLogicLine:
    def test_logic_lines(self) -> None:
        assert _is_logic_line("    x = 1")
        assert _is_logic_line("    return x + y")
        assert _is_logic_line("    if condition:")

    def test_not_logic_lines(self) -> None:
        assert not _is_logic_line("")
        assert not _is_logic_line("    ")
        assert not _is_logic_line("    # comment")
        assert not _is_logic_line("    // comment")


class TestDiffCategory:
    def test_cosmetic_whitespace(self) -> None:
        info = FunctionDiffInfo(
            category=DiffCategory.COSMETIC,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=3,
            is_comment_only=False,
            is_whitespace_only=True,
        )
        assert info.category == DiffCategory.COSMETIC
        assert info.is_whitespace_only

    def test_cosmetic_comments(self) -> None:
        info = FunctionDiffInfo(
            category=DiffCategory.COSMETIC,
            added_logic_lines=0,
            removed_logic_lines=0,
            total_changed_lines=5,
            is_comment_only=True,
            is_whitespace_only=False,
        )
        assert info.category == DiffCategory.COSMETIC
        assert info.is_comment_only

    def test_trivial(self) -> None:
        info = FunctionDiffInfo(
            category=DiffCategory.TRIVIAL,
            added_logic_lines=1,
            removed_logic_lines=1,
            total_changed_lines=2,
            is_comment_only=False,
            is_whitespace_only=False,
        )
        assert info.category == DiffCategory.TRIVIAL

    def test_meaningful(self) -> None:
        info = FunctionDiffInfo(
            category=DiffCategory.MEANINGFUL,
            added_logic_lines=5,
            removed_logic_lines=3,
            total_changed_lines=8,
            is_comment_only=False,
            is_whitespace_only=False,
        )
        assert info.category == DiffCategory.MEANINGFUL

    def test_major(self) -> None:
        info = FunctionDiffInfo(
            category=DiffCategory.MAJOR,
            added_logic_lines=15,
            removed_logic_lines=10,
            total_changed_lines=25,
            is_comment_only=False,
            is_whitespace_only=False,
        )
        assert info.category == DiffCategory.MAJOR


class TestGetEffortForDiff:
    def test_trivial_gets_low_effort(self) -> None:
        info = FunctionDiffInfo(DiffCategory.TRIVIAL, 1, 0, 1, False, False)
        assert get_effort_for_diff(info) == "low"

    def test_major_gets_high_effort(self) -> None:
        info = FunctionDiffInfo(DiffCategory.MAJOR, 20, 10, 30, False, False)
        assert get_effort_for_diff(info) == "high"

    def test_meaningful_gets_default(self) -> None:
        info = FunctionDiffInfo(DiffCategory.MEANINGFUL, 5, 3, 8, False, False)
        assert get_effort_for_diff(info) is None

    def test_cosmetic_gets_default(self) -> None:
        info = FunctionDiffInfo(DiffCategory.COSMETIC, 0, 0, 2, True, False)
        assert get_effort_for_diff(info) is None
