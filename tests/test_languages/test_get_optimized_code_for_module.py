from __future__ import annotations

from pathlib import Path

from codeflash.languages.code_replacer import get_optimized_code_for_module
from codeflash.models.models import CodeString, CodeStringsMarkdown


def _make_markdown(*code_strings: tuple[str, str | None]) -> CodeStringsMarkdown:
    return CodeStringsMarkdown(
        code_strings=[
            CodeString(code=code, file_path=Path(path) if path else None) for code, path in code_strings
        ]
    )


# --- Exact path match ---


def test_exact_path_match_single_file():
    md = _make_markdown(("def foo(): pass", "src/utils.py"))
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == "def foo(): pass"


def test_exact_path_match_picks_correct_file():
    md = _make_markdown(
        ("def foo(): pass", "src/utils.py"),
        ("def bar(): pass", "src/helpers.py"),
    )
    assert get_optimized_code_for_module(Path("src/helpers.py"), md) == "def bar(): pass"


def test_exact_match_preferred_over_basename():
    md = _make_markdown(
        ("def wrong(): pass", "other/utils.py"),
        ("def correct(): pass", "src/utils.py"),
    )
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == "def correct(): pass"


# --- Fallback 1: single None-path block ---


def test_none_path_fallback_single_block():
    md = _make_markdown(("def foo(): pass", None))
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == "def foo(): pass"


def test_none_path_fallback_ignored_when_named_blocks_exist():
    md = _make_markdown(("def foo(): pass", None), ("def bar(): pass", "src/other.py"))
    # None fallback requires exactly one entry in the dict keyed "None" and no other keys
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == ""


# --- Fallback 2: basename match ---


def test_basename_fallback_different_directory():
    md = _make_markdown(("def optimized(): pass", "wrong/dir/utils.py"))
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == "def optimized(): pass"


def test_basename_fallback_skips_non_matching_context_files():
    """Target file returned alongside unrelated context files — basename picks the right one."""
    md = _make_markdown(
        ("import logging", "codeflash/cli_cmds/console.py"),
        ("def optimized(): pass", "other/version.py"),
    )
    assert get_optimized_code_for_module(Path("codeflash/version.py"), md) == "def optimized(): pass"


def test_basename_fallback_ambiguous_returns_empty():
    md = _make_markdown(
        ("def foo(): pass", "a/utils.py"),
        ("def bar(): pass", "b/utils.py"),
    )
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == ""




def test_no_match_returns_empty():
    md = _make_markdown(("def foo(): pass", "src/helpers.py"))
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == ""


def test_empty_markdown_returns_empty():
    md = CodeStringsMarkdown(code_strings=[])
    assert get_optimized_code_for_module(Path("src/utils.py"), md) == ""


def test_context_files_only_returns_empty():
    """Reproduces the CI issue: LLM returns only context files, not the target."""
    md = _make_markdown(
        ("import logging\nlogger = logging.getLogger()", "codeflash/cli_cmds/console.py"),
        ("class AiServiceClient: ...", "codeflash/api/aiservice.py"),
    )
    assert get_optimized_code_for_module(Path("codeflash/version.py"), md) == ""
