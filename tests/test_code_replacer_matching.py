"""Safety tests for get_optimized_code_for_module() fallback chain.

These tests verify the matching logic that maps AI-generated code blocks
to the correct source file, including all fallback strategies.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.code_replacer import get_optimized_code_for_module


def _make_optimized_code(file_to_code: dict[str, str]) -> MagicMock:
    """Create a mock CodeStringsMarkdown with a given file_to_path mapping."""
    mock = MagicMock()
    mock.file_to_path.return_value = file_to_code
    return mock


class TestGetOptimizedCodeForModule:
    """Test the fallback chain in get_optimized_code_for_module."""

    def test_exact_path_match(self) -> None:
        """When the relative path matches exactly, return that code."""
        code = _make_optimized_code({"src/main/java/com/example/Foo.java": "class Foo {}"})
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        assert result == "class Foo {}"

    def test_none_key_fallback(self) -> None:
        """When there's a single code block with 'None' key, use it."""
        code = _make_optimized_code({"None": "class Foo { optimized }"})
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        assert result == "class Foo { optimized }"

    def test_basename_match(self) -> None:
        """When the AI returns just 'Algorithms.java', match by basename."""
        code = _make_optimized_code({"Algorithms.java": "class Algorithms { fast }"})
        result = get_optimized_code_for_module(
            Path("src/main/java/com/example/Algorithms.java"), code
        )
        assert result == "class Algorithms { fast }"

    def test_basename_match_with_different_prefix(self) -> None:
        """Basename match should work even with a different directory prefix."""
        code = _make_optimized_code({"com/other/Foo.java": "class Foo { v2 }"})
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        assert result == "class Foo { v2 }"

    @patch("codeflash.languages.current.is_python", return_value=False)
    def test_single_block_fallback_non_python(self, _mock: object) -> None:
        """For non-Python, a single code block with wrong path should still match."""
        code = _make_optimized_code({"wrong/path/Bar.java": "class Bar { fast }"})
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        assert result == "class Bar { fast }"

    @patch("codeflash.languages.current.is_python", return_value=True)
    def test_single_block_fallback_python_does_not_match(self, _mock: object) -> None:
        """For Python, a single code block with wrong path should NOT match."""
        code = _make_optimized_code({"wrong/path/bar.py": "def bar(): pass"})
        result = get_optimized_code_for_module(Path("src/foo.py"), code)
        assert result == ""

    def test_no_match_returns_empty(self) -> None:
        """When multiple blocks exist and none match, return empty string."""
        code = _make_optimized_code({
            "other/File1.java": "class File1 {}",
            "other/File2.java": "class File2 {}",
        })
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        assert result == ""

    def test_none_key_with_multiple_blocks_no_match(self) -> None:
        """When there are multiple blocks including 'None', don't use None fallback."""
        code = _make_optimized_code({
            "None": "class Default {}",
            "other/File.java": "class File {}",
        })
        result = get_optimized_code_for_module(Path("src/main/java/com/example/Foo.java"), code)
        # With multiple blocks, the None-key fallback should NOT trigger
        assert result == ""
