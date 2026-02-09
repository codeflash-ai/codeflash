"""Tests for normalize_ignore_paths function."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeflash.code_utils.code_utils import is_glob_pattern, normalize_ignore_paths


class TestIsGlobPattern:
    """Tests for is_glob_pattern function."""

    def test_asterisk_pattern(self) -> None:
        assert is_glob_pattern("*.py") is True
        assert is_glob_pattern("**/*.js") is True
        assert is_glob_pattern("node_modules/*") is True

    def test_question_mark_pattern(self) -> None:
        assert is_glob_pattern("file?.txt") is True
        assert is_glob_pattern("test_?.py") is True

    def test_bracket_pattern(self) -> None:
        assert is_glob_pattern("[abc].txt") is True
        assert is_glob_pattern("file[0-9].log") is True

    def test_literal_paths(self) -> None:
        assert is_glob_pattern("node_modules") is False
        assert is_glob_pattern("src/utils") is False
        assert is_glob_pattern("/absolute/path") is False
        assert is_glob_pattern("relative/path/file.py") is False


class TestNormalizeIgnorePaths:
    """Tests for normalize_ignore_paths function."""

    def test_empty_list(self) -> None:
        result = normalize_ignore_paths([])
        assert result == []

    def test_literal_existing_path(self, tmp_path: Path) -> None:
        # Create a directory
        test_dir = tmp_path / "node_modules"
        test_dir.mkdir()

        result = normalize_ignore_paths(["node_modules"], base_path=tmp_path)

        assert len(result) == 1
        assert result[0] == test_dir.resolve()

    def test_literal_nonexistent_path_skipped(self, tmp_path: Path) -> None:
        # Don't create the directory - should be silently skipped
        result = normalize_ignore_paths(["nonexistent_dir"], base_path=tmp_path)

        assert result == []

    def test_multiple_literal_paths(self, tmp_path: Path) -> None:
        # Create directories
        dir1 = tmp_path / "node_modules"
        dir2 = tmp_path / "dist"
        dir1.mkdir()
        dir2.mkdir()

        result = normalize_ignore_paths(["node_modules", "dist"], base_path=tmp_path)

        assert len(result) == 2
        assert set(result) == {dir1.resolve(), dir2.resolve()}

    def test_glob_pattern_single_asterisk(self, tmp_path: Path) -> None:
        # Create test files
        (tmp_path / "file1.log").touch()
        (tmp_path / "file2.log").touch()
        (tmp_path / "file.txt").touch()

        result = normalize_ignore_paths(["*.log"], base_path=tmp_path)

        assert len(result) == 2
        resolved_names = {p.name for p in result}
        assert resolved_names == {"file1.log", "file2.log"}

    def test_glob_pattern_double_asterisk(self, tmp_path: Path) -> None:
        # Create nested structure
        subdir = tmp_path / "src" / "utils"
        subdir.mkdir(parents=True)
        (subdir / "test_helper.py").touch()
        (tmp_path / "src" / "test_main.py").touch()
        (tmp_path / "test_root.py").touch()

        result = normalize_ignore_paths(["**/test_*.py"], base_path=tmp_path)

        assert len(result) == 3
        resolved_names = {p.name for p in result}
        assert resolved_names == {"test_helper.py", "test_main.py", "test_root.py"}

    def test_glob_pattern_directory_contents(self, tmp_path: Path) -> None:
        # Create directory with contents
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "package1").mkdir()
        (node_modules / "package2").mkdir()

        result = normalize_ignore_paths(["node_modules/*"], base_path=tmp_path)

        assert len(result) == 2
        resolved_names = {p.name for p in result}
        assert resolved_names == {"package1", "package2"}

    def test_glob_pattern_no_matches(self, tmp_path: Path) -> None:
        # Pattern with no matches should return empty list
        result = normalize_ignore_paths(["*.nonexistent"], base_path=tmp_path)

        assert result == []

    def test_mixed_literal_and_patterns(self, tmp_path: Path) -> None:
        # Create test structure
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (tmp_path / "debug.log").touch()
        (tmp_path / "error.log").touch()

        result = normalize_ignore_paths(["node_modules", "*.log"], base_path=tmp_path)

        assert len(result) == 3
        resolved_names = {p.name for p in result}
        assert resolved_names == {"node_modules", "debug.log", "error.log"}

    def test_deduplication(self, tmp_path: Path) -> None:
        # Create a file that matches multiple patterns
        (tmp_path / "test.log").touch()

        # Same file should only appear once
        result = normalize_ignore_paths(["test.log", "*.log"], base_path=tmp_path)

        assert len(result) == 1
        assert result[0].name == "test.log"

    def test_nested_directory_pattern(self, tmp_path: Path) -> None:
        # Create nested test directories
        tests_dir = tmp_path / "src" / "__tests__"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test1.js").touch()
        (tests_dir / "test2.js").touch()

        result = normalize_ignore_paths(["src/__tests__/*.js"], base_path=tmp_path)

        assert len(result) == 2
        resolved_names = {p.name for p in result}
        assert resolved_names == {"test1.js", "test2.js"}

    def test_absolute_path_literal(self, tmp_path: Path) -> None:
        # Create a directory
        test_dir = tmp_path / "absolute_test"
        test_dir.mkdir()

        # Use absolute path
        result = normalize_ignore_paths([str(test_dir)], base_path=tmp_path)

        assert len(result) == 1
        assert result[0] == test_dir.resolve()

    def test_relative_path_with_subdirectory(self, tmp_path: Path) -> None:
        # Create nested directory
        nested = tmp_path / "src" / "vendor"
        nested.mkdir(parents=True)

        result = normalize_ignore_paths(["src/vendor"], base_path=tmp_path)

        assert len(result) == 1
        assert result[0] == nested.resolve()

    def test_default_base_path_uses_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create a directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Call without base_path
        result = normalize_ignore_paths(["test_dir"])

        assert len(result) == 1
        assert result[0] == test_dir.resolve()

    def test_bracket_pattern(self, tmp_path: Path) -> None:
        # Create files matching bracket pattern
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.txt").touch()
        (tmp_path / "fileA.txt").touch()

        result = normalize_ignore_paths(["file[12].txt"], base_path=tmp_path)

        assert len(result) == 2
        resolved_names = {p.name for p in result}
        assert resolved_names == {"file1.txt", "file2.txt"}

    def test_question_mark_pattern(self, tmp_path: Path) -> None:
        # Create files matching question mark pattern
        (tmp_path / "test_a.py").touch()
        (tmp_path / "test_b.py").touch()
        (tmp_path / "test_ab.py").touch()

        result = normalize_ignore_paths(["test_?.py"], base_path=tmp_path)

        assert len(result) == 2
        resolved_names = {p.name for p in result}
        assert resolved_names == {"test_a.py", "test_b.py"}
