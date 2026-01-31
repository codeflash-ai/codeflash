"""Tests for Vitest test runner command construction.

These tests verify that Vitest commands are correctly constructed
with the appropriate flags and arguments.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.vitest_runner import (
    _build_vitest_behavioral_command,
    _build_vitest_benchmarking_command,
    _find_vitest_project_root,
)


class TestFindVitestProjectRoot:
    """Tests for _find_vitest_project_root function."""

    def test_finds_vitest_config_js(self) -> None:
        """Should find project root via vitest.config.js."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "vitest.config.js").write_text("export default {}")
            test_file = tmp_path / "tests" / "test.test.ts"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("")

            result = _find_vitest_project_root(test_file)

            assert result == tmp_path

    def test_finds_vitest_config_ts(self) -> None:
        """Should find project root via vitest.config.ts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "vitest.config.ts").write_text("export default {}")
            test_file = tmp_path / "tests" / "test.test.ts"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("")

            result = _find_vitest_project_root(test_file)

            assert result == tmp_path

    def test_finds_vite_config_js(self) -> None:
        """Should find project root via vite.config.js (Vitest can be configured in vite config)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "vite.config.js").write_text("export default {}")
            test_file = tmp_path / "tests" / "test.test.ts"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("")

            result = _find_vitest_project_root(test_file)

            assert result == tmp_path

    def test_falls_back_to_package_json(self) -> None:
        """Should fall back to package.json when no vitest config found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_file = tmp_path / "tests" / "test.test.ts"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("")

            result = _find_vitest_project_root(test_file)

            assert result == tmp_path

    def test_returns_none_when_no_config(self) -> None:
        """Should return None when no vitest/vite config or package.json found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "tests" / "test.test.ts"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("")

            result = _find_vitest_project_root(test_file)

            assert result is None


class TestBuildVitestBehavioralCommand:
    """Tests for _build_vitest_behavioral_command function."""

    def test_basic_command_structure(self) -> None:
        """Should build basic Vitest command with required flags."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert cmd[0] == "npx"
            assert cmd[1] == "vitest"
            assert cmd[2] == "run"

    def test_includes_reporter_flags(self) -> None:
        """Should include reporter flags for JUnit output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert "--reporter=default" in cmd
            assert "--reporter=junit" in cmd

    def test_includes_serial_execution_flag(self) -> None:
        """Should include flag for serial test execution."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert "--no-file-parallelism" in cmd

    def test_includes_test_files_as_absolute_paths(self) -> None:
        """Should include test files at the end of command as absolute paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file_a = tmp_path / "test_a.test.ts"
            test_file_b = tmp_path / "test_b.test.ts"
            test_file_a.write_text("")
            test_file_b.write_text("")

            cmd = _build_vitest_behavioral_command([test_file_a, test_file_b], timeout=60)

            assert str(test_file_a.resolve()) in cmd
            assert str(test_file_b.resolve()) in cmd

    def test_includes_timeout_in_milliseconds(self) -> None:
        """Should include test timeout in milliseconds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_behavioral_command([test_file], timeout=120)

            assert "--test-timeout=120000" in cmd

    def test_includes_output_file_when_provided(self) -> None:
        """Should include --outputFile flag when output_file is provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")
            output_file = tmp_path / "results.xml"

            cmd = _build_vitest_behavioral_command([test_file], timeout=60, output_file=output_file)

            assert f"--outputFile={output_file}" in cmd


class TestBuildVitestBenchmarkingCommand:
    """Tests for _build_vitest_benchmarking_command function."""

    def test_basic_command_structure(self) -> None:
        """Should build basic Vitest benchmarking command."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test__perf.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_benchmarking_command([test_file], timeout=60)

            assert cmd[0] == "npx"
            assert cmd[1] == "vitest"
            assert cmd[2] == "run"

    def test_includes_serial_execution(self) -> None:
        """Should include serial execution for consistent benchmarking."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test__perf.test.ts"
            test_file.write_text("")

            cmd = _build_vitest_benchmarking_command([test_file], timeout=60)

            assert "--no-file-parallelism" in cmd


class TestVitestVsJestCommandDifferences:
    """Tests documenting the key differences between Vitest and Jest commands."""

    def test_vitest_uses_run_subcommand(self) -> None:
        """Vitest uses 'run' for single execution, Jest doesn't need it."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            vitest_cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert vitest_cmd[0:3] == ["npx", "vitest", "run"]

    def test_vitest_uses_hyphenated_timeout(self) -> None:
        """Vitest uses --test-timeout, Jest uses --testTimeout (camelCase)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            vitest_cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            timeout_args = [arg for arg in vitest_cmd if "timeout" in arg.lower()]
            assert len(timeout_args) == 1
            assert timeout_args[0] == "--test-timeout=60000"

    def test_vitest_uses_no_file_parallelism(self) -> None:
        """Vitest uses --no-file-parallelism, Jest uses --runInBand."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            vitest_cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert "--no-file-parallelism" in vitest_cmd
            assert "--runInBand" not in vitest_cmd

    def test_vitest_positional_test_files(self) -> None:
        """Vitest uses positional args for test files, not --runTestsByPath."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.test.ts"
            test_file.write_text("")

            vitest_cmd = _build_vitest_behavioral_command([test_file], timeout=60)

            assert "--runTestsByPath" not in vitest_cmd
            assert str(test_file.resolve()) in vitest_cmd
