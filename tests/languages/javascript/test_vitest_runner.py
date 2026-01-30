"""Tests for Vitest test runner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.vitest_runner import (
    _build_vitest_behavioral_command,
    _build_vitest_benchmarking_command,
    _find_vitest_project_root,
    run_vitest_behavioral_tests,
    run_vitest_benchmarking_tests,
    run_vitest_line_profile_tests,
)

if TYPE_CHECKING:
    from codeflash.models.models import TestFiles


@pytest.fixture
def mock_test_paths() -> MagicMock:
    """Create a mock TestFiles object."""
    mock = MagicMock()
    mock_file = MagicMock()
    mock_file.instrumented_behavior_file_path = Path("/project/tests/test_func.test.ts")
    mock_file.benchmarking_file_path = Path("/project/tests/test_func__perf.test.ts")
    mock.test_files = [mock_file]
    return mock


class TestFindVitestProjectRoot:
    """Tests for _find_vitest_project_root function."""

    def test_finds_vitest_config_js(self, tmp_path: Path) -> None:
        """Should find project root via vitest.config.js."""
        (tmp_path / "vitest.config.js").write_text("export default {}")
        test_file = tmp_path / "tests" / "test.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("")

        result = _find_vitest_project_root(test_file)

        assert result == tmp_path

    def test_finds_vitest_config_ts(self, tmp_path: Path) -> None:
        """Should find project root via vitest.config.ts."""
        (tmp_path / "vitest.config.ts").write_text("export default {}")
        test_file = tmp_path / "tests" / "test.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("")

        result = _find_vitest_project_root(test_file)

        assert result == tmp_path

    def test_finds_vite_config_js(self, tmp_path: Path) -> None:
        """Should find project root via vite.config.js (Vitest can be configured in vite config)."""
        (tmp_path / "vite.config.js").write_text("export default {}")
        test_file = tmp_path / "tests" / "test.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("")

        result = _find_vitest_project_root(test_file)

        assert result == tmp_path

    def test_falls_back_to_package_json(self, tmp_path: Path) -> None:
        """Should fall back to package.json when no vitest config found."""
        (tmp_path / "package.json").write_text('{"name": "test"}')
        test_file = tmp_path / "tests" / "test.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("")

        result = _find_vitest_project_root(test_file)

        assert result == tmp_path

    def test_returns_none_when_no_config(self, tmp_path: Path) -> None:
        """Should return None when no vitest/vite config or package.json found."""
        test_file = tmp_path / "tests" / "test.test.ts"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("")

        result = _find_vitest_project_root(test_file)

        assert result is None


class TestBuildVitestBehavioralCommand:
    """Tests for _build_vitest_behavioral_command function."""

    def test_basic_command_structure(self) -> None:
        """Should build basic Vitest command with required flags."""
        test_files = [Path("/project/tests/test.test.ts")]

        cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        assert "npx" in cmd
        assert "vitest" in cmd
        assert "run" in cmd  # Vitest uses 'run' for single execution

    def test_includes_reporter_flags(self) -> None:
        """Should include reporter flags for JUnit output."""
        test_files = [Path("/project/tests/test.test.ts")]

        cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        assert "--reporter=default" in cmd
        assert "--reporter=junit" in cmd

    def test_includes_serial_execution_flag(self) -> None:
        """Should include flag for serial test execution."""
        test_files = [Path("/project/tests/test.test.ts")]

        cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        # Vitest uses --no-file-parallelism for serial execution
        assert "--no-file-parallelism" in cmd

    def test_includes_test_files(self) -> None:
        """Should include test files at the end of command."""
        test_files = [
            Path("/project/tests/test_a.test.ts"),
            Path("/project/tests/test_b.test.ts"),
        ]

        cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        assert str(Path("/project/tests/test_a.test.ts").resolve()) in cmd
        assert str(Path("/project/tests/test_b.test.ts").resolve()) in cmd

    def test_includes_timeout(self) -> None:
        """Should include test timeout in milliseconds."""
        test_files = [Path("/project/tests/test.test.ts")]

        cmd = _build_vitest_behavioral_command(test_files, timeout=120)

        # Vitest uses --test-timeout=<ms> (note the hyphen, not camelCase)
        assert "--test-timeout=120000" in cmd


class TestBuildVitestBenchmarkingCommand:
    """Tests for _build_vitest_benchmarking_command function."""

    def test_basic_command_structure(self) -> None:
        """Should build basic Vitest benchmarking command."""
        test_files = [Path("/project/tests/test__perf.test.ts")]

        cmd = _build_vitest_benchmarking_command(test_files, timeout=60)

        assert "npx" in cmd
        assert "vitest" in cmd
        assert "run" in cmd

    def test_includes_serial_execution(self) -> None:
        """Should include serial execution for consistent benchmarking."""
        test_files = [Path("/project/tests/test__perf.test.ts")]

        cmd = _build_vitest_benchmarking_command(test_files, timeout=60)

        assert "--no-file-parallelism" in cmd


class TestRunVitestBehavioralTests:
    """Tests for run_vitest_behavioral_tests function."""

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_sets_vitest_env_vars(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should set correct environment variables for Vitest."""
        mock_tmp_file.return_value = tmp_path / "vitest_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        run_vitest_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={"PATH": "/usr/bin"},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        call_args = mock_subprocess_run.call_args
        env = call_args.kwargs.get("env", {})

        # Check Vitest-specific env vars
        assert "CODEFLASH_OUTPUT_FILE" in env
        assert "CODEFLASH_MODE" in env
        assert env["CODEFLASH_MODE"] == "behavior"
        assert "CODEFLASH_LOOP_INDEX" in env

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_returns_result_file_path(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should return the result file path as first element of tuple."""
        result_path = tmp_path / "vitest_results.xml"
        mock_tmp_file.return_value = result_path
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        result_file_path, _, _, _ = run_vitest_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        assert result_file_path == result_path

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_uses_vitest_run_command(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should use 'vitest run' for single execution."""
        mock_tmp_file.return_value = tmp_path / "vitest_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        run_vitest_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        call_args = mock_subprocess_run.call_args
        cmd = call_args[0][0]

        assert "vitest" in cmd
        assert "run" in cmd


class TestRunVitestBenchmarkingTests:
    """Tests for run_vitest_benchmarking_tests function."""

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_sets_performance_mode(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should set CODEFLASH_MODE to 'performance'."""
        mock_tmp_file.return_value = tmp_path / "vitest_perf_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        run_vitest_benchmarking_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        call_args = mock_subprocess_run.call_args
        env = call_args.kwargs.get("env", {})

        assert env["CODEFLASH_MODE"] == "performance"

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_sets_loop_configuration(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should set loop configuration environment variables."""
        mock_tmp_file.return_value = tmp_path / "vitest_perf_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        run_vitest_benchmarking_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            min_loops=10,
            max_loops=50,
            target_duration_ms=5000,
        )

        call_args = mock_subprocess_run.call_args
        env = call_args.kwargs.get("env", {})

        assert env["CODEFLASH_PERF_MIN_LOOPS"] == "10"
        assert env["CODEFLASH_PERF_LOOP_COUNT"] == "50"
        assert env["CODEFLASH_PERF_TARGET_DURATION_MS"] == "5000"


class TestRunVitestLineProfileTests:
    """Tests for run_vitest_line_profile_tests function."""

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_sets_line_profile_mode(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should set CODEFLASH_MODE to 'line_profile'."""
        mock_tmp_file.return_value = tmp_path / "vitest_line_profile_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')

        run_vitest_line_profile_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        call_args = mock_subprocess_run.call_args
        env = call_args.kwargs.get("env", {})

        assert env["CODEFLASH_MODE"] == "line_profile"

    @patch("subprocess.run")
    @patch("codeflash.languages.javascript.vitest_runner._ensure_runtime_files")
    @patch("codeflash.languages.javascript.vitest_runner.get_run_tmp_file")
    def test_sets_line_profile_output_file(
        self,
        mock_tmp_file: MagicMock,
        mock_ensure_runtime: MagicMock,
        mock_subprocess_run: MagicMock,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should set CODEFLASH_LINE_PROFILE_OUTPUT when provided."""
        mock_tmp_file.return_value = tmp_path / "vitest_line_profile_results.xml"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        (tmp_path / "package.json").write_text('{"name": "test"}')
        line_profile_output = tmp_path / "line_profile.json"

        run_vitest_line_profile_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            line_profile_output_file=line_profile_output,
        )

        call_args = mock_subprocess_run.call_args
        env = call_args.kwargs.get("env", {})

        assert env["CODEFLASH_LINE_PROFILE_OUTPUT"] == str(line_profile_output)


class TestVitestVsJestCommandDifferences:
    """Tests documenting the key differences between Vitest and Jest commands."""

    def test_vitest_uses_run_subcommand(self) -> None:
        """Vitest uses 'run' for single execution, Jest doesn't need it."""
        test_files = [Path("/project/tests/test.test.ts")]

        vitest_cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        # Vitest: npx vitest run ...
        # Jest: npx jest ...
        assert vitest_cmd[0:3] == ["npx", "vitest", "run"]

    def test_vitest_uses_hyphenated_timeout(self) -> None:
        """Vitest uses --test-timeout, Jest uses --testTimeout (camelCase)."""
        test_files = [Path("/project/tests/test.test.ts")]

        vitest_cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        # Vitest: --test-timeout=<ms>
        # Jest: --testTimeout=<ms>
        assert any("--test-timeout=" in arg for arg in vitest_cmd)
        assert not any("--testTimeout=" in arg for arg in vitest_cmd)

    def test_vitest_uses_no_file_parallelism(self) -> None:
        """Vitest uses --no-file-parallelism, Jest uses --runInBand."""
        test_files = [Path("/project/tests/test.test.ts")]

        vitest_cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        # Vitest: --no-file-parallelism
        # Jest: --runInBand
        assert "--no-file-parallelism" in vitest_cmd
        assert "--runInBand" not in vitest_cmd

    def test_vitest_uses_output_file_flag(self) -> None:
        """Vitest uses --outputFile for JUnit output path."""
        test_files = [Path("/project/tests/test.test.ts")]

        vitest_cmd = _build_vitest_behavioral_command(
            test_files, timeout=60, output_file=Path("/tmp/results.xml")
        )

        # Vitest: --outputFile=/tmp/results.xml
        # Jest: uses JEST_JUNIT_OUTPUT_FILE env var
        assert any("--outputFile=" in arg for arg in vitest_cmd)

    def test_vitest_positional_test_files(self) -> None:
        """Vitest uses positional args for test files, not --runTestsByPath."""
        test_files = [Path("/project/tests/test.test.ts")]

        vitest_cmd = _build_vitest_behavioral_command(test_files, timeout=60)

        # Vitest: files are positional
        # Jest: --runTestsByPath <files>
        assert "--runTestsByPath" not in vitest_cmd
        # Test files should be at the end
        assert str(Path("/project/tests/test.test.ts").resolve()) in vitest_cmd
