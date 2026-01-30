"""Tests for JavaScript/TypeScript support.py test framework dispatch logic.

These tests verify that run_behavioral_tests, run_benchmarking_tests, and
run_line_profile_tests correctly dispatch to Jest or Vitest based on the
test_framework parameter.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport


@pytest.fixture
def js_support() -> JavaScriptSupport:
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


@pytest.fixture
def ts_support() -> TypeScriptSupport:
    """Create a TypeScriptSupport instance."""
    return TypeScriptSupport()


@pytest.fixture
def mock_test_paths() -> MagicMock:
    """Create a mock TestFiles object."""
    mock = MagicMock()
    mock_file = MagicMock()
    mock_file.instrumented_behavior_file_path = Path("/project/tests/test_func.test.ts")
    mock_file.benchmarking_file_path = Path("/project/tests/test_func__perf.test.ts")
    mock.test_files = [mock_file]
    return mock


class TestBehavioralTestsDispatch:
    """Tests for run_behavioral_tests dispatch logic."""

    @patch("codeflash.languages.javascript.test_runner.run_jest_behavioral_tests")
    def test_dispatches_to_jest_by_default(
        self,
        mock_jest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.test_runner.run_jest_behavioral_tests")
    def test_dispatches_to_jest_explicitly(
        self,
        mock_jest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Jest when test_framework='jest'."""
        mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="jest",
        )

        mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_behavioral_tests")
    def test_dispatches_to_vitest(
        self,
        mock_vitest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
        )

        mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_behavioral_tests")
    def test_typescript_support_dispatches_to_vitest(
        self,
        mock_vitest_runner: MagicMock,
        ts_support: TypeScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """TypeScriptSupport should also dispatch to Vitest when test_framework='vitest'."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)
        (tmp_path / "package.json").write_text('{"name": "test"}')

        ts_support.run_behavioral_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
        )

        mock_vitest_runner.assert_called_once()


class TestBenchmarkingTestsDispatch:
    """Tests for run_benchmarking_tests dispatch logic."""

    @patch("codeflash.languages.javascript.test_runner.run_jest_benchmarking_tests")
    def test_dispatches_to_jest_by_default(
        self,
        mock_jest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_benchmarking_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_benchmarking_tests")
    def test_dispatches_to_vitest(
        self,
        mock_vitest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_benchmarking_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
        )

        mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_benchmarking_tests")
    def test_passes_loop_parameters(
        self,
        mock_vitest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should pass loop parameters to Vitest runner."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_benchmarking_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
            min_loops=10,
            max_loops=50,
            target_duration_seconds=5.0,
        )

        call_kwargs = mock_vitest_runner.call_args.kwargs
        assert call_kwargs["min_loops"] == 10
        assert call_kwargs["max_loops"] == 50
        assert call_kwargs["target_duration_ms"] == 5000


class TestLineProfileTestsDispatch:
    """Tests for run_line_profile_tests dispatch logic."""

    @patch("codeflash.languages.javascript.test_runner.run_jest_line_profile_tests")
    def test_dispatches_to_jest_by_default(
        self,
        mock_jest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_line_profile_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
        )

        mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_line_profile_tests")
    def test_dispatches_to_vitest(
        self,
        mock_vitest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')

        js_support.run_line_profile_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
        )

        mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_line_profile_tests")
    def test_passes_line_profile_output_file(
        self,
        mock_vitest_runner: MagicMock,
        js_support: JavaScriptSupport,
        mock_test_paths: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should pass line_profile_output_file to Vitest runner."""
        mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())
        (tmp_path / "package.json").write_text('{"name": "test"}')
        output_file = tmp_path / "line_profile.json"

        js_support.run_line_profile_tests(
            test_paths=mock_test_paths,
            test_env={},
            cwd=tmp_path,
            project_root=tmp_path,
            test_framework="vitest",
            line_profile_output_file=output_file,
        )

        call_kwargs = mock_vitest_runner.call_args.kwargs
        assert call_kwargs["line_profile_output_file"] == output_file


class TestTestFrameworkProperty:
    """Tests for test_framework property."""

    def test_javascript_default_framework_is_jest(self, js_support: JavaScriptSupport) -> None:
        """JavaScriptSupport should have Jest as default test framework."""
        assert js_support.test_framework == "jest"

    def test_typescript_default_framework_is_jest(self, ts_support: TypeScriptSupport) -> None:
        """TypeScriptSupport should have Jest as default test framework."""
        assert ts_support.test_framework == "jest"
