"""Tests for JavaScript/TypeScript support.py test framework dispatch logic.

These tests verify that run_behavioral_tests, run_benchmarking_tests, and
run_line_profile_tests correctly dispatch to Jest or Vitest based on the
test_framework parameter or singleton.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport
from codeflash.languages.test_framework import reset_test_framework, set_current_test_framework
from codeflash.models.models import TestFile, TestFiles, TestType


@pytest.fixture
def js_support() -> JavaScriptSupport:
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


@pytest.fixture
def ts_support() -> TypeScriptSupport:
    """Create a TypeScriptSupport instance."""
    return TypeScriptSupport()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the test framework singleton before each test."""
    reset_test_framework()
    yield
    reset_test_framework()


def create_test_files(tmp_path: Path) -> TestFiles:
    """Create a TestFiles object with real file paths."""
    test_file = tmp_path / "tests" / "test_func.test.ts"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("// test file")

    perf_file = tmp_path / "tests" / "test_func__perf.test.ts"
    perf_file.write_text("// perf test file")

    return TestFiles(
        test_files=[
            TestFile(
                instrumented_behavior_file_path=test_file,
                benchmarking_file_path=perf_file,
                test_type=TestType.GENERATED_REGRESSION,
            )
        ]
    )


class TestBehavioralTestsDispatch:
    """Tests for run_behavioral_tests dispatch logic."""

    @patch("codeflash.languages.javascript.test_runner.run_jest_behavioral_tests")
    def test_dispatches_to_jest_by_default(self, mock_jest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            js_support.run_behavioral_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.test_runner.run_jest_behavioral_tests")
    def test_dispatches_to_jest_explicitly(self, mock_jest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Jest when test_framework='jest'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            js_support.run_behavioral_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="jest"
            )

            mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_behavioral_tests")
    def test_dispatches_to_vitest(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            js_support.run_behavioral_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="vitest"
            )

            mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_behavioral_tests")
    def test_typescript_support_dispatches_to_vitest(
        self, mock_vitest_runner: MagicMock, ts_support: TypeScriptSupport
    ) -> None:
        """TypeScriptSupport should also dispatch to Vitest when test_framework='vitest'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            ts_support.run_behavioral_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="vitest"
            )

            mock_vitest_runner.assert_called_once()


class TestBenchmarkingTestsDispatch:
    """Tests for run_benchmarking_tests dispatch logic."""

    @patch("codeflash.languages.javascript.test_runner.run_jest_benchmarking_tests")
    def test_dispatches_to_jest_by_default(self, mock_jest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_benchmarking_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_benchmarking_tests")
    def test_dispatches_to_vitest(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_benchmarking_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="vitest"
            )

            mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_benchmarking_tests")
    def test_passes_loop_parameters(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should pass loop parameters to Vitest runner."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_benchmarking_tests(
                test_paths=test_files,
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
    def test_dispatches_to_jest_by_default(self, mock_jest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Jest when test_framework is not specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_line_profile_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_line_profile_tests")
    def test_dispatches_to_vitest(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """Should dispatch to Vitest when test_framework='vitest'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_line_profile_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="vitest"
            )

            mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_line_profile_tests")
    def test_passes_line_profile_output_file(
        self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport
    ) -> None:
        """Should pass line_profile_output_file to Vitest runner."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)
            output_file = tmp_path / "line_profile.json"

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            js_support.run_line_profile_tests(
                test_paths=test_files,
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


class TestTestFrameworkSingleton:
    """Tests for test_framework singleton behavior."""

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_behavioral_tests")
    def test_uses_singleton_when_param_not_provided(
        self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport
    ) -> None:
        """Should use singleton test_framework when parameter is not provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            set_current_test_framework("vitest")

            js_support.run_behavioral_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.test_runner.run_jest_behavioral_tests")
    def test_explicit_param_overrides_singleton(
        self, mock_jest_runner: MagicMock, js_support: JavaScriptSupport
    ) -> None:
        """Explicit test_framework parameter should override singleton."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_jest_runner.return_value = (tmp_path / "result.xml", MagicMock(), None, None)

            set_current_test_framework("vitest")

            js_support.run_behavioral_tests(
                test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path, test_framework="jest"
            )

            mock_jest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_benchmarking_tests")
    def test_benchmarking_uses_singleton(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """run_benchmarking_tests should use singleton when param not provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            set_current_test_framework("vitest")

            js_support.run_benchmarking_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_vitest_runner.assert_called_once()

    @patch("codeflash.languages.javascript.vitest_runner.run_vitest_line_profile_tests")
    def test_line_profile_uses_singleton(self, mock_vitest_runner: MagicMock, js_support: JavaScriptSupport) -> None:
        """run_line_profile_tests should use singleton when param not provided."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "package.json").write_text('{"name": "test"}')
            test_files = create_test_files(tmp_path)

            mock_vitest_runner.return_value = (tmp_path / "result.xml", MagicMock())

            set_current_test_framework("vitest")

            js_support.run_line_profile_tests(test_paths=test_files, test_env={}, cwd=tmp_path, project_root=tmp_path)

            mock_vitest_runner.assert_called_once()


class TestTestFrameworkSingletonModule:
    """Tests for the test_framework singleton module itself."""

    def test_initial_state_is_none(self) -> None:
        """Singleton should start as None."""
        from codeflash.languages.test_framework import current_test_framework

        assert current_test_framework() is None

    def test_set_and_get(self) -> None:
        """Should be able to set and get test framework."""
        from codeflash.languages.test_framework import current_test_framework, set_current_test_framework

        set_current_test_framework("vitest")
        assert current_test_framework() == "vitest"

    def test_set_only_once(self) -> None:
        """Once set, singleton should not change."""
        from codeflash.languages.test_framework import current_test_framework, set_current_test_framework

        set_current_test_framework("jest")
        set_current_test_framework("vitest")
        assert current_test_framework() == "jest"

    def test_is_jest(self) -> None:
        """is_jest() should return True when framework is Jest."""
        from codeflash.languages.test_framework import is_jest, set_current_test_framework

        set_current_test_framework("jest")
        assert is_jest() is True

    def test_is_vitest(self) -> None:
        """is_vitest() should return True when framework is Vitest."""
        from codeflash.languages.test_framework import is_vitest, set_current_test_framework

        set_current_test_framework("vitest")
        assert is_vitest() is True

    def test_get_js_test_framework_or_default_returns_jest(self) -> None:
        """get_js_test_framework_or_default should return 'jest' when not set."""
        from codeflash.languages.test_framework import get_js_test_framework_or_default

        assert get_js_test_framework_or_default() == "jest"

    def test_get_js_test_framework_or_default_returns_vitest(self) -> None:
        """get_js_test_framework_or_default should return 'vitest' when set."""
        from codeflash.languages.test_framework import get_js_test_framework_or_default, set_current_test_framework

        set_current_test_framework("vitest")
        assert get_js_test_framework_or_default() == "vitest"
