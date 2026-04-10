from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from codeflash.languages.java.gradle_strategy import GradleStrategy

if TYPE_CHECKING:
    from pathlib import Path

COD_FLAG = "--configure-on-demand"
MOCK_TARGET = "codeflash.languages.java.test_runner._run_cmd_kill_pg_on_timeout"


class TestConfigureOnDemand:
    def test_compile_tests_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        with patch.object(strategy, "find_executable", return_value="gradlew"), patch(MOCK_TARGET) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            strategy.compile_tests(tmp_path, {}, test_module=None)
        cmd = mock_run.call_args[0][0]
        assert COD_FLAG in cmd

    def test_compile_tests_multimodule_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        with patch.object(strategy, "find_executable", return_value="gradlew"), patch(MOCK_TARGET) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            strategy.compile_tests(tmp_path, {}, test_module="core")
        cmd = mock_run.call_args[0][0]
        assert COD_FLAG in cmd
        assert ":core:testClasses" in cmd

    def test_compile_source_only_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        with patch.object(strategy, "find_executable", return_value="gradlew"), patch(MOCK_TARGET) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            strategy.compile_source_only(tmp_path, {}, test_module=None)
        cmd = mock_run.call_args[0][0]
        assert COD_FLAG in cmd

    def test_get_test_run_command_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        with patch.object(strategy, "find_executable", return_value="gradlew"):
            cmd = strategy.get_test_run_command(tmp_path)
        assert COD_FLAG in cmd

    def test_install_multi_module_deps_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        with (
            patch.object(strategy, "find_executable", return_value="gradlew"),
            patch(MOCK_TARGET) as mock_run,
            patch("codeflash.languages.java.gradle_strategy._multimodule_deps_installed", set()),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            strategy.install_multi_module_deps(tmp_path, test_module="core", env={})
        cmd = mock_run.call_args[0][0]
        assert COD_FLAG in cmd
        assert ":core:testClasses" in cmd

    def test_run_tests_via_build_tool_includes_configure_on_demand(self, tmp_path: Path) -> None:
        strategy = GradleStrategy()
        reports_dir = tmp_path / "build" / "test-results" / "test"
        reports_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(strategy, "find_executable", return_value="gradlew"), patch(MOCK_TARGET) as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            strategy.run_tests_via_build_tool(
                build_root=tmp_path,
                test_paths=["com.example.TestFoo"],
                env={},
                timeout=60,
                mode="behavior",
                test_module=None,
            )
        cmd = mock_run.call_args[0][0]
        assert COD_FLAG in cmd
