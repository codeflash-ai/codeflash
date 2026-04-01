"""Tests for Java project initialization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestInitJavaNonInteractive:
    """Test that non_interactive=True makes Java init fully non-interactive."""

    def test_non_interactive_skips_prompts_and_uses_auto_detected(self, tmp_path: Path) -> None:
        project_dir = (tmp_path / "java-project").resolve()
        project_dir.mkdir()
        (project_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        (project_dir / "src" / "main" / "java").mkdir(parents=True)
        (project_dir / "src" / "test" / "java").mkdir(parents=True)

        with (
            patch("codeflash.cli_cmds.init_java.Path.cwd", return_value=project_dir),
            patch("codeflash.cli_cmds.init_auth.prompt_api_key", return_value=False),
            patch("codeflash.cli_cmds.init_auth.install_github_app"),
            patch("codeflash.cli_cmds.github_workflow.install_github_actions"),
            patch("codeflash.cli_cmds.init_java.configure_java_project", return_value=True) as mock_configure,
            patch("codeflash.cli_cmds.init_java.ph"),
            patch("codeflash.cli_cmds.init_java.console"),
            patch("codeflash.cli_cmds.init_java.sys") as mock_sys,
        ):
            mock_sys.exit = MagicMock(side_effect=SystemExit(0))

            from codeflash.cli_cmds.init_java import JavaSetupInfo, init_java_project

            try:
                init_java_project(non_interactive=True)
            except SystemExit:
                pass

            mock_configure.assert_called_once()
            setup_info = mock_configure.call_args[0][0]
            assert isinstance(setup_info, JavaSetupInfo)
            assert setup_info.module_root_override is None
            assert setup_info.test_root_override is None
            assert setup_info.formatter_override is None
            assert setup_info.git_remote == "origin"

    def test_non_interactive_skips_should_modify_check(self, tmp_path: Path) -> None:
        project_dir = (tmp_path / "java-project").resolve()
        project_dir.mkdir()
        (project_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
        (project_dir / "src" / "main" / "java").mkdir(parents=True)
        (project_dir / "src" / "test" / "java").mkdir(parents=True)

        with (
            patch("codeflash.cli_cmds.init_java.Path.cwd", return_value=project_dir),
            patch("codeflash.cli_cmds.init_auth.prompt_api_key", return_value=False),
            patch("codeflash.cli_cmds.init_auth.install_github_app"),
            patch("codeflash.cli_cmds.github_workflow.install_github_actions"),
            patch("codeflash.cli_cmds.init_java.configure_java_project", return_value=True),
            patch("codeflash.cli_cmds.init_java.should_modify_java_config") as mock_should_modify,
            patch("codeflash.cli_cmds.init_java.ph"),
            patch("codeflash.cli_cmds.init_java.console"),
            patch("codeflash.cli_cmds.init_java.sys") as mock_sys,
        ):
            mock_sys.exit = MagicMock(side_effect=SystemExit(0))

            from codeflash.cli_cmds.init_java import init_java_project

            try:
                init_java_project(non_interactive=True)
            except SystemExit:
                pass

            mock_should_modify.assert_not_called()
