from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codeflash.cli_cmds.cmd_init import init_codeflash
from codeflash.cli_cmds.init_config import should_modify_pyproject_toml
from codeflash.cli_cmds.init_javascript import ProjectLanguage
from codeflash.main import main


def _args(command: str) -> Namespace:
    return Namespace(command=command, yes=True, config_file=None, verify_setup=False)


def test_main_passes_yes_and_api_key_state_to_init(monkeypatch) -> None:
    init_codeflash = Mock()

    monkeypatch.setenv("CODEFLASH_API_KEY", "cf-test-key")
    monkeypatch.setattr("codeflash.main.print_codeflash_banner", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.cli.parse_args", Mock(return_value=_args("init")))
    monkeypatch.setattr("codeflash.code_utils.version_check.check_for_newer_minor_version", Mock())
    monkeypatch.setattr("codeflash.telemetry.sentry.init_sentry", Mock())
    monkeypatch.setattr("codeflash.telemetry.posthog_cf.initialize_posthog", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.init_codeflash", init_codeflash)

    main()

    init_codeflash.assert_called_once_with(skip_confirm=True, skip_api_key=True)


def test_main_passes_yes_to_init_actions(monkeypatch) -> None:
    install_github_actions = Mock()

    monkeypatch.setattr("codeflash.main.print_codeflash_banner", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.cli.parse_args", Mock(return_value=_args("init-actions")))
    monkeypatch.setattr("codeflash.code_utils.version_check.check_for_newer_minor_version", Mock())
    monkeypatch.setattr("codeflash.telemetry.sentry.init_sentry", Mock())
    monkeypatch.setattr("codeflash.telemetry.posthog_cf.initialize_posthog", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.github_workflow.install_github_actions", install_github_actions)

    main()

    install_github_actions.assert_called_once_with(skip_confirm=True)


def test_should_modify_pyproject_toml_skip_confirm_skips_reconfigure_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.codeflash]\nmodule-root = "src"\ntests-root = "tests"\ngit-remote = "upstream"\n',
        encoding="utf-8",
    )

    with patch("rich.prompt.Confirm.ask", side_effect=AssertionError("Confirm.ask should not be called")):
        should_modify, config = should_modify_pyproject_toml(skip_confirm=True)

    assert should_modify is False
    assert config is not None
    assert config["git_remote"] == "upstream"


def test_should_modify_pyproject_toml_uses_default_on_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.codeflash]\nmodule-root = "src"\ntests-root = "tests"\ngit-remote = "upstream"\n',
        encoding="utf-8",
    )

    with patch("rich.prompt.Confirm.ask", side_effect=EOFError):
        should_modify, config = should_modify_pyproject_toml()

    assert should_modify is False
    assert config is not None
    assert config["git_remote"] == "upstream"


def test_init_codeflash_skip_confirm_reuses_existing_python_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.codeflash]\nmodule-root = "src"\ntests-root = "tests"\ngit-remote = "upstream"\n',
        encoding="utf-8",
    )

    install_github_app = Mock()
    install_github_actions = Mock()
    install_vscode_extension = Mock()
    detect_project = Mock(side_effect=AssertionError("detect_project should not be called"))
    write_config = Mock(side_effect=AssertionError("write_config should not be called"))
    exit_mock = Mock(side_effect=SystemExit(0))

    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.detect_project_language", Mock(return_value=ProjectLanguage.PYTHON))
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.install_github_app", install_github_app)
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.install_github_actions", install_github_actions)
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.install_vscode_extension", install_vscode_extension)
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.console.print", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.ph", Mock())
    monkeypatch.setattr("codeflash.cli_cmds.cmd_init.sys.exit", exit_mock)
    monkeypatch.setattr("codeflash.setup.detect_project", detect_project)
    monkeypatch.setattr("codeflash.setup.write_config", write_config)

    with pytest.raises(SystemExit) as exc_info:
        init_codeflash(skip_confirm=True, skip_api_key=True)

    assert exc_info.value.code == 0
    install_github_app.assert_called_once_with("upstream")
    install_github_actions.assert_called_once_with(override_formatter_check=True, skip_confirm=True)
    install_vscode_extension.assert_called_once()
    detect_project.assert_not_called()
    write_config.assert_not_called()
    exit_mock.assert_called_once_with(0)
