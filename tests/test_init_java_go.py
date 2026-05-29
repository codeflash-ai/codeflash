from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from codeflash.cli_cmds.init_go import collect_go_setup_info
from codeflash.cli_cmds.init_java import collect_java_setup_info, should_modify_java_config


def test_collect_go_setup_info_skip_confirm_uses_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.21\n", encoding="utf-8")

    get_git_remote = Mock(return_value="origin")
    monkeypatch.setattr("codeflash.cli_cmds.init_go._get_git_remote_for_setup", get_git_remote)
    monkeypatch.setattr(
        "codeflash.cli_cmds.init_config.ask_for_telemetry",
        Mock(side_effect=AssertionError("ask_for_telemetry should not be called")),
    )

    with patch("codeflash.cli_cmds.init_go.inquirer") as mock_inquirer:
        setup_info = collect_go_setup_info(skip_confirm=True)

    mock_inquirer.prompt.assert_not_called()
    get_git_remote.assert_called_once_with(skip_confirm=True)
    assert setup_info.git_remote == "origin"
    assert setup_info.disable_telemetry is False


def test_collect_go_setup_info_uses_default_telemetry_on_eof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.21\n", encoding="utf-8")

    get_git_remote = Mock(return_value="origin")
    monkeypatch.setattr("codeflash.cli_cmds.init_go._get_git_remote_for_setup", get_git_remote)

    with patch("rich.prompt.Confirm.ask", side_effect=EOFError):
        setup_info = collect_go_setup_info()

    get_git_remote.assert_called_once_with()
    assert setup_info.git_remote == "origin"
    assert setup_info.disable_telemetry is False


def test_collect_java_setup_info_skip_confirm_uses_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (tmp_path / "src" / "main" / "java").mkdir(parents=True)
    (tmp_path / "src" / "test" / "java").mkdir(parents=True)

    get_git_remote = Mock(return_value="origin")
    monkeypatch.setattr("codeflash.cli_cmds.init_java._get_git_remote_for_setup", get_git_remote)
    monkeypatch.setattr(
        "codeflash.cli_cmds.init_config.ask_for_telemetry",
        Mock(side_effect=AssertionError("ask_for_telemetry should not be called")),
    )

    with patch("codeflash.cli_cmds.init_java.inquirer") as mock_inquirer:
        setup_info = collect_java_setup_info(skip_confirm=True)

    mock_inquirer.prompt.assert_not_called()
    get_git_remote.assert_called_once_with(skip_confirm=True)
    assert setup_info.module_root_override is None
    assert setup_info.test_root_override is None
    assert setup_info.formatter_override is None
    assert setup_info.git_remote == "origin"
    assert setup_info.disable_telemetry is False


def test_collect_java_setup_info_uses_defaults_on_eof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (tmp_path / "src" / "main" / "java").mkdir(parents=True)
    (tmp_path / "src" / "test" / "java").mkdir(parents=True)

    get_git_remote = Mock(return_value="origin")
    monkeypatch.setattr("codeflash.cli_cmds.init_java._get_git_remote_for_setup", get_git_remote)
    monkeypatch.setattr("codeflash.cli_cmds.init_config.ask_for_telemetry", Mock(return_value=True))

    with patch("codeflash.cli_cmds.init_java.inquirer") as mock_inquirer, patch(
        "rich.prompt.Confirm.ask", side_effect=EOFError
    ):
        setup_info = collect_java_setup_info()

    mock_inquirer.prompt.assert_not_called()
    get_git_remote.assert_called_once_with()
    assert setup_info.module_root_override is None
    assert setup_info.test_root_override is None
    assert setup_info.formatter_override is None
    assert setup_info.git_remote == "origin"
    assert setup_info.disable_telemetry is False


def test_should_modify_java_config_skip_confirm_skips_reconfigure_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (tmp_path / "gradle.properties").write_text("codeflash.moduleRoot=src/main/java\n", encoding="utf-8")

    with patch("rich.prompt.Confirm.ask", side_effect=AssertionError("Confirm.ask should not be called")):
        should_modify, config = should_modify_java_config(skip_confirm=True)

    assert should_modify is False
    assert config is None


def test_should_modify_java_config_uses_default_on_eof(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")
    (tmp_path / "gradle.properties").write_text("codeflash.moduleRoot=src/main/java\n", encoding="utf-8")

    with patch("rich.prompt.Confirm.ask", side_effect=EOFError):
        should_modify, config = should_modify_java_config()

    assert should_modify is False
    assert config is None
