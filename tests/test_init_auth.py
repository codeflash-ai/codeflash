from __future__ import annotations

from unittest.mock import Mock, patch

from codeflash.cli_cmds.init_auth import install_github_app


def _echo_messages(mock: Mock) -> list[str]:
    return [str(call.args[0]) for call in mock.call_args_list if call.args]


def _mock_repo_context(monkeypatch) -> None:
    git_repo = object()
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.git.Repo", Mock(return_value=git_repo))
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.get_git_remotes", Mock(return_value=["origin"]))
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.get_repo_owner_and_name", Mock(return_value=("octocat", "demo")))


def test_install_github_app_allows_user_to_skip(monkeypatch) -> None:
    _mock_repo_context(monkeypatch)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.is_github_app_installed_on_repo", Mock(return_value=False))
    echo = Mock()
    prompt = Mock()
    launch = Mock()
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.echo", echo)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.prompt", prompt)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.launch", launch)

    with patch("rich.prompt.Confirm.ask", return_value=False) as confirm_ask:
        install_github_app("origin")

    confirm_ask.assert_called_once()
    prompt.assert_not_called()
    launch.assert_not_called()
    assert any("Skipping Codeflash GitHub app installation for octocat/demo." in message for message in _echo_messages(echo))


def test_install_github_app_skips_on_noninteractive_abort(monkeypatch) -> None:
    _mock_repo_context(monkeypatch)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.is_github_app_installed_on_repo", Mock(return_value=False))
    echo = Mock()
    prompt = Mock()
    launch = Mock()
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.echo", echo)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.prompt", prompt)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.launch", launch)

    with patch("rich.prompt.Confirm.ask", side_effect=EOFError):
        install_github_app("origin")

    prompt.assert_not_called()
    launch.assert_not_called()
    assert any("Skipping Codeflash GitHub app installation for octocat/demo." in message for message in _echo_messages(echo))


def test_install_github_app_still_runs_install_flow_when_confirmed(monkeypatch) -> None:
    _mock_repo_context(monkeypatch)
    monkeypatch.setattr(
        "codeflash.cli_cmds.init_auth.is_github_app_installed_on_repo",
        Mock(side_effect=[False, True]),
    )
    echo = Mock()
    prompt = Mock(return_value="")
    launch = Mock()
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.echo", echo)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.prompt", prompt)
    monkeypatch.setattr("codeflash.cli_cmds.init_auth.click.launch", launch)

    with patch("rich.prompt.Confirm.ask", return_value=True) as confirm_ask:
        install_github_app("origin")

    confirm_ask.assert_called_once()
    launch.assert_called_once_with("https://github.com/apps/codeflash-ai/installations/select_target")
    assert prompt.call_count == 2
