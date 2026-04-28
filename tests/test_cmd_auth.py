from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codeflash.cli_cmds.cmd_auth import auth_login
from codeflash.either import Success


class TestAuthLogin:
    @patch("codeflash.code_utils.env_utils.get_codeflash_api_key")
    @patch("codeflash.cli_cmds.console.console")
    def test_existing_api_key_skips_oauth(self, mock_console: MagicMock, mock_get_key: MagicMock) -> None:
        mock_get_key.return_value = "cf-test1234abcd"

        auth_login()

        mock_console.print.assert_any_call("[green]Already authenticated with API key cf-****abcd[/green]")
        mock_console.print.assert_any_call(
            "To re-authenticate, unset [bold]CODEFLASH_API_KEY[/bold] and run this command again."
        )

    @patch("codeflash.code_utils.env_utils.get_codeflash_api_key")
    @patch("codeflash.cli_cmds.console.console")
    def test_existing_api_key_oserror_treated_as_missing(
        self, mock_console: MagicMock, mock_get_key: MagicMock
    ) -> None:
        mock_get_key.side_effect = OSError("permission denied")

        with pytest.raises(SystemExit):
            with patch("codeflash.cli_cmds.oauth_handler.perform_oauth_signin", return_value=None):
                auth_login()

    @patch("codeflash.cli_cmds.oauth_handler.perform_oauth_signin")
    @patch("codeflash.code_utils.env_utils.get_codeflash_api_key", return_value="")
    def test_oauth_failure_exits_with_code_1(self, mock_get_key: MagicMock, mock_oauth: MagicMock) -> None:
        mock_oauth.return_value = None

        with pytest.raises(SystemExit, match="1"):
            auth_login()

    @patch("codeflash.cli_cmds.cmd_auth.os")
    @patch("codeflash.code_utils.shell_utils.save_api_key_to_rc")
    @patch("codeflash.cli_cmds.oauth_handler.perform_oauth_signin")
    @patch("codeflash.code_utils.env_utils.get_codeflash_api_key", return_value="")
    @patch("codeflash.cli_cmds.console.console")
    def test_successful_oauth_saves_key(
        self,
        mock_console: MagicMock,
        mock_get_key: MagicMock,
        mock_oauth: MagicMock,
        mock_save: MagicMock,
        mock_os: MagicMock,
    ) -> None:
        mock_oauth.return_value = "cf-newkey12345678"
        mock_save.return_value = Success("API key saved to ~/.zshrc")

        auth_login()

        mock_save.assert_called_once_with("cf-newkey12345678")
        mock_os.environ.__setitem__.assert_called_once_with("CODEFLASH_API_KEY", "cf-newkey12345678")
        mock_console.print.assert_called_with("[green]Signed in successfully![/green]")

    @patch("codeflash.cli_cmds.cmd_auth.os")
    @patch("codeflash.code_utils.shell_utils.save_api_key_to_rc")
    @patch("codeflash.cli_cmds.oauth_handler.perform_oauth_signin")
    @patch("codeflash.code_utils.env_utils.get_codeflash_api_key", return_value="")
    @patch("codeflash.cli_cmds.console.console")
    def test_windows_oauth_saves_key(
        self,
        mock_console: MagicMock,
        mock_get_key: MagicMock,
        mock_oauth: MagicMock,
        mock_save: MagicMock,
        mock_os: MagicMock,
    ) -> None:
        mock_oauth.return_value = "cf-newkey12345678"
        mock_os.name = "nt"
        mock_save.return_value = Success("API key saved")

        auth_login()

        mock_save.assert_called_once_with("cf-newkey12345678")
        mock_os.environ.__setitem__.assert_called_once_with("CODEFLASH_API_KEY", "cf-newkey12345678")


class TestAuthSubcommandParsing:
    def test_auth_login_parses(self) -> None:
        from codeflash.cli_cmds.cli import _build_parser

        _build_parser.cache_clear()
        parser = _build_parser()
        args = parser.parse_args(["auth", "login"])
        assert args.command == "auth"
        assert args.auth_command == "login"

    def test_auth_without_subcommand(self) -> None:
        from codeflash.cli_cmds.cli import _build_parser

        _build_parser.cache_clear()
        parser = _build_parser()
        args = parser.parse_args(["auth"])
        assert args.command == "auth"
        assert args.auth_command is None
