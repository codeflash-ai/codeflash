from __future__ import annotations

from argparse import Namespace
from unittest.mock import Mock

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
