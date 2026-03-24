from __future__ import annotations

import os

import click

from codeflash.cli_cmds.console import console
from codeflash.cli_cmds.oauth_handler import perform_oauth_signin
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.code_utils.shell_utils import get_shell_rc_path, save_api_key_to_rc
from codeflash.either import is_successful


def auth_login() -> None:
    """Perform OAuth login and save the API key."""
    try:
        existing_api_key = get_codeflash_api_key()
    except OSError:
        existing_api_key = None

    if existing_api_key:
        display_key = f"{existing_api_key[:3]}****{existing_api_key[-4:]}"
        console.print(f"[green]Already authenticated with API key {display_key}[/green]")
        console.print("To re-authenticate, unset [bold]CODEFLASH_API_KEY[/bold] and run this command again.")
        return

    api_key = perform_oauth_signin()
    if not api_key:
        click.echo("Authentication failed.")
        raise SystemExit(1)

    shell_rc_path = get_shell_rc_path()
    if not shell_rc_path.exists() and os.name == "nt":
        shell_rc_path.touch()

    result = save_api_key_to_rc(api_key)
    if is_successful(result):
        click.echo(result.unwrap())
    else:
        click.echo(result.failure())

    os.environ["CODEFLASH_API_KEY"] = api_key
    console.print("[green]Signed in successfully![/green]")
