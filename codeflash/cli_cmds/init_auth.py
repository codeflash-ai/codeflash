from __future__ import annotations

import os

import click
import git
import inquirer

from codeflash.api.cfapi import get_user_id, is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console
from codeflash.cli_cmds.init_config import CodeflashTheme
from codeflash.cli_cmds.oauth_handler import perform_oauth_signin
from codeflash.code_utils.compat import LF
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.code_utils.git_utils import get_git_remotes, get_repo_owner_and_name
from codeflash.code_utils.shell_utils import get_shell_rc_path, save_api_key_to_rc
from codeflash.either import is_successful
from codeflash.telemetry.posthog_cf import ph


class CFAPIKeyType(click.ParamType):
    name = "cfapi-key"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> str | None:
        value = value.strip()
        if not value.startswith("cf-") and value != "":
            self.fail(
                f"That key [{value}] seems to be invalid. It should start with a 'cf-' prefix. Please try again.",
                param,
                ctx,
            )
        return value


# Returns True if the user entered a new API key, False if they used an existing one
def prompt_api_key() -> bool:
    """Prompt user for API key via OAuth or manual entry."""
    from rich.panel import Panel
    from rich.text import Text

    # Check for existing API key
    try:
        existing_api_key = get_codeflash_api_key()
    except OSError:
        existing_api_key = None

    if existing_api_key:
        display_key = f"{existing_api_key[:3]}****{existing_api_key[-4:]}"
        api_key_panel = Panel(
            Text(
                f"🔑 I found a CODEFLASH_API_KEY in your environment [{display_key}]!\n\n"
                "✅ You're all set with API authentication!",
                style="green",
                justify="center",
            ),
            title="🔑 API Key Found",
            border_style="bright_green",
        )
        console.print(api_key_panel)
        console.print()
        return False

    # Prompt for authentication method
    auth_choices = ["🔐 Login in with Codeflash", "🔑 Use Codeflash API key"]

    questions = [
        inquirer.List(
            "auth_method",
            message="How would you like to authenticate?",
            choices=auth_choices,
            default=auth_choices[0],
            carousel=True,
        )
    ]

    answers = inquirer.prompt(questions, theme=CodeflashTheme())
    if not answers:
        apologize_and_exit()

    method = answers["auth_method"]

    if method == auth_choices[1]:
        enter_api_key_and_save_to_rc()
        ph("cli-new-api-key-entered")
        return True

    # Perform OAuth sign-in
    api_key = perform_oauth_signin()

    if not api_key:
        apologize_and_exit()

    # Save API key
    shell_rc_path = get_shell_rc_path()
    if not shell_rc_path.exists() and os.name == "nt":
        shell_rc_path.touch()
        click.echo(f"✅ Created {shell_rc_path}")

    result = save_api_key_to_rc(api_key)
    if is_successful(result):
        click.echo(result.unwrap())
        click.echo("✅ Signed in successfully and API key saved!")
    else:
        click.echo(result.failure())
        click.pause()

    os.environ["CODEFLASH_API_KEY"] = api_key
    ph("cli-oauth-signin-completed")
    return True


def enter_api_key_and_save_to_rc() -> None:
    browser_launched = False
    api_key = ""
    while api_key == "":
        api_key = click.prompt(
            f"Enter your Codeflash API key{' [or press Enter to open your API key page]' if not browser_launched else ''}",
            hide_input=False,
            default="",
            type=CFAPIKeyType(),
            show_default=False,
        ).strip()
        if api_key:
            break
        if not browser_launched:
            click.echo(
                f"Opening your Codeflash API key page. Grab a key from there!{LF}"
                "You can also open this link manually: https://app.codeflash.ai/app/apikeys"
            )
            click.launch("https://app.codeflash.ai/app/apikeys")
            browser_launched = True  # This does not work on remote consoles
    shell_rc_path = get_shell_rc_path()
    if not shell_rc_path.exists() and os.name == "nt":
        # On Windows, create the appropriate file (PowerShell .ps1 or CMD .bat) in the user's home directory
        shell_rc_path.parent.mkdir(parents=True, exist_ok=True)
        shell_rc_path.touch()
        click.echo(f"✅ Created {shell_rc_path}")
    get_user_id(api_key=api_key)  # Used to verify whether the API key is valid.
    result = save_api_key_to_rc(api_key)
    if is_successful(result):
        click.echo(result.unwrap())
    else:
        click.echo(result.failure())
        click.pause()

    os.environ["CODEFLASH_API_KEY"] = api_key


def install_github_app(git_remote: str) -> None:
    try:
        git_repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        click.echo("Skipping GitHub app installation because you're not in a git repository.")
        return

    if git_remote not in get_git_remotes(git_repo):
        click.echo(f"Skipping GitHub app installation, remote ({git_remote}) does not exist in this repository.")
        return

    owner, repo = get_repo_owner_and_name(git_repo, git_remote)

    if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
        click.echo(
            f"🐙 Looks like you've already installed the Codeflash GitHub app on this repository ({owner}/{repo})! Continuing…"
        )

    else:
        try:
            click.prompt(
                f"Finally, you'll need to install the Codeflash GitHub app by choosing the repository you want to install Codeflash on.{LF}"
                f"I will attempt to open the github app page - https://github.com/apps/codeflash-ai/installations/select_target {LF}"
                f"Please, press ENTER to open the app installation page{LF}",
                default="",
                type=click.STRING,
                prompt_suffix=">>> ",
                show_default=False,
            )
            click.launch("https://github.com/apps/codeflash-ai/installations/select_target")
            click.prompt(
                f"Please, press ENTER once you've finished installing the github app from https://github.com/apps/codeflash-ai/installations/select_target{LF}",
                default="",
                type=click.STRING,
                prompt_suffix=">>> ",
                show_default=False,
            )

            count = 2
            while not is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
                if count == 0:
                    click.echo(
                        f"❌ It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo}.{LF}"
                        f"You won't be able to create PRs with Codeflash until you install the app.{LF}"
                        f"In the meantime you can make local only optimizations by using the '--no-pr' flag with codeflash.{LF}"
                    )
                    break
                click.prompt(
                    f"❌ It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo}.{LF}"
                    f"Please install it from https://github.com/apps/codeflash-ai/installations/select_target {LF}"
                    f"Please, press ENTER to continue once you've finished installing the github app…{LF}",
                    default="",
                    type=click.STRING,
                    prompt_suffix=">>> ",
                    show_default=False,
                )
                count -= 1
        except (KeyboardInterrupt, EOFError, click.exceptions.Abort):
            # leave empty line for the next prompt to be properly rendered
            click.echo()
