"""Go project initialization for Codeflash."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import click
import inquirer
from git import InvalidGitRepositoryError, Repo
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeflash.cli_cmds.console import console
from codeflash.code_utils.compat import LF
from codeflash.code_utils.git_utils import get_git_remotes
from codeflash.code_utils.shell_utils import get_shell_rc_path, is_powershell
from codeflash.languages.golang.config import detect_go_project, detect_go_version
from codeflash.telemetry.posthog_cf import ph


@dataclass(frozen=True)
class GoSetupInfo:
    module_root_override: Union[str, None] = None
    test_root_override: Union[str, None] = None
    formatter_override: Union[list[str], None] = None
    git_remote: str = "origin"
    disable_telemetry: bool = False
    ignore_paths: list[str] | None = None


def _get_theme() -> Any:
    from codeflash.cli_cmds.init_config import CodeflashTheme

    return CodeflashTheme()


def init_go_project() -> None:
    from codeflash.cli_cmds.github_workflow import install_github_actions
    from codeflash.cli_cmds.init_auth import install_github_app, prompt_api_key

    lang_panel = Panel(
        Text(
            "Go project detected!\n\nI'll help you set up Codeflash for your project.", style="cyan", justify="center"
        ),
        title="Go Setup",
        border_style="bright_cyan",
    )
    console.print(lang_panel)
    console.print()

    did_add_new_key = prompt_api_key()

    setup_info = collect_go_setup_info()
    git_remote = setup_info.git_remote or "origin"

    install_github_app(git_remote)

    install_github_actions(override_formatter_check=True)

    usage_table = Table(show_header=False, show_lines=False, border_style="dim")
    usage_table.add_column("Command", style="cyan")
    usage_table.add_column("Description", style="white")

    usage_table.add_row("codeflash --file <path-to-file> --function <function-name>", "Optimize a specific function")
    usage_table.add_row("codeflash --all", "Optimize all functions in all files")
    usage_table.add_row("codeflash --help", "See all available options")

    completion_message = "Codeflash is now set up for your Go project!\n\nYou can now run any of these commands:"

    if did_add_new_key:
        completion_message += (
            "\n\nDon't forget to restart your shell to load the CODEFLASH_API_KEY environment variable!"
        )
        if os.name == "nt":
            reload_cmd = f". {get_shell_rc_path()}" if is_powershell() else f"call {get_shell_rc_path()}"
        else:
            reload_cmd = f"source {get_shell_rc_path()}"
        completion_message += f"\nOr run: {reload_cmd}"

    completion_panel = Panel(
        Group(Text(completion_message, style="bold green"), Text(""), usage_table),
        title="Setup Complete!",
        border_style="bright_green",
        padding=(1, 2),
    )
    console.print(completion_panel)

    ph("cli-go-installation-successful", {"did_add_new_key": did_add_new_key})
    sys.exit(0)


def collect_go_setup_info() -> GoSetupInfo:

    from codeflash.cli_cmds.init_config import ask_for_telemetry

    curdir = Path.cwd()

    if not os.access(curdir, os.W_OK):
        click.echo(f"The current directory isn't writable, please check your folder permissions and try again.{LF}")
        sys.exit(1)

    config = detect_go_project(curdir)
    module_path = config.module_path if config else "unknown"
    go_version = (config.go_version if config else None) or detect_go_version() or "unknown"
    has_vendor = config.has_vendor if config else False

    detection_table = Table(show_header=False, box=None, padding=(0, 2))
    detection_table.add_column("Setting", style="cyan")
    detection_table.add_column("Value", style="green")
    detection_table.add_row("Module", module_path)
    detection_table.add_row("Go version", go_version)
    detection_table.add_row("Source root", ".")
    detection_table.add_row("Test root", ". (co-located)")
    detection_table.add_row("Formatter", "gofmt")
    if has_vendor:
        detection_table.add_row("Vendor", "yes (vendor/ detected)")

    detection_panel = Panel(
        Group(Text("Auto-detected settings for your Go project:\n", style="cyan"), detection_table),
        title="Auto-Detection Results",
        border_style="bright_blue",
    )
    console.print(detection_panel)
    console.print()

    git_remote = _get_git_remote_for_setup()

    disable_telemetry = not ask_for_telemetry()

    return GoSetupInfo(git_remote=git_remote, disable_telemetry=disable_telemetry)


def _get_git_remote_for_setup() -> str:
    try:
        repo = Repo(Path.cwd(), search_parent_directories=True)
        git_remotes = get_git_remotes(repo)
        if not git_remotes:
            return ""

        if len(git_remotes) == 1:
            return git_remotes[0]

        git_panel = Panel(
            Text(
                "Configure Git Remote for Pull Requests.\n\nCodeflash will use this remote to create pull requests.",
                style="blue",
            ),
            title="Git Remote Setup",
            border_style="bright_blue",
        )
        console.print(git_panel)
        console.print()

        git_questions = [
            inquirer.List(
                "git_remote",
                message="Which git remote should Codeflash use?",
                choices=git_remotes,
                default="origin",
                carousel=True,
            )
        ]

        git_answers = inquirer.prompt(git_questions, theme=_get_theme())
        return git_answers["git_remote"] if git_answers else git_remotes[0]
    except InvalidGitRepositoryError:
        return ""


def get_go_runtime_setup_steps() -> str:
    return """- name: Set up Go
        uses: actions/setup-go@v5
        with:
          go-version: 'stable'"""


def get_go_dependency_installation_commands() -> str:
    return "go mod download"


def get_go_test_command() -> str:
    return "go test ./..."
