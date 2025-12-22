from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING, Optional

from codeflash.api.cfapi import is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.cli_cmds.console import console, paneled_text
from codeflash.code_utils.compat import LF
from codeflash.code_utils.git_utils import get_git_remotes, get_repo_owner_and_name

if TYPE_CHECKING:
    import git
    from git import Repo


def get_github_secrets_page_url(repo: Optional[Repo] = None) -> str:
    owner, repo_name = get_repo_owner_and_name(repo)
    return f"https://github.com/{owner}/{repo_name}/settings/secrets/actions"


def require_github_app_or_exit(owner: str, repo: str) -> None:
    # Suppress low-level HTTP error logging to avoid duplicate logs; we present a friendly panel instead
    if not is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
        # Show a clear, user-friendly panel instead of raw error logs
        message = (
            f"It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo} "
            f"or the GitHub account linked to your CODEFLASH_API_KEY does not have access to the repository {owner}/{repo}.{LF}{LF}"
            "To continue, install the Codeflash GitHub App on your repository:"
            f"{LF}https://github.com/apps/codeflash-ai/installations/select_target{LF}{LF}"
            "Tip: If you want to find optimizations without opening PRs, run Codeflash with the --no-pr flag."
        )
        paneled_text(
            message,
            panel_args={"title": "GitHub App Required", "border_style": "red", "expand": False},
            text_args={"style": "bold red"},
        )
        apologize_and_exit()


def github_pr_url(owner: str, repo: str, pr_number: str) -> str:
    return f"https://github.com/{owner}/{repo}/pull/{pr_number}"


def prompt_github_app_install(owner: str, repo: str) -> None:
    """Prompt user to install GitHub app."""
    # Avoid circular import
    from codeflash.cli_cmds import themed_prompts as prompts

    app_url = "https://github.com/apps/codeflash-ai/installations/select_target"

    open_page = prompts.select_or_exit(
        f"Open GitHub App installation page? ({app_url})",
        choices=["Yes", "No"],
        default="Yes",
        header=(
            f"🐙 GitHub App Installation\n\n"
            f"You'll need to install the Codeflash GitHub app for {owner}/{repo}.\n\n"
            "I'll open the installation page where you can select your repository."
        ),
    )
    if open_page == "Yes":
        webbrowser.open(app_url)

    prompts.confirm("Continue once you've completed the installation?", default=True)


def install_github_app(git_repo: git.Repo, git_remote: str = "origin") -> None:
    """Install GitHub app with user prompts and verification."""
    if git_remote not in get_git_remotes(git_repo):
        console.print(f"Skipping GitHub app installation, remote ({git_remote}) does not exist in this repository.")
        return

    owner, repo = get_repo_owner_and_name(git_repo, git_remote)

    if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
        console.print(
            f"🐙 Looks like you've already installed the Codeflash GitHub app on this repository ({owner}/{repo})! Continuing…"
        )
        return

    # Not installed - prompt for installation
    try:
        prompt_github_app_install(owner, repo)

        # Verify installation with retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            if is_github_app_installed_on_repo(owner, repo, suppress_errors=True):
                console.print(f"✅ GitHub App installed successfully for {owner}/{repo}!")
                break

            if attempt == max_retries:
                console.print(
                    f"❌ GitHub App not detected on {owner}/{repo}.\n"
                    f"You won't be able to create PRs until you install it.\n"
                    f"Use the '--no-pr' flag for local-only optimizations."
                )
                break

            console.print(
                f"❌ GitHub App not detected on {owner}/{repo}.\nPress Enter to check again after installation..."
            )
            console.input()
    except (KeyboardInterrupt, EOFError):
        console.print()  # Clean line for next prompt
