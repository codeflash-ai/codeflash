from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash_python.api.cfapi import is_github_app_installed_on_repo
from codeflash_python.cli_common import apologize_and_exit
from codeflash_python.code_utils.compat import LF
from codeflash_python.code_utils.git_utils import get_repo_owner_and_name

logger = logging.getLogger("codeflash_python")

if TYPE_CHECKING:
    from git import Repo


def get_github_secrets_page_url(repo: Repo | None = None) -> str:
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
        logger.warning("GitHub App Required: %s", message)
        apologize_and_exit()


def github_pr_url(owner: str, repo: str, pr_number: str) -> str:
    return f"https://github.com/{owner}/{repo}/pull/{pr_number}"
