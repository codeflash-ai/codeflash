import logging
from typing import Optional

from git import Repo

from codeflash.api.cfapi import is_github_app_installed_on_repo
from codeflash.cli_cmds.cli_common import apologize_and_exit
from codeflash.code_utils.compat import LF
from codeflash.code_utils.git_utils import get_repo_owner_and_name


def get_github_secrets_page_url(repo: Optional[Repo] = None) -> str:
    owner, repo_name = get_repo_owner_and_name(repo)
    return f"https://github.com/{owner}/{repo_name}/settings/secrets/actions"


def require_github_app_or_exit(owner: str, repo: str) -> None:
    if not is_github_app_installed_on_repo(owner, repo):
        logging.error(
            f"It looks like the Codeflash GitHub App is not installed on the repository {owner}/{repo} or the GitHub"
            f" account linked to your CODEFLASH_API_KEY does not have access to the repository {owner}/{repo}.{LF}"
            "Before continuing, please install the Codeflash GitHub App on your repository by visiting "
            f"https://github.com/apps/codeflash-ai{LF}",
        )
        logging.error(
            f"Note: if you want to find optimizations without opening PRs, you can run Codeflash with the --no-pr flag.{LF}",
        )
        apologize_and_exit()
