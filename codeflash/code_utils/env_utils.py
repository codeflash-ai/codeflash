import logging
import os
import sys
from functools import lru_cache
from typing import Optional

import click
import git

from codeflash.code_utils.shell_utils import read_api_key_from_shell_config


@lru_cache(maxsize=1)
def get_codeflash_api_key() -> Optional[str]:
    api_key = os.environ.get("CODEFLASH_API_KEY") or read_api_key_from_shell_config()
    if not api_key:
        raise EnvironmentError(
            "I didn't find a Codeflash API key in your environment.\n"
            + "You can generate one at https://app.codeflash.ai/app/apikeys,\n"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
    if not api_key.startswith("cf-"):
        raise EnvironmentError(
            f"Your Codeflash API key seems to be invalid. It should start with a 'cf-' prefix; I found '{api_key}' instead.\n"
            + "You can generate one at https://app.codeflash.ai/app/apikeys,\n"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
    return api_key


def ensure_codeflash_api_key() -> bool:
    try:
        get_codeflash_api_key()
    except EnvironmentError as e:
        logging.error(
            "Codeflash API key not found in your environment.\n"
            + "You can generate one at https://app.codeflash.ai/app/apikeys,\n"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
        return False
    return True


@lru_cache(maxsize=1)
def get_codeflash_org_key() -> Optional[str]:
    api_key = os.environ.get("CODEFLASH_ORG_KEY")
    return api_key


@lru_cache(maxsize=1)
def get_pr_number() -> Optional[int]:
    pr_number = os.environ.get("CODEFLASH_PR_NUMBER")
    if not pr_number:
        return None
    else:
        return int(pr_number)


def ensure_pr_number() -> bool:
    if not get_pr_number():
        raise EnvironmentError(
            f"CODEFLASH_PR_NUMBER not found in environment variables; make sure the Github Action is setting this so Codeflash can comment on the right PR"
        )
    return True


def ensure_git_repo(module_root: str) -> bool:
    try:
        _ = git.Repo(module_root, search_parent_directories=True).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        # Only ask for the prompt if running in non-interactive mode
        if sys.__stdin__.isatty():
            response = click.prompt(
                "I did not find a git repository for the code. If you run codeflash, it might overwrite the"
                " code and you might irreversibly lose your current code. Proceed?",
                type=click.Choice(["yes", "no"], case_sensitive=False),
                show_choices=True,
            )
            if response == "no":
                return False
            if response == "yes":
                return True
        else:
            # continue running, important for GitHub actions
            return True
