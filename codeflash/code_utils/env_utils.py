import logging
import os
import re
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def get_codeflash_api_key() -> Optional[str]:
    api_key = os.environ.get("CODEFLASH_API_KEY") or read_api_key_from_shell_config()
    if not api_key:
        raise EnvironmentError(
            "I didn't find a CodeFlash API key in your environment.\n"
            + "You can generate one at https://app.codeflash.ai/app/apikeys,\n"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
    if not api_key.startswith("cf-"):
        raise EnvironmentError(
            f"Your CodeFlash API key seems to be invalid. It should start with a 'cf-' prefix; I found '{api_key}' instead.\n"
            + "You can generate one at https://app.codeflash.ai/app/apikeys,\n"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
    return api_key


def ensure_codeflash_api_key() -> bool:
    try:
        get_codeflash_api_key()
    except EnvironmentError as e:
        logging.error(
            "CodeFlash API key not found in your environment.\n"
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
            f"CODEFLASH_PR_NUMBER not found in environment variables; make sure the Github Action is setting this so CodeFlash can comment on the right PR"
        )
    return True


SHELL_RC_EXPORT_PATTERN = re.compile(r'^export CODEFLASH_API_KEY="?(.*)"?$', re.M)


def read_api_key_from_shell_config() -> Optional[str]:
    shell_rc_path = get_shell_rc_path()
    with open(shell_rc_path, "r", encoding="utf8") as shell_rc:
        shell_contents = shell_rc.read()
        match = SHELL_RC_EXPORT_PATTERN.search(shell_contents)
        return match.group(1) if match else None


def get_shell_rc_path() -> str:
    """Get the path to the user's shell configuration file."""
    shell = os.environ.get("SHELL", "/bin/bash").split("/")[-1]
    if shell == "bash":
        shell_rc_filename = ".bashrc"
    elif shell == "zsh":
        shell_rc_filename = ".zshrc"
    elif shell == "ksh":
        shell_rc_filename = ".kshrc"
    elif shell == "csh" or shell == "tcsh":
        shell_rc_filename = ".cshrc"
    elif shell == "dash":
        shell_rc_filename = ".profile"
    else:
        shell_rc_filename = ".bashrc"  # default to bash if unknown shell
    return os.path.expanduser(f"~/{shell_rc_filename}")
