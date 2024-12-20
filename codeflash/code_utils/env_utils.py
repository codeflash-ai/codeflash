import os
from functools import lru_cache
from typing import Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.shell_utils import read_api_key_from_shell_config


@lru_cache(maxsize=1)
def get_codeflash_api_key() -> str:
    api_key = os.environ.get("CODEFLASH_API_KEY") or read_api_key_from_shell_config()
    if not api_key:
        msg = (
            "I didn't find a Codeflash API key in your environment.\nYou can generate one at "
            "https://app.codeflash.ai/app/apikeys ,\nthen set it as a CODEFLASH_API_KEY environment variable."
        )
        raise OSError(msg)
    if not api_key.startswith("cf-"):
        msg = (
            f"Your Codeflash API key seems to be invalid. It should start with a 'cf-' prefix; I found '{api_key}' "
            f"instead.\nYou can generate one at https://app.codeflash.ai/app/apikeys ,\nthen set it as a "
            f"CODEFLASH_API_KEY environment variable."
        )
        raise OSError(msg)
    return api_key


def ensure_codeflash_api_key() -> bool:
    try:
        get_codeflash_api_key()
    except OSError:
        logger.error(
            "Codeflash API key not found in your environment.\nYou can generate one at "
            "https://app.codeflash.ai/app/apikeys ,\nthen set it as a CODEFLASH_API_KEY environment variable."
        )
        return False
    return True


@lru_cache(maxsize=1)
def get_pr_number() -> Optional[int]:
    pr_number = os.environ.get("CODEFLASH_PR_NUMBER")
    if not pr_number:
        return None
    return int(pr_number)


def ensure_pr_number() -> bool:
    if not get_pr_number():
        msg = (
            "CODEFLASH_PR_NUMBER not found in environment variables; make sure the Github Action is setting this so "
            "Codeflash can comment on the right PR"
        )
        raise OSError(msg)
    return True


@lru_cache(maxsize=1)
def is_end_to_end() -> bool:
    return bool(os.environ.get("CODEFLASH_END_TO_END"))
