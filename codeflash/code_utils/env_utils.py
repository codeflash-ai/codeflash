import logging
import os
from typing import Optional


def get_codeflash_api_key() -> Optional[str]:
    api_key = os.environ.get("CODEFLASH_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "CodeFlash API key not found in your environment."
            + "You can generate one at https://app.codeflash.ai/app/apikeys,"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
    return api_key


def ensure_codeflash_api_key() -> bool:
    try:
        get_codeflash_api_key()
    except EnvironmentError as e:
        logging.error(
            "CodeFlash API key not found in your environment."
            + "You can generate one at https://app.codeflash.ai/app/apikeys,"
            + "then set it as a CODEFLASH_API_KEY environment variable."
        )
        return False
    return True


def get_codeflash_org_key() -> Optional[str]:
    api_key = os.environ.get("CODEFLASH_ORG_KEY")
    return api_key


def get_pr_number() -> Optional[int]:
    pr_number = os.environ.get("PR_NUMBER")
    if not pr_number:
        return None
    else:
        return int(pr_number)


def ensure_pr_number() -> bool:
    if not get_pr_number():
        raise EnvironmentError(
            f"PR_NUMBER not found in environment variables; make sure the Github Action is setting this so CodeFlash can comment on the right PR"
        )
    return True
