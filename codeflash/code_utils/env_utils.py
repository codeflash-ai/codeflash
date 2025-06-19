from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import exit_with_message
from codeflash.code_utils.formatter import format_code
from codeflash.code_utils.shell_utils import read_api_key_from_shell_config


def check_formatter_installed(formatter_cmds: list[str], exit_on_failure: bool = True) -> bool:  # noqa
    return_code = True
    if formatter_cmds[0] == "disabled":
        return return_code
    tmp_code = """print("hello world")"""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py") as f:
        f.write(tmp_code)
        f.flush()
        tmp_file = Path(f.name)
        try:
            format_code(formatter_cmds, tmp_file, print_status=False, exit_on_failure=exit_on_failure)
        except Exception:
            exit_with_message(
                "⚠️ Codeflash requires a code formatter to be installed in your environment, but none was found. Please install a supported formatter, verify the formatter-cmds in your codeflash pyproject.toml config and try again.",
                error_on_exit=True,
            )
    return return_code


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
