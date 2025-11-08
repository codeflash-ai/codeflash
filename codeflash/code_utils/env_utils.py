from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import exit_with_message
from codeflash.code_utils.formatter import format_code
from codeflash.code_utils.shell_utils import read_api_key_from_shell_config, save_api_key_to_rc
from codeflash.lsp.helpers import is_LSP_enabled


def check_formatter_installed(formatter_cmds: list[str], exit_on_failure: bool = True) -> bool:  # noqa
    if not formatter_cmds or formatter_cmds[0] == "disabled":
        return True

    first_cmd = formatter_cmds[0]
    cmd_tokens = shlex.split(first_cmd) if isinstance(first_cmd, str) else [first_cmd]

    if not cmd_tokens:
        return True

    exe_name = cmd_tokens[0]
    command_str = " ".join(formatter_cmds).replace(" $file", "")

    if shutil.which(exe_name) is None:
        logger.error(
            f"Could not find formatter: {command_str}\n"
            f"Please install it or update 'formatter-cmds' in your codeflash configuration"
        )
        return False

    tmp_code = """print("hello world")"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "test_codeflash_formatter.py"
            tmp_file.write_text(tmp_code, encoding="utf-8")
            format_code(formatter_cmds, tmp_file, print_status=False, exit_on_failure=False)
            return True
    except FileNotFoundError:
        logger.error(
            f"Could not find formatter: {command_str}\n"
            f"Please install it or update 'formatter-cmds' in your codeflash configuration"
        )
        return False
    except Exception as e:
        logger.error(f"Formatter failed to run: {command_str}\nError: {e}")
        return False


@lru_cache(maxsize=1)
def get_codeflash_api_key() -> str:
    # Check environment variable first
    env_api_key = os.environ.get("CODEFLASH_API_KEY")
    shell_api_key = read_api_key_from_shell_config()

    # If we have an env var but it's not in shell config, save it for persistence
    if env_api_key and not shell_api_key:
        try:
            from codeflash.either import is_successful

            result = save_api_key_to_rc(env_api_key)
            if is_successful(result):
                logger.debug(f"Automatically saved API key from environment to shell config: {result.unwrap()}")
        except Exception as e:
            logger.debug(f"Failed to automatically save API key to shell config: {e}")

    # Prefer the shell configuration over environment variables for lsp,
    # as the API key may change in the RC file during lsp runtime. Since the LSP client (extension) can restart
    # within the same process, the environment variable could become outdated.
    api_key = shell_api_key or env_api_key if is_LSP_enabled() else env_api_key or shell_api_key

    api_secret_docs_message = "For more information, refer to the documentation at [https://docs.codeflash.ai/getting-started/codeflash-github-actions#add-your-api-key-to-your-repository-secrets]."  # noqa
    if not api_key:
        msg = (
            "I didn't find a Codeflash API key in your environment.\nYou can generate one at "
            "https://app.codeflash.ai/app/apikeys ,\nthen set it as a CODEFLASH_API_KEY environment variable.\n"
            f"{api_secret_docs_message}"
        )
        if is_repo_a_fork():
            msg = (
                "Codeflash API key not detected in your environment. It appears you're running Codeflash from a GitHub fork.\n"
                "For external contributors, please ensure you've added your own API key to your fork's repository secrets and set it as the CODEFLASH_API_KEY environment variable.\n"
                f"{api_secret_docs_message}"
            )
            exit_with_message(msg)
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
    event_data = get_cached_gh_event_data()
    pr_number = event_data.get("number")
    if pr_number:
        return int(pr_number)

    pr_number = os.environ.get("CODEFLASH_PR_NUMBER")
    if pr_number:
        return int(pr_number)
    return None


def ensure_pr_number() -> bool:
    if not get_pr_number():
        msg = (
            "Codeflash couldn't detect your pull request number. Are you running Codeflash within a GitHub Action?"
            "If not, please set the CODEFLASH_PR_NUMBER environment variable to ensure Codeflash can comment on the correct PR."
        )
        raise OSError(msg)
    return True


@lru_cache(maxsize=1)
def is_end_to_end() -> bool:
    return bool(os.environ.get("CODEFLASH_END_TO_END"))


@lru_cache(maxsize=1)
def get_cached_gh_event_data() -> dict[str, Any]:
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    with Path(event_path).open(encoding="utf-8") as f:
        return json.load(f)  # type: ignore  # noqa


def is_repo_a_fork() -> bool:
    event = get_cached_gh_event_data()
    return bool(event.get("pull_request", {}).get("head", {}).get("repo", {}).get("fork", False))


@lru_cache(maxsize=1)
def is_ci() -> bool:
    """Check if running in a CI environment."""
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def is_pr_draft() -> bool:
    """Check if the PR is draft. in the github action context."""
    event = get_cached_gh_event_data()
    return bool(event.get("pull_request", {}).get("draft", False))
