from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path

    from git import Repo


def apologize_and_exit() -> None:
    console.rule()
    logger.info(
        "💡 If you're having trouble, see https://docs.codeflash.ai/getting-started/local-installation for further help getting started with Codeflash!"
    )
    console.rule()
    logger.info("👋 Exiting...")
    sys.exit(1)


def get_git_repo_or_none(search_path: Optional[Path] = None) -> Optional[Repo]:
    """Get git repository or None if not in a git repo."""
    import git

    try:
        if search_path:
            return git.Repo(search_path, search_parent_directories=True)
        return git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None


def require_git_repo_or_exit(search_path: Optional[Path] = None, error_message: Optional[str] = None) -> Repo:
    """Get git repository or exit with error."""
    repo = get_git_repo_or_none(search_path)
    if repo is None:
        if error_message:
            logger.error(error_message)
        else:
            logger.error(
                "I couldn't find a git repository in the current directory. "
                "A git repository is required for this operation."
            )
        apologize_and_exit()
    # After checking for None and calling apologize_and_exit(), we know repo is not None
    # but mypy doesn't understand apologize_and_exit() never returns, so we assert
    assert repo is not None
    return repo


def parse_config_file_or_exit(config_file: Optional[Path] = None, **kwargs: Any) -> tuple[dict[str, Any], Path]:
    """Parse config file or exit with error."""
    from codeflash.code_utils.code_utils import exit_with_message
    from codeflash.code_utils.config_parser import parse_config_file

    try:
        return parse_config_file(config_file, **kwargs)
    except ValueError as e:
        exit_with_message(f"Error parsing config file: {e}", error_on_exit=True)
        # exit_with_message never returns when error_on_exit=True, but mypy doesn't know that
        raise  # pragma: no cover
