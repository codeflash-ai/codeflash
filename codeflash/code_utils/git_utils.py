from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from functools import cache
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git
from rich.prompt import Confirm
from unidiff import PatchSet

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_consts import N_CANDIDATES_EFFECTIVE

if TYPE_CHECKING:
    from git import Repo


def get_git_diff(
    repo_directory: Path | None = None, *, only_this_commit: Optional[str] = None, uncommitted_changes: bool = False
) -> dict[str, list[int]]:
    if repo_directory is None:
        repo_directory = Path.cwd()
    repository = git.Repo(repo_directory, search_parent_directories=True)
    commit = repository.head.commit
    if only_this_commit:
        uni_diff_text = repository.git.diff(
            only_this_commit + "^1", only_this_commit, ignore_blank_lines=True, ignore_space_at_eol=True
        )
    elif uncommitted_changes:
        uni_diff_text = repository.git.diff(None, "HEAD", ignore_blank_lines=True, ignore_space_at_eol=True)
    else:
        uni_diff_text = repository.git.diff(
            commit.hexsha + "^1", commit.hexsha, ignore_blank_lines=True, ignore_space_at_eol=True
        )
    patch_set = PatchSet(StringIO(uni_diff_text))
    change_list: dict[str, list[int]] = {}  # list of changes
    for patched_file in patch_set:
        file_path: Path = Path(patched_file.path)
        if file_path.suffix != ".py":
            continue
        file_path = Path(repository.working_dir) / file_path
        logger.debug(f"file name: {file_path}")

        add_line_no: list[int] = [
            line.target_line_no for hunk in patched_file for line in hunk if line.is_added and line.value.strip() != ""
        ]  # the row number of deleted lines

        logger.debug(f"added lines: {add_line_no}")

        del_line_no: list[int] = [
            line.source_line_no
            for hunk in patched_file
            for line in hunk
            if line.is_removed and line.value.strip() != ""
        ]  # the row number of added lines

        logger.debug(f"deleted lines: {del_line_no}")

        change_list[file_path] = add_line_no
    return change_list


def get_current_branch(repo: Repo | None = None) -> str:
    """Return the name of the current branch in the given repository.

    :param repo: An optional Repo object. If not provided, the function will
                 search for a repository in the current and parent directories.
    :return: The name of the current branch.
    """
    repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
    return repository.active_branch.name


def get_remote_url(repo: Repo | None = None, git_remote: str | None = "origin") -> str:
    repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
    available_remotes = get_git_remotes(repository)
    if not available_remotes:
        raise ValueError("No git remotes configured in this repository")
    if git_remote not in available_remotes:
        msg = f"Git remote '{git_remote}' does not exist. Available remotes: {', '.join(available_remotes)}"
        raise ValueError(msg)
    return repository.remote(name=git_remote).url


def get_git_remotes(repo: Repo) -> list[str]:
    repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
    return [remote.name for remote in repository.remotes]


@cache
def get_repo_owner_and_name(repo: Repo | None = None, git_remote: str | None = "origin") -> tuple[str, str]:
    remote_url = get_remote_url(repo, git_remote)  # call only once
    remote_url = remote_url.removesuffix(".git") if remote_url.endswith(".git") else remote_url
    # remote_url = get_remote_url(repo, git_remote).removesuffix(".git") if remote_url.endswith(".git") else remote_url
    remote_url = remote_url.rstrip("/")
    split_url = remote_url.split("/")
    repo_owner_with_github, repo_name = split_url[-2], split_url[-1]
    repo_owner = repo_owner_with_github.split(":")[1] if ":" in repo_owner_with_github else repo_owner_with_github
    return repo_owner, repo_name


def git_root_dir(repo: Repo | None = None) -> Path:
    repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
    return Path(repository.working_dir)


def check_running_in_git_repo(module_root: str) -> bool:
    try:
        _ = git.Repo(module_root, search_parent_directories=True).git_dir
    except git.InvalidGitRepositoryError:
        return False
    else:
        return True


def confirm_proceeding_with_no_git_repo() -> str | bool:
    if sys.__stdin__.isatty():
        return Confirm.ask(
            "WARNING: I did not find a git repository for your code. If you proceed with running codeflash, "
            "optimized code will be written over your current code and you could irreversibly lose your current code. Proceed?",
            default=False,
        )
    # continue running on non-interactive environments, important for GitHub actions
    return True


def check_and_push_branch(repo: git.Repo, git_remote: str | None = "origin", *, wait_for_push: bool = False) -> bool:
    current_branch = repo.active_branch
    current_branch_name = current_branch.name
    available_remotes = get_git_remotes(repo)
    if not available_remotes:
        logger.error(
            f"❌ No git remotes configured in this repository.\n"
            f"This appears to be a local-only git repository. To use codeflash with PR features, you need to:\n"
            f"  1. Create a repository on GitHub (or another git hosting service)\n"
            f"  2. Add it as a remote: git remote add origin <repository-url>\n"
            f"  3. Push your branch: git push -u origin {current_branch_name}\n\n"
            f"Alternatively, you can run codeflash with the '--no-pr' flag to optimize locally without creating PRs."
        )
        return False

    # Check if the specified remote exists
    if git_remote not in available_remotes:
        logger.error(
            f"❌ Git remote '{git_remote}' does not exist in this repository.\n"
            f"Available remotes: {', '.join(available_remotes)}\n\n"
            f"You can either:\n"
            f"  1. Use one of the existing remotes by setting 'git-remote' in pyproject.toml\n"
            f"  2. Add the '{git_remote}' remote: git remote add {git_remote} <repository-url>\n"
            f"  3. Run codeflash with '--no-pr' to optimize locally without creating PRs."
        )
        return False

    remote = repo.remote(name=git_remote)

    # Check if the branch is pushed
    if f"{git_remote}/{current_branch_name}" not in repo.refs:
        logger.warning(f"⚠️ The branch '{current_branch_name}' is not pushed to the remote repository.")
        if not sys.__stdin__.isatty():
            logger.warning("Non-interactive shell detected. Branch will not be pushed.")
            return False
        if sys.__stdin__.isatty() and Confirm.ask(
            f"⚡️ In order for me to create PRs, your current branch needs to be pushed. Do you want to push "
            f"the branch '{current_branch_name}' to the remote repository?",
            default=False,
        ):
            remote.push(current_branch)
            logger.info(f"⬆️ Branch '{current_branch_name}' has been pushed to {git_remote}.")
            if wait_for_push:
                time.sleep(3)  # adding this to give time for the push to register with GitHub,
                # so that our modifications to it are not rejected
            return True
        logger.info(f"🔘 Branch '{current_branch_name}' has not been pushed to {git_remote}.")
        return False
    logger.debug(f"The branch '{current_branch_name}' is present in the remote repository.")
    return True


def create_worktree_root_dir(module_root: Path) -> tuple[Path | None, Path | None]:
    git_root = git_root_dir() if check_running_in_git_repo(module_root) else None
    worktree_root_dir = Path(tempfile.mkdtemp()) if git_root else None
    return git_root, worktree_root_dir


def create_git_worktrees(
    git_root: Path | None, worktree_root_dir: Path | None, module_root: Path
) -> tuple[Path | None, list[Path]]:
    if git_root and worktree_root_dir:
        worktree_root = Path(tempfile.mkdtemp(dir=worktree_root_dir))
        worktrees = [Path(tempfile.mkdtemp(dir=worktree_root)) for _ in range(N_CANDIDATES_EFFECTIVE + 1)]
        for worktree in worktrees:
            subprocess.run(["git", "worktree", "add", "-d", worktree], cwd=module_root, check=True)
    else:
        worktree_root = None
        worktrees = []
    return worktree_root, worktrees


def remove_git_worktrees(worktree_root: Path | None, worktrees: list[Path]) -> None:
    try:
        for worktree in worktrees:
            subprocess.run(["git", "worktree", "remove", "-f", worktree], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error removing worktrees: {e}")
    if worktree_root:
        shutil.rmtree(worktree_root)


def get_last_commit_author_if_pr_exists(repo: Repo | None = None) -> str | None:
    """Return the author's name of the last commit in the current branch if PR_NUMBER is set.

    Otherwise, return None.
    """
    if "PR_NUMBER" not in os.environ:
        return None
    try:
        repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
        last_commit = repository.head.commit
    except Exception:
        logger.exception("Failed to get last commit author.")
        return None
    else:
        return last_commit.author.name
