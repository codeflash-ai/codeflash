import os
from io import StringIO
from typing import Optional

import git
from git import Repo
from unidiff import PatchSet


def get_git_diff(
    repo_directory: str = os.getcwd(), uncommitted_changes: bool = False
) -> dict[str, list[int]]:
    repository = git.Repo(repo_directory)
    commit = repository.head.commit
    if uncommitted_changes:
        uni_diff_text = repository.git.diff(
            None, "HEAD", ignore_blank_lines=True, ignore_space_at_eol=True
        )
    else:
        uni_diff_text = repository.git.diff(
            commit.hexsha + "^1",
            commit.hexsha,
            ignore_blank_lines=True,
            ignore_space_at_eol=True,
        )
    patch_set = PatchSet(StringIO(uni_diff_text))
    change_list: dict[str, list[int]] = {}  # list of changes
    for patched_file in patch_set:
        file_path: str = patched_file.path  # file name
        if not file_path.endswith(".py"):
            continue
        print("file name :" + file_path)
        add_line_no: list[int] = [
            line.target_line_no
            for hunk in patched_file
            for line in hunk
            if line.is_added and line.value.strip() != ""
        ]  # the row number of deleted lines

        print("added lines : " + str(add_line_no))
        del_line_no: list[int] = [
            line.source_line_no
            for hunk in patched_file
            for line in hunk
            if line.is_removed and line.value.strip() != ""
        ]  # the row number of added liens

        print("deleted lines : " + str(del_line_no))

        change_list[file_path] = add_line_no
    return change_list


def get_remote_url(repo: Optional[Repo] = None) -> str:
    repository: Repo = repo if repo else git.Repo(search_parent_directories=True)
    return repository.remote().url


def get_repo_owner_and_name(repo: Optional[Repo] = None) -> tuple[str, str]:
    remote_url = get_remote_url(repo)
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    if "://" in remote_url:
        # It's an HTTP/HTTPS URL
        repo_owner, repo_name = remote_url.split("/")[-2:]
    else:
        # It's an SSH URL and should contain ':' after the domain
        repo_owner_with_github, repo_name = remote_url.split("/")[-2:]
        repo_owner = (
            repo_owner_with_github.split(":")[1]
            if ":" in repo_owner_with_github
            else repo_owner_with_github
        )
    return repo_owner, repo_name


def get_github_secrets_page_url(repo: Optional[Repo] = None) -> str:
    owner, repo_name = get_repo_owner_and_name(repo)
    return f"https://github.com/{owner}/{repo_name}/settings/secrets/actions"
