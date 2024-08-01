from __future__ import annotations

import logging
import os.path
import pathlib
from typing import Dict, Optional

import git

from codeflash.api import cfapi
from codeflash.code_utils import env_utils
from codeflash.code_utils.git_utils import (
    check_and_push_branch,
    get_current_branch,
    get_repo_owner_and_name,
    git_root_dir,
)
from codeflash.discovery.discover_unit_tests import TestsInFile
from codeflash.github.PrComment import FileDiffContent, PrComment
from codeflash.result.explanation import Explanation


def existing_tests_source_for(
    function_qualified_name_with_modules_from_root: str,
    function_to_tests: Dict[str, list[TestsInFile]],
    tests_root: str,
) -> str:
    test_files = function_to_tests.get(function_qualified_name_with_modules_from_root)
    existing_tests_unique = set()
    if test_files:
        for test_file in test_files:
            existing_tests_unique.add("- " + os.path.relpath(test_file.test_file, tests_root))
    return "\n".join(sorted(existing_tests_unique))


def check_create_pr(
    original_code: dict[str, str],
    new_code: dict[str, str],
    explanation: Explanation,
    existing_tests_source: str,
    generated_original_test_source: str,
    function_trace_id: str,
) -> None:
    pr_number: Optional[int] = env_utils.get_pr_number()
    git_repo = git.Repo(search_parent_directories=True)

    if pr_number is not None:
        logging.info(f"Suggesting changes to PR #{pr_number} ...")
        owner, repo = get_repo_owner_and_name(git_repo)
        relative_path = str(pathlib.Path(os.path.relpath(explanation.file_path, git_root_dir())).as_posix())
        response = cfapi.suggest_changes(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            file_changes={
                str(pathlib.Path(os.path.relpath(p, git_root_dir())).as_posix()): FileDiffContent(
                    oldContent=original_code[p],
                    newContent=new_code[p],
                )
                for p in original_code
            },
            pr_comment=PrComment(
                optimization_explanation=explanation.explanation_message(),
                best_runtime=explanation.best_runtime_ns,
                original_runtime=explanation.original_runtime_ns,
                function_name=explanation.function_name,
                relative_file_path=relative_path,
                speedup_x=explanation.speedup_x,
                speedup_pct=explanation.speedup_pct,
                winning_test_results=explanation.winning_test_results,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
        )
        if response.ok:
            logging.info(f"Suggestions were successfully made to PR #{pr_number}")
        else:
            logging.error(
                f"Optimization was successful, but I failed to suggest changes to PR #{pr_number}."
                f" Response from server was: {response.text}",
            )
    else:
        logging.info("Creating a new PR with the optimized code...")
        owner, repo = get_repo_owner_and_name(git_repo)

        if not check_and_push_branch(git_repo, wait_for_push=True):
            logging.warning("⏭️ Branch is not pushed, skipping PR creation...")
            return
        relative_path = str(pathlib.Path(os.path.relpath(explanation.file_path, git_root_dir())).as_posix())
        base_branch = get_current_branch()
        response = cfapi.create_pr(
            owner=owner,
            repo=repo,
            base_branch=base_branch,
            file_changes={
                str(pathlib.Path(os.path.relpath(p, git_root_dir())).as_posix()): FileDiffContent(
                    oldContent=original_code[p],
                    newContent=new_code[p],
                )
                for p in original_code
            },
            pr_comment=PrComment(
                optimization_explanation=explanation.explanation_message(),
                best_runtime=explanation.best_runtime_ns,
                original_runtime=explanation.original_runtime_ns,
                function_name=explanation.function_name,
                relative_file_path=relative_path,
                speedup_x=explanation.speedup_x,
                speedup_pct=explanation.speedup_pct,
                winning_test_results=explanation.winning_test_results,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
        )
        if response.ok:
            logging.info(f"Successfully created a new PR #{response.text} with the optimized code.")
        else:
            logging.error(
                f"Optimization was successful, but I failed to create a PR with the optimized code."
                f" Response from server was: {response.text}",
            )
