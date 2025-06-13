from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git

from codeflash.api import cfapi
from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_replacer import is_zero_diff
from codeflash.code_utils.git_utils import (
    check_and_push_branch,
    get_current_branch,
    get_repo_owner_and_name,
    git_root_dir,
)
from codeflash.code_utils.github_utils import github_pr_url
from codeflash.github.PrComment import FileDiffContent, PrComment

if TYPE_CHECKING:
    from codeflash.models.models import FunctionCalledInTest, TestResults
    from codeflash.result.explanation import Explanation


def existing_tests_source_for(
    function_qualified_name_with_modules_from_root: str,
    function_to_tests: dict[str, set[FunctionCalledInTest]],
    tests_root: Path,
    original_test_results: Optional[TestResults] = None,
    optimized_test_results: Optional[TestResults] = None,
) -> str:
    test_files = function_to_tests.get(function_qualified_name_with_modules_from_root)
    if not test_files:
        return ""
    existing_tests_unique = set()
    # a lot of loops, need to do in a single loop
    #original_runtime_by_test = original_test_results.usable_runtime_data_by_test_case()
    #optimized_runtime_by_test = optimized_test_results.usable_runtime_data_by_test_case()
    # Group test cases by test file
    test_files_grouped = {}
    for test_file in test_files:
        file_path = Path(test_file.tests_in_file.test_file)
        relative_path = str(file_path.relative_to(tests_root))

        if relative_path not in test_files_grouped:
            test_files_grouped[relative_path] = []
        test_files_grouped.setdefault(relative_path,[]).append(test_file)

    # Create detailed report for each test file
    # for relative_path, tests_in_file in sorted(test_files_grouped.items()):
        file_line = f"- {relative_path}"

        # Add test case details with timing information if available
        #if original_test_results and optimized_test_results:
        test_case_details = []
        # Collect test function names for this file
        test_functions_in_file = {test_file.tests_in_file.test_function for test_file in tests_in_file}

        # Create timing report for each test function
        for test_function_name in sorted(test_functions_in_file):
            # Find matching runtime data
            original_runtimes = []
            optimized_runtimes = []

            for invocation_id, runtimes in original_runtime_by_test.items():
                if invocation_id.test_function_name == test_function_name:
                    original_runtimes.extend(runtimes)

            for invocation_id, runtimes in optimized_runtime_by_test.items():
                if invocation_id.test_function_name == test_function_name:
                    optimized_runtimes.extend(runtimes)

            if original_runtimes and optimized_runtimes:
                # Use minimum timing like the generated tests function does
                original_time = min(original_runtimes)
                optimized_time = min(optimized_runtimes)

                from codeflash.code_utils.time_utils import format_time

                original_str = format_time(original_time)
                optimized_str = format_time(optimized_time)

                test_case_details.append(f"    - {test_function_name}: {original_str} -> {optimized_str}")

        if test_case_details:
            file_line += "\n" + "\n".join(test_case_details)

        existing_tests_unique.add(file_line)

    return "\n".join(sorted(existing_tests_unique))


def check_create_pr(
    original_code: dict[Path, str],
    new_code: dict[Path, str],
    explanation: Explanation,
    existing_tests_source: str,
    generated_original_test_source: str,
    function_trace_id: str,
    coverage_message: str,
    git_remote: Optional[str] = None,
) -> None:
    pr_number: Optional[int] = env_utils.get_pr_number()
    git_repo = git.Repo(search_parent_directories=True)

    if pr_number is not None:
        logger.info(f"Suggesting changes to PR #{pr_number} ...")
        owner, repo = get_repo_owner_and_name(git_repo)
        relative_path = explanation.file_path.relative_to(git_root_dir()).as_posix()
        build_file_changes = {
            Path(p).relative_to(git_root_dir()).as_posix(): FileDiffContent(
                oldContent=original_code[p], newContent=new_code[p]
            )
            for p in original_code
            if not is_zero_diff(original_code[p], new_code[p])
        }
        if not build_file_changes:
            logger.info("No changes to suggest to PR.")
            return
        response = cfapi.suggest_changes(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            file_changes=build_file_changes,
            pr_comment=PrComment(
                optimization_explanation=explanation.explanation_message(),
                best_runtime=explanation.best_runtime_ns,
                original_runtime=explanation.original_runtime_ns,
                function_name=explanation.function_name,
                relative_file_path=relative_path,
                speedup_x=explanation.speedup_x,
                speedup_pct=explanation.speedup_pct,
                winning_behavioral_test_results=explanation.winning_behavioral_test_results,
                winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
                benchmark_details=explanation.benchmark_details,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
            coverage_message=coverage_message,
        )
        if response.ok:
            logger.info(f"Suggestions were successfully made to PR #{pr_number}")
        else:
            logger.error(
                f"Optimization was successful, but I failed to suggest changes to PR #{pr_number}."
                f" Response from server was: {response.text}"
            )
    else:
        logger.info("Creating a new PR with the optimized code...")
        console.rule()
        owner, repo = get_repo_owner_and_name(git_repo, git_remote)
        logger.info(f"Pushing to {git_remote} - Owner: {owner}, Repo: {repo}")
        console.rule()
        if not check_and_push_branch(git_repo, wait_for_push=True):
            logger.warning("⏭️ Branch is not pushed, skipping PR creation...")
            return
        relative_path = explanation.file_path.relative_to(git_root_dir()).as_posix()
        base_branch = get_current_branch()
        build_file_changes = {
            Path(p).relative_to(git_root_dir()).as_posix(): FileDiffContent(
                oldContent=original_code[p], newContent=new_code[p]
            )
            for p in original_code
        }

        response = cfapi.create_pr(
            owner=owner,
            repo=repo,
            base_branch=base_branch,
            file_changes=build_file_changes,
            pr_comment=PrComment(
                optimization_explanation=explanation.explanation_message(),
                best_runtime=explanation.best_runtime_ns,
                original_runtime=explanation.original_runtime_ns,
                function_name=explanation.function_name,
                relative_file_path=relative_path,
                speedup_x=explanation.speedup_x,
                speedup_pct=explanation.speedup_pct,
                winning_behavioral_test_results=explanation.winning_behavioral_test_results,
                winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
                benchmark_details=explanation.benchmark_details,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
            coverage_message=coverage_message,
        )
        if response.ok:
            pr_id = response.text
            pr_url = github_pr_url(owner, repo, pr_id)
            logger.info(f"Successfully created a new PR #{pr_id} with the optimized code: {pr_url}")
        else:
            logger.error(
                f"Optimization was successful, but I failed to create a PR with the optimized code."
                f" Response from server was: {response.text}"
            )
        console.rule()
