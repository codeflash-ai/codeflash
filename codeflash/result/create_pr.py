from __future__ import annotations

import os
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
from codeflash.code_utils.tabulate import tabulate
from codeflash.code_utils.time_utils import format_perf, format_time
from codeflash.github.PrComment import FileDiffContent, PrComment
from codeflash.models.models import FunctionCalledInTest, InvocationId
from codeflash.result.critic import performance_gain
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from codeflash.models.models import FunctionCalledInTest, InvocationId
    from codeflash.result.explanation import Explanation
    from codeflash.verification.verification_utils import TestConfig


def existing_tests_source_for(
    function_qualified_name_with_modules_from_root: str,
    function_to_tests: dict[str, set[FunctionCalledInTest]],
    test_cfg: TestConfig,
    original_runtimes_all: dict[InvocationId, list[int]],
    optimized_runtimes_all: dict[InvocationId, list[int]],
) -> str:
    test_files = function_to_tests.get(function_qualified_name_with_modules_from_root)
    if not test_files:
        return ""
    output = ""
    rows = []
    headers = ["Test File::Test Function", "Original ⏱️", "Optimized ⏱️", "Speedup"]
    tests_root = test_cfg.tests_root

    # Build non_generated_tests set (minimal extra work)
    non_generated_tests = {t.tests_in_file.test_file for t in test_files}

    # Use defaultdict paddings (saves many dict lookups and conditionals)
    from collections import defaultdict

    original_tests_to_runtimes = defaultdict(dict)
    optimized_tests_to_runtimes = defaultdict(dict)

    # Union all invocation ids from both dicts, filter early by test file
    all_invocation_ids = set(original_runtimes_all) | set(optimized_runtimes_all)
    path_cache = {}
    for invocation_id in all_invocation_ids:
        # Path construction optimized: use a cache
        tid = invocation_id.test_module_path
        if tid in path_cache:
            abs_path = path_cache[tid]
        else:
            abs_path = Path(tid.replace(".", os.sep)).with_suffix(".py").resolve()  # expensive
            path_cache[tid] = abs_path
        if abs_path not in non_generated_tests:
            continue
        # Update per-path, per-name runtime dicts
        qualified_name = (
            f"{invocation_id.test_class_name}.{invocation_id.test_function_name}"
            if invocation_id.test_class_name
            else invocation_id.test_function_name
        )
        # Initialize to 0 only if not present; avoid redundant assignment
        orig = original_tests_to_runtimes[abs_path]
        opt = optimized_tests_to_runtimes[abs_path]
        if qualified_name not in orig:
            orig[qualified_name] = 0
        if qualified_name not in opt:
            opt[qualified_name] = 0
        if invocation_id in original_runtimes_all:
            orig[qualified_name] += min(original_runtimes_all[invocation_id])
        if invocation_id in optimized_runtimes_all:
            opt[qualified_name] += min(optimized_runtimes_all[invocation_id])

    # Collect output rows in one pass
    all_abs_paths = list(original_tests_to_runtimes.keys())
    for filename in sorted(all_abs_paths):
        orig_runtimes = original_tests_to_runtimes[filename]
        opt_runtimes = optimized_tests_to_runtimes[filename]
        qualified_names = sorted(orig_runtimes)
        # Only process func names with nonzero original and optimized
        for qn in qualified_names:
            o_time = orig_runtimes[qn]
            opt_time = opt_runtimes[qn]
            if o_time != 0 and opt_time != 0:
                print_optimized_runtime = format_time(opt_time)
                print_original_runtime = format_time(o_time)
                print_filename = filename.relative_to(tests_root)
                # Branch for emoji
                perf_gain_val = performance_gain(original_runtime_ns=o_time, optimized_runtime_ns=opt_time) * 100
                perf_gain_str = format_perf(perf_gain_val)
                if opt_time > o_time:
                    emoji = "⚠️"
                else:
                    emoji = "✅"
                rows.append(
                    [
                        f"`{print_filename.as_posix()}::{qn}`",
                        print_original_runtime,
                        print_optimized_runtime,
                        f"{emoji}{perf_gain_str}%",
                    ]
                )

    output += tabulate(
        headers=headers, tabular_data=rows, tablefmt="pipe", colglobalalign=None, preserve_whitespace=True
    )
    output += "\n"
    return output


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
