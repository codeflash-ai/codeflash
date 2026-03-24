from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import git

from codeflash_python.api import cfapi
from codeflash_python.code_utils import env_utils
from codeflash_python.code_utils.git_utils import check_and_push_branch, get_current_branch, get_repo_owner_and_name
from codeflash_python.code_utils.tabulate import tabulate
from codeflash_python.code_utils.time_utils import format_perf, format_time
from codeflash_python.result.critic import performance_gain
from codeflash_python.result.github_utils import github_pr_url
from codeflash_python.result.pr_comment import FileDiffContent, PrComment
from codeflash_python.static_analysis.code_replacer import is_zero_diff

if TYPE_CHECKING:
    from codeflash_core.config import TestConfig
    from codeflash_python.models.models import FunctionCalledInTest, InvocationId, TestFiles
    from codeflash_python.result.explanation import Explanation


logger = logging.getLogger("codeflash_python")


def existing_tests_source_for(
    function_qualified_name_with_modules_from_root: str,
    function_to_tests: dict[str, set[FunctionCalledInTest]],
    test_cfg: TestConfig,
    original_runtimes_all: dict[InvocationId, list[int]],
    optimized_runtimes_all: dict[InvocationId, list[int]],
    test_files_registry: TestFiles | None = None,
) -> tuple[str, str, str]:
    logger.debug(
        "[PR-DEBUG] existing_tests_source_for called with func=%s", function_qualified_name_with_modules_from_root
    )
    logger.debug("[PR-DEBUG] function_to_tests keys: %s", list(function_to_tests.keys()))
    logger.debug("[PR-DEBUG] original_runtimes_all has %s entries", len(original_runtimes_all))
    logger.debug("[PR-DEBUG] optimized_runtimes_all has %s entries", len(optimized_runtimes_all))
    test_files = function_to_tests.get(function_qualified_name_with_modules_from_root)
    if not test_files:
        logger.debug("[PR-DEBUG] No test_files found for %s", function_qualified_name_with_modules_from_root)
        return "", "", ""
    logger.debug("[PR-DEBUG] Found %s test_files", len(test_files))
    for tf in test_files:
        logger.debug("[PR-DEBUG]   test_file: %s, test_type=%s", tf.tests_in_file.test_file, tf.tests_in_file.test_type)
    output_existing: str = ""
    output_concolic: str = ""
    output_replay: str = ""
    rows_existing = []
    rows_concolic = []
    rows_replay = []
    headers = ["Test File::Test Function", "Original \u23f1\ufe0f", "Optimized \u23f1\ufe0f", "Speedup"]
    tests_root = test_cfg.tests_root
    original_tests_to_runtimes: dict[Path, dict[str, int]] = {}
    optimized_tests_to_runtimes: dict[Path, dict[str, int]] = {}

    # Build lookup from instrumented path -> original path using the test_files_registry
    # Include both behavior and benchmarking paths since test results might come from either
    instrumented_to_original: dict[Path, Path] = {}
    if test_files_registry:
        for registry_tf in test_files_registry.test_files:
            if registry_tf.original_file_path:
                if registry_tf.instrumented_behavior_file_path:
                    instrumented_to_original[registry_tf.instrumented_behavior_file_path.resolve()] = (
                        registry_tf.original_file_path.resolve()
                    )
                    logger.debug(
                        "[PR-DEBUG] Mapping (behavior): %s -> %s",
                        registry_tf.instrumented_behavior_file_path.name,
                        registry_tf.original_file_path.name,
                    )
                if registry_tf.benchmarking_file_path:
                    instrumented_to_original[registry_tf.benchmarking_file_path.resolve()] = (
                        registry_tf.original_file_path.resolve()
                    )
                    logger.debug(
                        "[PR-DEBUG] Mapping (perf): %s -> %s",
                        registry_tf.benchmarking_file_path.name,
                        registry_tf.original_file_path.name,
                    )

    # Resolve all paths to absolute for consistent comparison
    non_generated_tests: set[Path] = set()
    for test_file in test_files:
        resolved = test_file.tests_in_file.test_file.resolve()
        non_generated_tests.add(resolved)
        logger.debug("[PR-DEBUG] Added to non_generated_tests: %s", resolved)
    all_invocation_ids = original_runtimes_all.keys() | optimized_runtimes_all.keys()
    logger.debug("[PR-DEBUG] Processing %s invocation_ids", len(all_invocation_ids))
    matched_count = 0
    skipped_count = 0
    for invocation_id in all_invocation_ids:
        test_module_path = invocation_id.test_module_path
        abs_path = Path(test_module_path.replace(".", os.sep)).with_suffix(".py").resolve()
        if abs_path not in non_generated_tests:
            skipped_count += 1
            if skipped_count <= 5:
                logger.debug("[PR-DEBUG] SKIP: abs_path=%s", abs_path.name)
                logger.debug("[PR-DEBUG]   Expected one of: %s", [p.name for p in list(non_generated_tests)[:3]])
            continue
        matched_count += 1
        logger.debug("[PR-DEBUG] MATCHED: %s", abs_path.name)
        if abs_path not in original_tests_to_runtimes:
            original_tests_to_runtimes[abs_path] = {}
        if abs_path not in optimized_tests_to_runtimes:
            optimized_tests_to_runtimes[abs_path] = {}
        qualified_name = (
            invocation_id.test_class_name + "." + invocation_id.test_function_name  # type: ignore[operator]
            if invocation_id.test_class_name
            else invocation_id.test_function_name
        )
        if qualified_name not in original_tests_to_runtimes[abs_path]:
            original_tests_to_runtimes[abs_path][qualified_name] = 0  # type: ignore[index]
        if qualified_name not in optimized_tests_to_runtimes[abs_path]:
            optimized_tests_to_runtimes[abs_path][qualified_name] = 0  # type: ignore[index]
        if invocation_id in original_runtimes_all:
            original_tests_to_runtimes[abs_path][qualified_name] += min(original_runtimes_all[invocation_id])  # type: ignore[index]
        if invocation_id in optimized_runtimes_all:
            optimized_tests_to_runtimes[abs_path][qualified_name] += min(optimized_runtimes_all[invocation_id])  # type: ignore[index]
    logger.debug("[PR-DEBUG] SUMMARY: matched=%s, skipped=%s", matched_count, skipped_count)
    logger.debug("[PR-DEBUG] original_tests_to_runtimes has %s files", len(original_tests_to_runtimes))
    # parse into string
    all_abs_paths = (
        original_tests_to_runtimes.keys()
    )  # both will have the same keys as some default values are assigned in the previous loop
    for filename in sorted(all_abs_paths):
        all_qualified_names = original_tests_to_runtimes[
            filename
        ].keys()  # both will have the same keys as some default values are assigned in the previous loop
        for qualified_name in sorted(all_qualified_names):
            # if not present in optimized output nan
            if (
                original_tests_to_runtimes[filename][qualified_name] != 0
                and optimized_tests_to_runtimes[filename][qualified_name] != 0
            ):
                print_optimized_runtime = format_time(optimized_tests_to_runtimes[filename][qualified_name])
                print_original_runtime = format_time(original_tests_to_runtimes[filename][qualified_name])
                print_filename = filename.resolve().relative_to(tests_root.resolve()).as_posix()
                greater = (
                    optimized_tests_to_runtimes[filename][qualified_name]
                    > original_tests_to_runtimes[filename][qualified_name]
                )
                perf_gain = format_perf(
                    performance_gain(
                        original_runtime_ns=original_tests_to_runtimes[filename][qualified_name],
                        optimized_runtime_ns=optimized_tests_to_runtimes[filename][qualified_name],
                    )
                    * 100
                )
                if greater:
                    if "__replay_test_" in str(print_filename):
                        rows_replay.append(
                            [
                                f"`{print_filename}::{qualified_name}`",
                                f"{print_original_runtime}",
                                f"{print_optimized_runtime}",
                                f"{perf_gain}%\u26a0\ufe0f",
                            ]
                        )
                    elif "codeflash_concolic" in str(print_filename):
                        rows_concolic.append(
                            [
                                f"`{print_filename}::{qualified_name}`",
                                f"{print_original_runtime}",
                                f"{print_optimized_runtime}",
                                f"{perf_gain}%\u26a0\ufe0f",
                            ]
                        )
                    else:
                        rows_existing.append(
                            [
                                f"`{print_filename}::{qualified_name}`",
                                f"{print_original_runtime}",
                                f"{print_optimized_runtime}",
                                f"{perf_gain}%\u26a0\ufe0f",
                            ]
                        )
                elif "__replay_test_" in str(print_filename):
                    rows_replay.append(
                        [
                            f"`{print_filename}::{qualified_name}`",
                            f"{print_original_runtime}",
                            f"{print_optimized_runtime}",
                            f"{perf_gain}%\u2705",
                        ]
                    )
                elif "codeflash_concolic" in str(print_filename):
                    rows_concolic.append(
                        [
                            f"`{print_filename}::{qualified_name}`",
                            f"{print_original_runtime}",
                            f"{print_optimized_runtime}",
                            f"{perf_gain}%\u2705",
                        ]
                    )
                else:
                    rows_existing.append(
                        [
                            f"`{print_filename}::{qualified_name}`",
                            f"{print_original_runtime}",
                            f"{print_optimized_runtime}",
                            f"{perf_gain}%\u2705",
                        ]
                    )
    output_existing += tabulate(
        headers=headers, tabular_data=rows_existing, tablefmt="pipe", colglobalalign=None, preserve_whitespace=True
    )
    output_existing += "\n"
    if len(rows_existing) == 0:
        output_existing = ""
    output_concolic += tabulate(
        headers=headers, tabular_data=rows_concolic, tablefmt="pipe", colglobalalign=None, preserve_whitespace=True
    )
    output_concolic += "\n"
    if len(rows_concolic) == 0:
        output_concolic = ""
    output_replay += tabulate(
        headers=headers, tabular_data=rows_replay, tablefmt="pipe", colglobalalign=None, preserve_whitespace=True
    )
    output_replay += "\n"
    if len(rows_replay) == 0:
        output_replay = ""
    return output_existing, output_replay, output_concolic


def check_create_pr(
    original_code: dict[Path, str],
    new_code: dict[Path, str],
    explanation: Explanation,
    existing_tests_source: str,
    generated_original_test_source: str,
    function_trace_id: str,
    coverage_message: str,
    replay_tests: str,
    root_dir: Path,
    concolic_tests: str = "",
    git_remote: str | None = None,
    optimization_review: str = "",
    original_line_profiler: str | None = None,
    optimized_line_profiler: str | None = None,
) -> None:
    pr_number: int | None = env_utils.get_pr_number()
    git_repo = git.Repo(search_parent_directories=True)

    if pr_number is not None:
        logger.info("Suggesting changes to PR #%s ...", pr_number)
        owner, repo = get_repo_owner_and_name(git_repo)
        relative_path = explanation.file_path.resolve().relative_to(root_dir.resolve()).as_posix()
        build_file_changes = {
            Path(p).resolve().relative_to(root_dir.resolve()).as_posix(): FileDiffContent(
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
                winning_behavior_test_results=explanation.winning_behavior_test_results,
                winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
                benchmark_details=explanation.benchmark_details,
                original_async_throughput=explanation.original_async_throughput,
                best_async_throughput=explanation.best_async_throughput,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
            coverage_message=coverage_message,
            replay_tests=replay_tests,
            concolic_tests=concolic_tests,
            optimization_review=optimization_review,
            original_line_profiler=original_line_profiler,
            optimized_line_profiler=optimized_line_profiler,
        )
        if response.ok:
            logger.info("Suggestions were successfully made to PR #%s", pr_number)
        else:
            logger.error(
                "Optimization was successful, but I failed to suggest changes to PR #%s. Response from server was: %s",
                pr_number,
                response.text,
            )
    else:
        logger.info("Creating a new PR with the optimized code...")

        owner, repo = get_repo_owner_and_name(git_repo, git_remote)
        logger.info("Pushing to %s - Owner: %s, Repo: %s", git_remote, owner, repo)

        if not check_and_push_branch(git_repo, git_remote, wait_for_push=True):
            logger.warning("\u23ed\ufe0f Branch is not pushed, skipping PR creation...")
            return
        relative_path = explanation.file_path.resolve().relative_to(root_dir.resolve()).as_posix()
        base_branch = get_current_branch()
        build_file_changes = {
            Path(p).resolve().relative_to(root_dir.resolve()).as_posix(): FileDiffContent(
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
                winning_behavior_test_results=explanation.winning_behavior_test_results,
                winning_benchmarking_test_results=explanation.winning_benchmarking_test_results,
                benchmark_details=explanation.benchmark_details,
                original_async_throughput=explanation.original_async_throughput,
                best_async_throughput=explanation.best_async_throughput,
            ),
            existing_tests=existing_tests_source,
            generated_tests=generated_original_test_source,
            trace_id=function_trace_id,
            coverage_message=coverage_message,
            replay_tests=replay_tests,
            concolic_tests=concolic_tests,
            optimization_review=optimization_review,
            original_line_profiler=original_line_profiler,
            optimized_line_profiler=optimized_line_profiler,
        )
        if response.ok:
            pr_id = response.text
            pr_url = github_pr_url(owner, repo, pr_id)
            logger.info("Successfully created a new PR #%s with the optimized code: %s", pr_id, pr_url)
        else:
            logger.error(
                "Optimization was successful, but I failed to create a PR with the optimized code."
                " Response from server was: %s",
                response.text,
            )
