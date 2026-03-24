"""Function filtering and validation for optimization discovery."""

from __future__ import annotations

import ast
import contextlib
import logging
import os
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git

from codeflash_python.api.cfapi import get_blocklisted_functions, is_function_being_optimized_again
from codeflash_python.code_utils.code_utils import module_name_from_file_path, path_belongs_to_site_packages
from codeflash_python.code_utils.env_utils import get_pr_number
from codeflash_python.code_utils.git_utils import get_repo_owner_and_name
from codeflash_python.models.function_types import qualified_name_with_modules_from_root

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.models.models import CodeOptimizationContext
    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


def is_git_repo(file_path: str) -> bool:
    try:
        git.Repo(file_path, search_parent_directories=True)
        return True
    except git.InvalidGitRepositoryError:
        return False


@cache
def ignored_submodule_paths(module_root: str) -> list[Path]:
    if is_git_repo(module_root):
        git_repo = git.Repo(module_root, search_parent_directories=True)
        try:
            working_dir = git_repo.working_tree_dir
            if working_dir is not None:
                return [Path(working_dir, submodule.path).resolve() for submodule in git_repo.submodules]
        except Exception as e:
            logger.warning("Error getting submodule paths: %s", e)
    return []


def was_function_previously_optimized(
    function_to_optimize: FunctionToOptimize, code_context: CodeOptimizationContext, args: Namespace
) -> bool:
    """Check which functions have already been optimized and filter them out.

    This function calls the optimization API to:
    1. Check which functions are already optimized
    2. Log new function hashes to the database
    3. Return only functions that need optimization

    Returns:
        Tuple of (filtered_functions_dict, remaining_count)

    """
    # was_function_previously_optimized is for the checking the optimization duplicates in the github action, no need to do this in the LSP mode

    # Check optimization status if repository info is provided
    # already_optimized_count = 0

    # Check optimization status if repository info is provided
    # already_optimized_count = 0
    owner = None
    repo = None
    with contextlib.suppress(git.exc.InvalidGitRepositoryError):
        owner, repo = get_repo_owner_and_name()

    pr_number = get_pr_number()

    if not owner or not repo or pr_number is None or getattr(args, "no_pr", False):
        return False

    func_hash = code_context.hashing_code_context_hash

    code_contexts = [
        {
            "file_path": str(function_to_optimize.file_path),
            "function_name": function_to_optimize.qualified_name,
            "code_hash": func_hash,
        }
    ]

    try:
        result = is_function_being_optimized_again(owner, repo, pr_number, code_contexts)
        already_optimized_paths: list[tuple[str, str]] = result.get("already_optimized_tuples", [])
        return len(already_optimized_paths) > 0

    except Exception as e:
        logger.warning("Failed to check optimization status: %s", e)
        # Return all functions if API call fails
        return False


def filter_functions(
    modified_functions: dict[Path, list[FunctionToOptimize]],
    tests_root: Path,
    ignore_paths: list[Path],
    project_root: Path,
    module_root: Path,
    previous_checkpoint_functions: dict[str, dict[str, Any]] | None = None,
    *,
    disable_logs: bool = False,
) -> tuple[dict[Path, list[FunctionToOptimize]], int]:
    resolved_project_root = project_root.resolve()
    filtered_modified_functions: dict[Path, list[FunctionToOptimize]] = {}
    blocklist_funcs = get_blocklisted_functions()
    logger.debug("Blocklisted functions: %s", blocklist_funcs)
    # Remove any function that we don't want to optimize
    # already_optimized_paths = check_optimization_status(modified_functions, project_root)

    # Ignore files with submodule path, cache the submodule paths
    submodule_paths = ignored_submodule_paths(module_root)

    functions_count: int = 0
    test_functions_removed_count: int = 0
    non_modules_removed_count: int = 0
    site_packages_removed_count: int = 0
    ignore_paths_removed_count: int = 0
    malformed_paths_count: int = 0
    submodule_ignored_paths_count: int = 0
    blocklist_funcs_removed_count: int = 0
    previous_checkpoint_functions_removed_count: int = 0
    # Normalize paths for case-insensitive comparison on Windows
    tests_root_str = os.path.normcase(str(tests_root))
    module_root_str = os.path.normcase(str(module_root))
    project_root_str = os.path.normcase(str(project_root))

    # Check if tests_root overlaps with module_root or project_root
    # In this case, we need to use file pattern matching instead of directory matching
    tests_root_overlaps_source = tests_root_str in (module_root_str, project_root_str) or module_root_str.startswith(
        tests_root_str + os.sep
    )

    # Test file patterns for when tests_root overlaps with source
    test_file_name_patterns = (".test.", ".spec.", "_test.", "_spec.")
    test_dir_patterns = (os.sep + "test" + os.sep, os.sep + "tests" + os.sep, os.sep + "__tests__" + os.sep)

    def is_test_file(file_path_normalized: str) -> bool:
        if tests_root_overlaps_source:
            file_lower = file_path_normalized.lower()
            basename = Path(file_lower).name
            if basename.startswith("test_") or basename == "conftest.py":
                return True
            if any(pattern in file_lower for pattern in test_file_name_patterns):
                return True
            if project_root_str and file_lower.startswith(project_root_str.lower()):
                relative_path = file_lower[len(project_root_str) :]
                return any(pattern in relative_path for pattern in test_dir_patterns)
            return False
        return file_path_normalized.startswith(tests_root_str + os.sep)

    # We desperately need Python 3.10+ only support to make this code readable with structural pattern matching
    for file_path_path, functions in modified_functions.items():
        _functions = functions
        file_path = str(file_path_path)
        file_path_normalized = os.path.normcase(file_path)
        if is_test_file(file_path_normalized):
            test_functions_removed_count += len(_functions)
            continue
        if file_path_path in ignore_paths or any(
            file_path_normalized.startswith(os.path.normcase(str(ignore_path)) + os.sep) for ignore_path in ignore_paths
        ):
            ignore_paths_removed_count += 1
            continue
        if file_path_path in submodule_paths or any(
            file_path_normalized.startswith(os.path.normcase(str(submodule_path)) + os.sep)
            for submodule_path in submodule_paths
        ):
            submodule_ignored_paths_count += 1
            continue
        if path_belongs_to_site_packages(Path(file_path)):
            site_packages_removed_count += len(_functions)
            continue
        if not file_path_normalized.startswith(module_root_str + os.sep):
            non_modules_removed_count += len(_functions)
            continue

        try:
            ast.parse(f"import {module_name_from_file_path(Path(file_path), resolved_project_root)}")
        except SyntaxError:
            malformed_paths_count += 1
            continue

        if blocklist_funcs:
            functions_tmp = []
            for function in _functions:
                if (
                    function.file_path.name in blocklist_funcs
                    and function.qualified_name in blocklist_funcs[function.file_path.name]
                ):
                    # This function is in blocklist, we can skip it
                    blocklist_funcs_removed_count += 1
                    continue
                # This function is NOT in blocklist. we can keep it
                functions_tmp.append(function)
            _functions = functions_tmp

        if previous_checkpoint_functions:
            functions_tmp = []
            for function in _functions:
                if (
                    qualified_name_with_modules_from_root(function, resolved_project_root)
                    in previous_checkpoint_functions
                ):
                    previous_checkpoint_functions_removed_count += 1
                    continue
                functions_tmp.append(function)
            _functions = functions_tmp

        filtered_modified_functions[file_path_path] = _functions
        functions_count += len(_functions)

    if not disable_logs:
        log_info = {
            "Test functions removed": test_functions_removed_count,
            "Site-package functions removed": site_packages_removed_count,
            "Non-importable file paths": malformed_paths_count,
            "Functions outside module-root": non_modules_removed_count,
            "Files from ignored paths": ignore_paths_removed_count,
            "Files from ignored submodules": submodule_ignored_paths_count,
            "Blocklisted functions removed": blocklist_funcs_removed_count,
            "Functions skipped from checkpoint": previous_checkpoint_functions_removed_count,
        }
        entries = [f"{label}: {cnt}" for label, cnt in log_info.items() if cnt > 0]
        if entries:
            logger.info("Ignored functions and files: %s", ", ".join(entries))
    return {k: v for k, v in filtered_modified_functions.items() if v}, functions_count


def is_test_file_by_pattern(file_path: Path) -> bool:
    """Check if a file is a test file using naming conventions.

    Used when tests_root overlaps with module_root, so directory-based filtering would
    incorrectly exclude all source files. Falls back to filename and directory patterns.
    """
    name = file_path.name.lower()
    if name.startswith("test_") or name == "conftest.py":
        return True
    test_name_patterns = (".test.", ".spec.", "_test.", "_spec.")
    if any(p in name for p in test_name_patterns):
        return True
    path_str = str(file_path).lower()
    test_dir_patterns = (os.sep + "test" + os.sep, os.sep + "tests" + os.sep, os.sep + "__tests__" + os.sep)
    return any(p in path_str for p in test_dir_patterns)


def filter_files_optimized(file_path: Path, tests_root: Path, ignore_paths: list[Path], module_root: Path) -> bool:
    """Optimized version of the filter_functions function above.

    Takes in file paths and returns the count of files that are to be optimized.
    """
    submodule_paths = None
    # When tests_root overlaps module_root (e.g., both are "src"), use pattern matching
    # instead of directory matching to avoid filtering out all source files.
    tests_root_overlaps = tests_root == module_root or module_root.is_relative_to(tests_root)
    if tests_root_overlaps:
        if is_test_file_by_pattern(file_path):
            return False
    elif file_path.is_relative_to(tests_root):
        return False
    if file_path in ignore_paths or any(file_path.is_relative_to(ignore_path) for ignore_path in ignore_paths):
        return False
    if path_belongs_to_site_packages(file_path):
        return False
    if not file_path.is_relative_to(module_root):
        return False
    if submodule_paths is None:
        submodule_paths = ignored_submodule_paths(module_root)
    return not (
        file_path in submodule_paths
        or any(file_path.is_relative_to(submodule_path) for submodule_path in submodule_paths)
    )
