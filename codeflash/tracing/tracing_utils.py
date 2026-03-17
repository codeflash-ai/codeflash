from __future__ import annotations

import os
import site
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Optional, cast

import git


# This can't be pydantic dataclass because then conflicts with the logfire pytest plugin
# for pydantic tracing. We want to not use pydantic in the tracing code.
@dataclass
class FunctionModules:
    function_name: str
    file_name: Path
    module_name: str
    class_name: Optional[str] = None
    line_no: Optional[int] = None


def path_belongs_to_site_packages(file_path: Path) -> bool:
    site_packages = [Path(p) for p in site.getsitepackages()]
    return any(file_path.resolve().is_relative_to(site_package_path) for site_package_path in site_packages)


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
        working_tree_dir = cast("Path", git_repo.working_tree_dir)
        try:
            return [Path(working_tree_dir, submodule.path).resolve() for submodule in git_repo.submodules]
        except Exception as e:
            print(f"Failed to get submodule paths {e!s}")  # no logger since used in the tracer
    return []


def module_name_from_file_path(file_path: Path, project_root_path: Path) -> str:
    relative_path = file_path.relative_to(project_root_path)
    return relative_path.with_suffix("").as_posix().replace("/", ".")


def _is_test_file_by_pattern(file_path: Path) -> bool:
    """Check if a file is a test file using naming conventions.

    Used when tests_root overlaps with module_root, so directory-based filtering would
    incorrectly exclude all source files.
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
    tests_root_overlaps = tests_root == module_root or module_root.is_relative_to(tests_root)
    if tests_root_overlaps:
        if _is_test_file_by_pattern(file_path):
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
