from __future__ import annotations

import ast
import logging
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_python.code_utils.code_utils import exit_with_message
from codeflash_python.code_utils.config_consts import PYTHON_DIR_EXCLUDES, PYTHON_FILE_EXTENSIONS
from codeflash_python.code_utils.git_utils import get_git_diff
from codeflash_python.discovery.discover_unit_tests import discover_unit_tests
from codeflash_python.discovery.filter_criteria import FunctionFilterCriteria
from codeflash_python.discovery.function_filtering import filter_functions
from codeflash_python.discovery.function_visitors import discover_functions, is_class_defined_in_file
from codeflash_python.telemetry.posthog_cf import ph

if TYPE_CHECKING:
    from codeflash_core.config import TestConfig
    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


# =============================================================================
# Multi-language support helpers
# =============================================================================

_VCS_EXCLUDES = frozenset({".git", ".hg", ".svn"})


def parse_dir_excludes(patterns: frozenset[str]) -> tuple[frozenset[str], tuple[str, ...], tuple[str, ...]]:
    """Split glob patterns into exact names, prefixes, and suffixes.

    Patterns ending with ``*`` become prefix matches, patterns starting with ``*``
    become suffix matches, and plain strings become exact matches.
    """
    exact: set[str] = set()
    prefixes: list[str] = []
    suffixes: list[str] = []
    for p in patterns:
        if p.endswith("*"):
            prefixes.append(p[:-1])
        elif p.startswith("*"):
            suffixes.append(p[1:])
        else:
            exact.add(p)
    return frozenset(exact), tuple(prefixes), tuple(suffixes)


def get_files_for_language(
    module_root_path: Path, ignore_paths: list[Path] | None = None, language: str | None = None
) -> list[Path]:
    """Get all source files for supported languages.

    Args:
        module_root_path: Root path to search for source files.
        ignore_paths: List of paths to ignore (can be files or directories).
        language: Optional specific language to filter for. If None, includes all supported languages.

    Returns:
        List of file paths matching supported extensions.

    """
    if ignore_paths is None:
        ignore_paths = []

    extensions = PYTHON_FILE_EXTENSIONS
    all_patterns = PYTHON_DIR_EXCLUDES | _VCS_EXCLUDES

    dir_excludes, prefixes, suffixes = parse_dir_excludes(all_patterns)

    ignore_dirs: set[str] = set()
    ignore_files: set[Path] = set()
    for p in ignore_paths:
        p = Path(p) if not isinstance(p, Path) else p
        if p.is_file():
            ignore_files.add(p)
        else:
            ignore_dirs.add(str(p))

    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(module_root_path):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in dir_excludes
            and not (prefixes and d.startswith(prefixes))
            and not (suffixes and d.endswith(suffixes))
            and str(Path(dirpath) / d) not in ignore_dirs
        ]
        for fname in filenames:
            if fname.endswith(extensions):
                fpath = Path(dirpath, fname)
                if fpath not in ignore_files:
                    files.append(fpath)
    return files


def find_all_functions_via_language_support(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find all optimizable functions using the language support abstraction.

    This function uses the registered language support for the file's language
    to discover functions, then converts them to FunctionToOptimize instances.
    """
    functions: dict[Path, list[FunctionToOptimize]] = {}

    try:
        criteria = FunctionFilterCriteria(require_return=True)
        source = file_path.read_text(encoding="utf-8")
        functions[file_path] = discover_functions(source, file_path, criteria)
    except Exception as e:
        logger.debug("Failed to discover functions in %s: %s", file_path, e)

    return functions


def get_functions_to_optimize(
    optimize_all: str | None,
    replay_test: list[Path] | None,
    file: Path | str | None,
    only_get_this_function: str | None,
    test_cfg: TestConfig,
    ignore_paths: list[Path],
    project_root: Path,
    module_root: Path,
    previous_checkpoint_functions: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[Path, list[FunctionToOptimize]], int, Path | None]:
    assert sum([bool(optimize_all), bool(replay_test), bool(file)]) <= 1, (
        "Only one of optimize_all, replay_test, or file should be provided"
    )
    functions: dict[Path, list[FunctionToOptimize]]
    trace_file_path: Path | None = None
    is_lsp = False
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=SyntaxWarning)
        if optimize_all:
            functions = get_all_files_and_functions(Path(optimize_all), ignore_paths)
        elif replay_test:
            functions, trace_file_path = get_all_replay_test_functions(
                replay_test=replay_test, test_cfg=test_cfg, project_root_path=project_root
            )
        elif file is not None:
            file = Path(file) if isinstance(file, str) else file
            functions = find_all_functions_in_file(file)
            if only_get_this_function is not None:
                split_function = only_get_this_function.split(".")
                if len(split_function) > 2:
                    if is_lsp:
                        return functions, 0, None
                    exit_with_message(
                        "Function name should be in the format 'function_name' or 'class_name.function_name'"
                    )
                if len(split_function) == 2:
                    class_name, only_function_name = split_function
                else:
                    class_name = None
                    only_function_name = split_function[0]
                found_function = None
                for fn in functions.get(file, []):
                    if only_function_name == fn.function_name and (
                        class_name is None or class_name == fn.top_level_parent_name
                    ):
                        found_function = fn
                if found_function is None:
                    if is_lsp:
                        return functions, 0, None

                    found = closest_matching_file_function_name(only_get_this_function, functions)
                    if found is not None:
                        file, found_function = found
                        exit_with_message(
                            f"Function {only_get_this_function} not found in file {file}\nor the function does not have a 'return' statement or is a property.\n"
                            f"Did you mean {found_function.qualified_name} instead?"
                        )

                    exit_with_message(
                        f"Function {only_get_this_function} not found in file {file}\nor the function does not have a 'return' statement or is a property"
                    )

                assert found_function is not None
                functions[file] = [found_function]
        else:
            logger.info("Finding all functions modified in the current git diff ...")
            ph("cli-optimizing-git-diff")
            functions = get_functions_within_git_diff(uncommitted_changes=False)
        filtered_modified_functions, functions_count = filter_functions(
            functions, test_cfg.tests_root, ignore_paths, project_root, module_root, previous_checkpoint_functions
        )

        return filtered_modified_functions, functions_count, trace_file_path


def get_functions_within_git_diff(uncommitted_changes: bool) -> dict[Path, list[FunctionToOptimize]]:
    modified_lines: dict[str, list[int]] = get_git_diff(uncommitted_changes=uncommitted_changes)
    return get_functions_within_lines(modified_lines)


def closest_matching_file_function_name(
    qualified_fn_to_find: str, found_fns: dict[Path, list[FunctionToOptimize]]
) -> tuple[Path, FunctionToOptimize] | None:
    """Find the closest matching function name using Levenshtein distance.

    Args:
        qualified_fn_to_find: Function name to find in format "Class.function" or "function"
        found_fns: Dictionary of file paths to list of functions

    Returns:
        Tuple of (file_path, function) for closest match, or None if no matches found

    """
    min_distance = 4
    closest_match = None
    closest_file = None

    qualified_fn_to_find_lower = qualified_fn_to_find.lower()

    # Cache levenshtein_distance locally for improved lookup speed
    _levenshtein = levenshtein_distance

    for file_path, functions in found_fns.items():
        for function in functions:
            # Compare either full qualified name or just function name
            fn_name = function.qualified_name.lower()
            # If the absolute length difference is already >= min_distance, skip calculation
            if abs(len(qualified_fn_to_find_lower) - len(fn_name)) >= min_distance:
                continue
            dist = _levenshtein(qualified_fn_to_find_lower, fn_name)

            if dist < min_distance:
                min_distance = dist
                closest_match = function
                closest_file = file_path

    if closest_match is not None and closest_file is not None:
        return closest_file, closest_match
    return None


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    len1 = len(s1)
    len2 = len(s2)
    # Use a preallocated list instead of creating a new list every iteration
    previous = list(range(len1 + 1))
    current = [0] * (len1 + 1)

    for index2 in range(len2):
        char2 = s2[index2]
        current[0] = index2 + 1
        for index1 in range(len1):
            char1 = s1[index1]
            if char1 == char2:
                current[index1 + 1] = previous[index1]
            else:
                # Fast min calculation without tuple construct
                a = previous[index1]
                b = previous[index1 + 1]
                c = current[index1]
                min_val = min(b, a)
                min_val = min(c, min_val)
                current[index1 + 1] = 1 + min_val
        # Swap references instead of copying
        previous, current = current, previous
    return previous[len1]


def get_functions_inside_a_commit(commit_hash: str) -> dict[Path, list[FunctionToOptimize]]:
    modified_lines: dict[str, list[int]] = get_git_diff(only_this_commit=commit_hash)
    return get_functions_within_lines(modified_lines)


def get_functions_within_lines(modified_lines: dict[str, list[int]]) -> dict[Path, list[FunctionToOptimize]]:
    functions: dict[Path, list[FunctionToOptimize]] = {}
    for path_str, lines_in_file in modified_lines.items():
        path = Path(path_str)
        if not path.exists():
            continue
        all_functions = find_all_functions_in_file(path)
        functions[path] = [
            func
            for func in all_functions.get(path, [])
            if func.starting_line is not None
            and func.ending_line is not None
            and any(func.starting_line <= line <= func.ending_line for line in lines_in_file)
        ]
    return functions


def get_all_files_and_functions(
    module_root_path: Path, ignore_paths: list[Path], language: str | None = None
) -> dict[Path, list[FunctionToOptimize]]:
    """Get all optimizable functions from files in the module root.

    Args:
        module_root_path: Root path to search for source files.
        ignore_paths: List of paths to ignore.
        language: Optional specific language to filter for. If None, includes all supported languages.

    Returns:
        Dictionary mapping file paths to lists of FunctionToOptimize.

    """
    functions: dict[Path, list[FunctionToOptimize]] = {}
    for file_path in get_files_for_language(module_root_path, ignore_paths, language):
        functions.update(find_all_functions_in_file(file_path).items())
    # Randomize the order of the files to optimize to avoid optimizing the same file in the same order every time.
    # Helpful if an optimize-all run is stuck and we restart it.
    files_list = list(functions.items())
    random.shuffle(files_list)
    return dict(files_list)


def find_all_functions_in_file(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find all optimizable functions in a file using the language support abstraction."""
    if file_path.suffix.lower() not in PYTHON_FILE_EXTENSIONS:
        return {}
    try:
        criteria = FunctionFilterCriteria(require_return=True)
        source = file_path.read_text(encoding="utf-8")
        return {file_path: discover_functions(source, file_path, criteria)}
    except Exception as e:
        logger.debug("Failed to discover functions in %s: %s", file_path, e)
        return {}


def get_all_replay_test_functions(
    replay_test: list[Path], test_cfg: TestConfig, project_root_path: Path
) -> tuple[dict[Path, list[FunctionToOptimize]], Path]:
    trace_file_path: Path | None = None
    for replay_test_file in replay_test:
        try:
            with replay_test_file.open("r", encoding="utf8") as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "trace_file_path"
                                and isinstance(node.value, ast.Constant)
                                and isinstance(node.value.value, str)
                            ):
                                trace_file_path = Path(node.value.value)
                                break
                        if trace_file_path:
                            break
            if trace_file_path:
                break
        except Exception as e:
            logger.warning("Error parsing replay test file %s: %s", replay_test_file, e)

    if trace_file_path is None:
        logger.error("Could not find trace_file_path in replay test files.")
        exit_with_message("Could not find trace_file_path in replay test files.")
        raise AssertionError("Unreachable")  # exit_with_message never returns

    if not trace_file_path.exists():
        logger.error("Trace file not found: %s", trace_file_path)
        exit_with_message(
            f"Trace file not found: {trace_file_path}\n"
            "The trace file referenced in the replay test no longer exists.\n"
            "This can happen if the trace file was cleaned up after a previous optimization run.\n"
            "Please regenerate the replay test by re-running 'codeflash optimize' with your command."
        )

    function_tests, _, _ = discover_unit_tests(test_cfg, discover_only_these_tests=replay_test)
    # Get the absolute file paths for each function, excluding class name if present
    filtered_valid_functions = defaultdict(list)
    file_to_functions_map = defaultdict(list)
    # below logic can be cleaned up with a better data structure to store the function paths
    for function in function_tests:
        parts = function.split(".")
        module_path_parts = parts[:-1]  # Exclude the function or method name
        function_name = parts[-1]
        # Check if the second-to-last part is a class name
        class_name = (
            module_path_parts[-1]
            if module_path_parts
            and is_class_defined_in_file(
                module_path_parts[-1], Path(project_root_path, *module_path_parts[:-1]).with_suffix(".py")
            )
            else None
        )
        if class_name:
            # If there is a class name, append it to the module path
            qualified_function_name = class_name + "." + function_name
            file_path_parts = module_path_parts[:-1]  # Exclude the class name
        else:
            qualified_function_name = function_name
            file_path_parts = module_path_parts
        file_path = Path(project_root_path, *file_path_parts).with_suffix(".py")
        if not file_path.exists():
            continue
        file_to_functions_map[file_path].append((qualified_function_name, function_name, class_name))
    for file_path, functions_in_file in file_to_functions_map.items():
        all_valid_functions: dict[Path, list[FunctionToOptimize]] = find_all_functions_in_file(file_path=file_path)
        filtered_list = []
        for func_data in functions_in_file:
            qualified_name_to_match, _, _ = func_data
            filtered_list.extend(
                [
                    valid_function
                    for valid_function in all_valid_functions[file_path]
                    if valid_function.qualified_name == qualified_name_to_match
                ]
            )
        if filtered_list:
            filtered_valid_functions[file_path] = filtered_list

    return dict(filtered_valid_functions), trace_file_path
