from __future__ import annotations

import ast
import os
import random
import warnings
from _ast import AsyncFunctionDef, ClassDef, FunctionDef
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import git
import libcst as cst
from pydantic.dataclasses import dataclass
from rich.tree import Tree

from codeflash.api.cfapi import get_blocklisted_functions, is_function_being_optimized_again
from codeflash.cli_cmds.console import DEBUG_MODE, console, logger
from codeflash.code_utils.code_utils import (
    exit_with_message,
    is_class_defined_in_file,
    module_name_from_file_path,
    path_belongs_to_site_packages,
)
from codeflash.code_utils.env_utils import get_pr_number
from codeflash.code_utils.git_utils import get_git_diff, get_repo_owner_and_name
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.languages.language_enum import Language
from codeflash.languages.registry import get_language_support, get_supported_extensions, is_language_supported
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.models.function_types import FunctionParent, FunctionToOptimize
from codeflash.telemetry.posthog_cf import ph

# Re-export for backward compatibility
__all__ = ["FunctionParent", "FunctionToOptimize"]

if TYPE_CHECKING:
    from argparse import Namespace

    from libcst import CSTNode
    from libcst.metadata import CodeRange

    from codeflash.models.models import CodeOptimizationContext
    from codeflash.verification.verification_utils import TestConfig
import contextlib

from rich.text import Text

_property_id = "property"

_ast_name = ast.Name


@dataclass(frozen=True)
class FunctionProperties:
    is_top_level: bool
    has_args: Optional[bool]
    is_staticmethod: Optional[bool]
    is_classmethod: Optional[bool]
    staticmethod_class_name: Optional[str]


class ReturnStatementVisitor(cst.CSTVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_return_statement: bool = False

    def visit_Return(self, node: cst.Return) -> None:
        self.has_return_statement = True


class FunctionVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider, cst.metadata.ParentNodeProvider)

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path: str = file_path
        self.functions: list[FunctionToOptimize] = []

    @staticmethod
    def is_pytest_fixture(node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            dec = decorator.decorator
            if isinstance(dec, cst.Call):
                dec = dec.func
            if isinstance(dec, cst.Attribute) and dec.attr.value == "fixture":
                if isinstance(dec.value, cst.Name) and dec.value.value == "pytest":
                    return True
            if isinstance(dec, cst.Name) and dec.value == "fixture":
                return True
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        return_visitor: ReturnStatementVisitor = ReturnStatementVisitor()
        node.visit(return_visitor)
        if return_visitor.has_return_statement and not self.is_pytest_fixture(node):
            pos: CodeRange = self.get_metadata(cst.metadata.PositionProvider, node)
            parents: CSTNode | None = self.get_metadata(cst.metadata.ParentNodeProvider, node)
            ast_parents: list[FunctionParent] = []
            while parents is not None:
                if isinstance(parents, (cst.FunctionDef, cst.ClassDef)):
                    ast_parents.append(FunctionParent(parents.name.value, parents.__class__.__name__))
                parents = self.get_metadata(cst.metadata.ParentNodeProvider, parents, default=None)
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name.value,
                    file_path=self.file_path,
                    parents=list(reversed(ast_parents)),
                    starting_line=pos.start.line,
                    ending_line=pos.end.line,
                    is_async=bool(node.asynchronous),
                )
            )


class FunctionWithReturnStatement(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.functions: list[FunctionToOptimize] = []
        self.ast_path: list[FunctionParent] = []
        self.file_path: Path = file_path

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        if function_has_return_statement(node) and not function_is_a_property(node):
            self.functions.append(
                FunctionToOptimize(function_name=node.name, file_path=self.file_path, parents=self.ast_path[:])
            )

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        if function_has_return_statement(node) and not function_is_a_property(node):
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name, file_path=self.file_path, parents=self.ast_path[:], is_async=True
                )
            )

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.append(FunctionParent(node.name, node.__class__.__name__))
        super().generic_visit(node)
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.pop()


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
    module_root_path: Path, ignore_paths: list[Path] | None = None, language: Language | None = None
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

    all_patterns: frozenset[str]
    if language is not None:
        support = get_language_support(language)
        extensions = support.file_extensions
        all_patterns = support.dir_excludes | _VCS_EXCLUDES
    else:
        extensions = tuple(get_supported_extensions())
        all_patterns = _VCS_EXCLUDES
        for lang in Language:
            if is_language_supported(lang):
                all_patterns = all_patterns | get_language_support(lang).dir_excludes

    dir_excludes, prefixes, suffixes = parse_dir_excludes(all_patterns)

    ignore_dirs: set[str] = set()
    ignore_files: set[Path] = set()
    for p in ignore_paths:
        p = Path(p) if not isinstance(p, Path) else p
        if p.is_file():
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


def _is_js_ts_function_exported(file_path: Path, function_name: str) -> tuple[bool, str | None]:
    """Check if a JavaScript/TypeScript function is exported from its module.

    For JS/TS, functions that are not exported cannot be imported by tests,
    making them impossible to optimize.

    Args:
        file_path: Path to the source file.
        function_name: Name of the function to check.

    Returns:
        Tuple of (is_exported, export_name). export_name may be 'default' for default exports.

    """
    from codeflash.languages.javascript.treesitter import get_analyzer_for_file

    try:
        source = file_path.read_text(encoding="utf-8")
        analyzer = get_analyzer_for_file(file_path)
        return analyzer.is_function_exported(source, function_name)
    except Exception as e:
        logger.debug(f"Failed to check export status for {function_name}: {e}")
        # Return True to avoid blocking in case of errors
        return True, None


def _find_all_functions_in_python_file(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find all optimizable functions in a Python file using AST parsing.

    This is the original Python implementation preserved for backward compatibility.
    """
    functions: dict[Path, list[FunctionToOptimize]] = {}
    with file_path.open(encoding="utf8") as f:
        try:
            ast_module = ast.parse(f.read())
        except Exception as e:
            if DEBUG_MODE:
                logger.exception(e)
            return functions
        function_name_visitor = FunctionWithReturnStatement(file_path)
        function_name_visitor.visit(ast_module)
        functions[file_path] = function_name_visitor.functions
    return functions


def _find_all_functions_via_language_support(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find all optimizable functions using the language support abstraction.

    This function uses the registered language support for the file's language
    to discover functions, then converts them to FunctionToOptimize instances.
    """
    from codeflash.languages.base import FunctionFilterCriteria

    functions: dict[Path, list[FunctionToOptimize]] = {}

    try:
        lang_support = get_language_support(file_path)
        criteria = FunctionFilterCriteria(require_return=True)
        # discover_functions already returns FunctionToOptimize objects
        functions[file_path] = lang_support.discover_functions(file_path, criteria)
    except Exception as e:
        logger.debug(f"Failed to discover functions in {file_path}: {e}")

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
    functions: dict[str, list[FunctionToOptimize]]
    trace_file_path: Path | None = None
    is_lsp = is_LSP_enabled()
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=SyntaxWarning)
        if optimize_all:
            logger.info("!lsp|Finding all functions in the module '%s'…", optimize_all)
            console.rule()
            functions = get_all_files_and_functions(Path(optimize_all), ignore_paths)
        elif replay_test:
            functions, trace_file_path = get_all_replay_test_functions(
                replay_test=replay_test, test_cfg=test_cfg, project_root_path=project_root
            )
        elif file is not None:
            logger.info("!lsp|Finding all functions in the file '%s'…", file)
            console.rule()
            file = Path(file) if isinstance(file, str) else file
            functions: dict[Path, list[FunctionToOptimize]] = find_all_functions_in_file(file)
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

                # For JavaScript/TypeScript, verify that the function (or its parent class) is exported
                # Non-exported functions cannot be imported by tests
                if found_function.language in ("javascript", "typescript"):
                    # For class methods, check if the parent class is exported
                    # For standalone functions, check if the function itself is exported
                    if found_function.parents:
                        # It's a class method - check if the class is exported
                        name_to_check = found_function.top_level_parent_name
                    else:
                        # It's a standalone function - check if the function is exported
                        name_to_check = found_function.function_name

                    is_exported, _ = _is_js_ts_function_exported(file, name_to_check)
                    if not is_exported:
                        if found_function.parents:
                            logger.debug(
                                f"Class '{name_to_check}' containing method '{found_function.function_name}' "
                                f"is not exported from {file}. "
                                f"In JavaScript/TypeScript, only exported classes/functions can be optimized "
                                f"because tests need to import them."
                            )
                        else:
                            logger.debug(
                                f"Function '{found_function.function_name}' is not exported from {file}. "
                                f"In JavaScript/TypeScript, only exported functions can be optimized because "
                                f"tests need to import them."
                            )
                        return {}, 0, None

                functions[file] = [found_function]
        else:
            logger.info("Finding all functions modified in the current git diff ...")
            console.rule()
            ph("cli-optimizing-git-diff")
            functions = get_functions_within_git_diff(uncommitted_changes=False)
        filtered_modified_functions, functions_count = filter_functions(
            functions, test_cfg.tests_root, ignore_paths, project_root, module_root, previous_checkpoint_functions
        )

        logger.info(f"!lsp|Found {functions_count} function{'s' if functions_count > 1 else ''} to optimize")
        return filtered_modified_functions, functions_count, trace_file_path


def get_functions_within_git_diff(uncommitted_changes: bool) -> dict[str, list[FunctionToOptimize]]:
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

    if closest_match is not None:
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


def get_functions_inside_a_commit(commit_hash: str) -> dict[str, list[FunctionToOptimize]]:
    modified_lines: dict[str, list[int]] = get_git_diff(only_this_commit=commit_hash)
    return get_functions_within_lines(modified_lines)


def get_functions_within_lines(modified_lines: dict[str, list[int]]) -> dict[str, list[FunctionToOptimize]]:
    functions: dict[str, list[FunctionToOptimize]] = {}
    for path_str, lines_in_file in modified_lines.items():
        path = Path(path_str)
        if not path.exists():
            continue
        with path.open(encoding="utf8") as f:
            file_content = f.read()
            try:
                wrapper = cst.metadata.MetadataWrapper(cst.parse_module(file_content))
            except Exception as e:
                logger.exception(e)
                continue
            function_lines = FunctionVisitor(file_path=str(path))
            wrapper.visit(function_lines)
            functions[str(path)] = [
                function_to_optimize
                for function_to_optimize in function_lines.functions
                if (start_line := function_to_optimize.starting_line) is not None
                and (end_line := function_to_optimize.ending_line) is not None
                and any(start_line <= line <= end_line for line in lines_in_file)
            ]
    return functions


def get_all_files_and_functions(
    module_root_path: Path, ignore_paths: list[Path], language: Language | None = None
) -> dict[str, list[FunctionToOptimize]]:
    """Get all optimizable functions from files in the module root.

    Args:
        module_root_path: Root path to search for source files.
        ignore_paths: List of paths to ignore.
        language: Optional specific language to filter for. If None, includes all supported languages.

    Returns:
        Dictionary mapping file paths to lists of FunctionToOptimize.

    """
    functions: dict[str, list[FunctionToOptimize]] = {}
    for file_path in get_files_for_language(module_root_path, ignore_paths, language):
        # Find all the functions in the file
        functions.update(find_all_functions_in_file(file_path).items())
    # Randomize the order of the files to optimize to avoid optimizing the same file in the same order every time.
    # Helpful if an optimize-all run is stuck and we restart it.
    files_list = list(functions.items())
    random.shuffle(files_list)
    return dict(files_list)


def find_all_functions_in_file(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find all optimizable functions in a file, routing to the appropriate language handler.

    This function checks if the file extension is supported and routes to either
    the Python-specific implementation (for backward compatibility) or the
    language support abstraction for other languages.

    Args:
        file_path: Path to the source file.

    Returns:
        Dictionary mapping file path to list of FunctionToOptimize.

    """
    # Check if the file extension is supported
    if not is_language_supported(file_path):
        return {}

    try:
        lang_support = get_language_support(file_path)
    except Exception:
        return {}

    # Route to Python-specific implementation for backward compatibility
    if lang_support.language == Language.PYTHON:
        return _find_all_functions_in_python_file(file_path)

    # Use language support abstraction for other languages
    return _find_all_functions_via_language_support(file_path)


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
            logger.warning(f"Error parsing replay test file {replay_test_file}: {e}")

    if trace_file_path is None:
        logger.error("Could not find trace_file_path in replay test files.")
        exit_with_message("Could not find trace_file_path in replay test files.")
        raise AssertionError("Unreachable")  # exit_with_message never returns

    if not trace_file_path.exists():
        logger.error(f"Trace file not found: {trace_file_path}")
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
            logger.warning(f"Error getting submodule paths: {e}")
    return []


class TopLevelFunctionOrMethodVisitor(ast.NodeVisitor):
    def __init__(
        self, file_name: Path, function_or_method_name: str, class_name: str | None = None, line_no: int | None = None
    ) -> None:
        self.file_name = file_name
        self.class_name = class_name
        self.function_name = function_or_method_name
        self.is_top_level = False
        self.function_has_args: bool | None = None
        self.line_no = line_no
        self.is_staticmethod = False
        self.is_classmethod = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.class_name is None and node.name == self.function_name:
            self.is_top_level = True
            self.function_has_args = any(
                (
                    bool(node.args.args),
                    bool(node.args.kwonlyargs),
                    bool(node.args.kwarg),
                    bool(node.args.posonlyargs),
                    bool(node.args.vararg),
                )
            )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.class_name is None and node.name == self.function_name:
            self.is_top_level = True
            self.function_has_args = any(
                (
                    bool(node.args.args),
                    bool(node.args.kwonlyargs),
                    bool(node.args.kwarg),
                    bool(node.args.posonlyargs),
                    bool(node.args.vararg),
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # iterate over the class methods
        if node.name == self.class_name:
            for body_node in node.body:
                if (
                    isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and body_node.name == self.function_name
                ):
                    self.is_top_level = True
                    if any(
                        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                        for decorator in body_node.decorator_list
                    ):
                        self.is_classmethod = True
                    elif any(
                        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                        for decorator in body_node.decorator_list
                    ):
                        self.is_staticmethod = True
                    return
        elif self.line_no:
            # If we have line number info, check if class has a static method with the same line number
            # This way, if we don't have the class name, we can still find the static method
            for body_node in node.body:
                if (
                    isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and body_node.name == self.function_name
                    and body_node.lineno in {self.line_no, self.line_no + 1}
                    and any(
                        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                        for decorator in body_node.decorator_list
                    )
                ):
                    self.is_staticmethod = True
                    self.is_top_level = True
                    self.class_name = node.name
                    return

        return


def inspect_top_level_functions_or_methods(
    file_name: Path, function_or_method_name: str, class_name: str | None = None, line_no: int | None = None
) -> FunctionProperties | None:
    with file_name.open(encoding="utf8") as file:
        try:
            ast_module = ast.parse(file.read())
        except Exception:
            return None
    visitor = TopLevelFunctionOrMethodVisitor(
        file_name=file_name, function_or_method_name=function_or_method_name, class_name=class_name, line_no=line_no
    )
    visitor.visit(ast_module)
    staticmethod_class_name = visitor.class_name if visitor.is_staticmethod else None
    return FunctionProperties(
        is_top_level=visitor.is_top_level,
        has_args=visitor.function_has_args,
        is_staticmethod=visitor.is_staticmethod,
        is_classmethod=visitor.is_classmethod,
        staticmethod_class_name=staticmethod_class_name,
    )


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
    if is_LSP_enabled():
        # was_function_previously_optimized is for the checking the optimization duplicates in the github action, no need to do this in the LSP mode
        return False

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
        logger.warning(f"Failed to check optimization status: {e}")
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
    filtered_modified_functions: dict[str, list[FunctionToOptimize]] = {}
    blocklist_funcs = get_blocklisted_functions()
    logger.debug(f"Blocklisted functions: {blocklist_funcs}")
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

        lang_support = get_language_support(Path(file_path))
        if lang_support.language == Language.PYTHON:
            try:
                ast.parse(f"import {module_name_from_file_path(Path(file_path), project_root)}")
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
                if function.qualified_name_with_modules_from_root(project_root) in previous_checkpoint_functions:
                    previous_checkpoint_functions_removed_count += 1
                    continue
                functions_tmp.append(function)
            _functions = functions_tmp

        filtered_modified_functions[file_path] = _functions
        functions_count += len(_functions)

    if not disable_logs:
        log_info = {
            "Test functions removed": (test_functions_removed_count, "yellow"),
            "Site-package functions removed": (site_packages_removed_count, "magenta"),
            "Non-importable file paths": (malformed_paths_count, "red"),
            "Functions outside module-root": (non_modules_removed_count, "cyan"),
            "Files from ignored paths": (ignore_paths_removed_count, "blue"),
            "Files from ignored submodules": (submodule_ignored_paths_count, "bright_black"),
            "Blocklisted functions removed": (blocklist_funcs_removed_count, "bright_red"),
            "Functions skipped from checkpoint": (previous_checkpoint_functions_removed_count, "green"),
        }
        tree = Tree(Text("Ignored functions and files", style="bold"))
        for label, (count, color) in log_info.items():
            if count > 0:
                tree.add(Text(f"{label}: {count}", style=color))
        if len(tree.children) > 0:
            console.print(tree)
            console.rule()
    return {Path(k): v for k, v in filtered_modified_functions.items() if v}, functions_count


def filter_files_optimized(file_path: Path, tests_root: Path, ignore_paths: list[Path], module_root: Path) -> bool:
    """Optimized version of the filter_functions function above.

    Takes in file paths and returns the count of files that are to be optimized.
    """
    submodule_paths = None
    if file_path.is_relative_to(tests_root):
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


def function_has_return_statement(function_node: FunctionDef | AsyncFunctionDef) -> bool:
    # Custom DFS, return True as soon as a Return node is found
    stack: list[ast.AST] = [function_node]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def function_is_a_property(function_node: FunctionDef | AsyncFunctionDef) -> bool:
    for node in function_node.decorator_list:  # noqa: SIM110
        # Use isinstance rather than type(...) is ... for better performance with single inheritance trees like ast
        if isinstance(node, _ast_name) and node.id == _property_id:
            return True
    return False
