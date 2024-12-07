from __future__ import annotations

import ast
import os
import random
import warnings
from _ast import AsyncFunctionDef, ClassDef, FunctionDef
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git
import libcst as cst
from pydantic.dataclasses import dataclass

from codeflash.api.cfapi import get_blocklisted_functions
from codeflash.cli_cmds.console import DEBUG_MODE, console, logger
from codeflash.code_utils.code_utils import (
    is_class_defined_in_file,
    module_name_from_file_path,
    path_belongs_to_site_packages,
)
from codeflash.code_utils.git_utils import get_git_diff
from codeflash.discovery.discover_unit_tests import discover_unit_tests
from codeflash.models.models import FunctionParent
from codeflash.telemetry.posthog_cf import ph

if TYPE_CHECKING:
    from libcst import CSTNode
    from libcst.metadata import CodeRange

    from codeflash.verification.verification_utils import TestConfig


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

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        return_visitor: ReturnStatementVisitor = ReturnStatementVisitor()
        node.visit(return_visitor)
        if return_visitor.has_return_statement:
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
                )
            )


class FunctionWithReturnStatement(ast.NodeVisitor):
    def __init__(self, file_path: Path) -> None:
        self.functions: list[FunctionToOptimize] = []
        self.ast_path: list[FunctionParent] = []
        self.file_path: Path = file_path

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        # Check if the function has a return statement and add it to the list
        if function_has_return_statement(node) and not function_is_a_property(node):
            self.functions.append(
                FunctionToOptimize(function_name=node.name, file_path=self.file_path, parents=self.ast_path[:])
            )
        # Continue visiting the body of the function to find nested functions
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.append(FunctionParent(node.name, node.__class__.__name__))
        super().generic_visit(node)
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.pop()


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FunctionToOptimize:
    """Represents a function that is a candidate for optimization.

    Attributes
    ----------
        function_name: The name of the function.
        file_path: The absolute file path where the function is located.
        parents: A list of parent scopes, which could be classes or functions.
        starting_line: The starting line number of the function in the file.
        ending_line: The ending line number of the function in the file.

    The qualified_name property provides the full name of the function, including
    any parent class or function names. The qualified_name_with_modules_from_root
    method extends this with the module name from the project root.

    """

    function_name: str
    file_path: Path
    parents: list[FunctionParent]  # list[ClassDef | FunctionDef | AsyncFunctionDef]
    starting_line: Optional[int] = None
    ending_line: Optional[int] = None

    @property
    def top_level_parent_name(self) -> str:
        return self.function_name if not self.parents else self.parents[0].name

    def __str__(self) -> str:
        return (
            f"{self.file_path}:{'.'.join([p.name for p in self.parents])}"
            f"{'.' if self.parents else ''}{self.function_name}"
        )

    @property
    def qualified_name(self) -> str:
        return self.function_name if self.parents == [] else f"{self.parents[0].name}.{self.function_name}"

    def qualified_name_with_modules_from_root(self, project_root_path: Path) -> str:
        return f"{module_name_from_file_path(self.file_path, project_root_path)}.{self.qualified_name}"


def get_functions_to_optimize(
    optimize_all: str | None,
    replay_test: str | None,
    file: Path | None,
    only_get_this_function: str | None,
    test_cfg: TestConfig,
    ignore_paths: list[Path],
    project_root: Path,
    module_root: Path,
) -> tuple[dict[Path, list[FunctionToOptimize]], int]:
    assert (
        sum([bool(optimize_all), bool(replay_test), bool(file)]) <= 1
    ), "Only one of optimize_all, replay_test, or file should be provided"
    functions: dict[str, list[FunctionToOptimize]]
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=SyntaxWarning)
        if optimize_all:
            logger.info("Finding all functions in the module '%s'…", optimize_all)
            console.rule()
            functions = get_all_files_and_functions(Path(optimize_all))
        elif replay_test is not None:
            functions = get_all_replay_test_functions(
                replay_test=replay_test, test_cfg=test_cfg, project_root_path=project_root
            )
        elif file is not None:
            logger.info("Finding all functions in the file '%s'…", file)
            console.rule()
            functions = find_all_functions_in_file(file)
            if only_get_this_function is not None:
                split_function = only_get_this_function.split(".")
                if len(split_function) > 2:
                    msg = "Function name should be in the format 'function_name' or 'class_name.function_name'"
                    raise ValueError(msg)
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
                    msg = f"Function {only_function_name} not found in file {file} or the function does not have a 'return' statement or is a property"
                    raise ValueError(msg)
                functions[file] = [found_function]
        else:
            logger.info("Finding all functions modified in the current git diff ...")
            ph("cli-optimizing-git-diff")
            functions = get_functions_within_git_diff()
        filtered_modified_functions, functions_count = filter_functions(
            functions, test_cfg.tests_root, ignore_paths, project_root, module_root
        )
        filtered_modified_functions, functions_count = filter_functions(
            functions, test_cfg.tests_root, ignore_paths, project_root, module_root
        )
        logger.info(f"Found {functions_count} function{'s' if functions_count > 1 else ''} to optimize")
        return filtered_modified_functions, functions_count


def get_functions_within_git_diff() -> dict[str, list[FunctionToOptimize]]:
    modified_lines: dict[str, list[int]] = get_git_diff(uncommitted_changes=False)
    modified_functions: dict[str, list[FunctionToOptimize]] = {}
    for path_str in modified_lines:
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
            modified_functions[str(path)] = [
                function_to_optimize
                for function_to_optimize in function_lines.functions
                if (start_line := function_to_optimize.starting_line) is not None
                and (end_line := function_to_optimize.ending_line) is not None
                and any(start_line <= line <= end_line for line in modified_lines[path_str])
            ]
    return modified_functions


def get_all_files_and_functions(module_root_path: Path) -> dict[str, list[FunctionToOptimize]]:
    functions: dict[str, list[FunctionToOptimize]] = {}
    for file_path in module_root_path.rglob("*.py"):
        # Find all the functions in the file
        functions.update(find_all_functions_in_file(file_path).items())
    # Randomize the order of the files to optimize to avoid optimizing the same file in the same order every time.
    # Helpful if an optimize-all run is stuck and we restart it.
    files_list = list(functions.items())
    random.shuffle(files_list)
    return dict(files_list)


def find_all_functions_in_file(file_path: Path) -> dict[Path, list[FunctionToOptimize]]:
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


def get_all_replay_test_functions(
    replay_test: Path, test_cfg: TestConfig, project_root_path: Path
) -> dict[Path, list[FunctionToOptimize]]:
    function_tests = discover_unit_tests(test_cfg, discover_only_these_tests=[replay_test])
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
            function = class_name + "." + function_name
            file_path_parts = module_path_parts[:-1]  # Exclude the class name
        else:
            function = function_name
            file_path_parts = module_path_parts
        file_path = Path(project_root_path, *file_path_parts).with_suffix(".py")
        file_to_functions_map[file_path].append((function, function_name, class_name))
    for file_path, functions in file_to_functions_map.items():
        all_valid_functions: dict[Path, list[FunctionToOptimize]] = find_all_functions_in_file(file_path=file_path)
        filtered_list = []
        for function in functions:
            function_name, function_name_only, class_name = function
            filtered_list.extend(
                [
                    valid_function
                    for valid_function in all_valid_functions[file_path]
                    if valid_function.qualified_name == function_name
                ]
            )
        if len(filtered_list):
            filtered_valid_functions[file_path] = filtered_list

    return filtered_valid_functions


def is_git_repo(file_path: str) -> bool:
    try:
        git.Repo(file_path, search_parent_directories=True)
        return True
    except git.InvalidGitRepositoryError:
        return False


@cache
def ignored_submodule_paths(module_root: str) -> list[str]:
    if is_git_repo(module_root):
        git_repo = git.Repo(module_root, search_parent_directories=True)
        return [Path(git_repo.working_tree_dir, submodule.path).resolve() for submodule in git_repo.submodules]
    return []


class TopLevelFunctionOrMethodVisitor(ast.NodeVisitor):
    def __init__(
        self, file_name: Path, function_or_method_name: str, class_name: str | None = None, line_no: int | None = None
    ) -> None:
        self.file_name = file_name
        self.class_name = class_name
        self.function_name = function_or_method_name
        self.is_top_level = False
        self.function_has_args = None
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

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # iterate over the class methods
        if node.name == self.class_name:
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef) and body_node.name == self.function_name:
                    self.is_top_level = True
                    if any(
                        isinstance(decorator, ast.Name) and decorator.id == "classmethod"
                        for decorator in body_node.decorator_list
                    ):
                        self.is_classmethod = True
                    return
        else:
            # search if the class has a staticmethod with the same name and on the same line number
            for body_node in node.body:
                if (
                    isinstance(body_node, ast.FunctionDef)
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
) -> FunctionProperties:
    with open(file_name, encoding="utf8") as file:
        try:
            ast_module = ast.parse(file.read())
        except Exception as e:
            logger.exception(e)
            return False
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


def filter_functions(
    modified_functions: dict[Path, list[FunctionToOptimize]],
    tests_root: Path,
    ignore_paths: list[Path],
    project_root: Path,
    module_root: Path,
    disable_logs: bool = False,
) -> tuple[dict[Path, list[FunctionToOptimize]], int]:
    blocklist_funcs = get_blocklisted_functions()
    # Remove any function that we don't want to optimize

    # Ignore files with submodule path, cache the submodule paths
    submodule_paths = ignored_submodule_paths(module_root)

    filtered_modified_functions: dict[str, list[FunctionToOptimize]] = {}
    functions_count: int = 0
    test_functions_removed_count: int = 0
    non_modules_removed_count: int = 0
    site_packages_removed_count: int = 0
    ignore_paths_removed_count: int = 0
    malformed_paths_count: int = 0
    submodule_ignored_paths_count: int = 0
    tests_root_str = str(tests_root)
    module_root_str = str(module_root)
    # We desperately need Python 3.10+ only support to make this code readable with structural pattern matching
    for file_path_path, functions in modified_functions.items():
        file_path = str(file_path_path)
        if file_path.startswith(tests_root_str + os.sep):
            test_functions_removed_count += len(functions)
            continue
        if file_path in ignore_paths or any(
            # file_path.startswith(ignore_path + os.sep) for ignore_path in ignore_paths if ignore_path
            file_path.startswith(str(ignore_path) + os.sep)
            for ignore_path in ignore_paths
        ):
            ignore_paths_removed_count += 1
            continue
        if file_path in submodule_paths or any(
            file_path.startswith(str(submodule_path) + os.sep) for submodule_path in submodule_paths
        ):
            submodule_ignored_paths_count += 1
            continue
        if path_belongs_to_site_packages(Path(file_path)):
            site_packages_removed_count += len(functions)
            continue
        if not file_path.startswith(module_root_str + os.sep):
            non_modules_removed_count += len(functions)
            continue
        try:
            ast.parse(f"import {module_name_from_file_path(Path(file_path), project_root)}")
        except SyntaxError:
            malformed_paths_count += 1
            continue
        if blocklist_funcs:
            for function in functions.copy():
                path = Path(function.file_path).name
                if path in blocklist_funcs and function.function_name in blocklist_funcs[path]:
                    functions.remove(function)
                    logger.debug(f"Skipping {function.function_name} in {path} as it has already been optimized")
                    continue

        filtered_modified_functions[file_path] = functions
        functions_count += len(functions)
    if not disable_logs:
        log_info = {
            f"{test_functions_removed_count} test function{'s' if test_functions_removed_count != 1 else ''}": test_functions_removed_count,
            f"{site_packages_removed_count} site-package function{'s' if site_packages_removed_count != 1 else ''}": site_packages_removed_count,
            f"{malformed_paths_count} non-importable file path{'s' if malformed_paths_count != 1 else ''}": malformed_paths_count,
            f"{non_modules_removed_count} function{'s' if non_modules_removed_count != 1 else ''} outside module-root": non_modules_removed_count,
            f"{ignore_paths_removed_count} file{'s' if ignore_paths_removed_count != 1 else ''} from ignored paths": ignore_paths_removed_count,
            f"{submodule_ignored_paths_count} file{'s' if submodule_ignored_paths_count != 1 else ''} from ignored submodules": submodule_ignored_paths_count,
        }
        log_string: str
        if log_string := "\n".join([k for k, v in log_info.items() if v > 0]):
            logger.info(f"Ignoring: {log_string}")
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
    return any(isinstance(node, ast.Return) for node in ast.walk(function_node))


def function_is_a_property(function_node: FunctionDef | AsyncFunctionDef) -> bool:
    return any(isinstance(node, ast.Name) and node.id == "property" for node in function_node.decorator_list)
