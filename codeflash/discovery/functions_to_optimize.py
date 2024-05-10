import ast
import logging
import os
import random
from _ast import AsyncFunctionDef, ClassDef, FunctionDef
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import git
import libcst as cst
from libcst import CSTNode
from libcst.metadata import CodeRange
from pydantic.dataclasses import dataclass

from codeflash.code_utils.code_utils import (
    module_name_from_file_path,
    path_belongs_to_site_packages,
)
from codeflash.code_utils.git_utils import get_git_diff
from codeflash.verification.verification_utils import TestConfig


@dataclass(frozen=True)
class FunctionProperties:
    is_top_level: bool
    has_args: bool | None


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
            parents: Optional[CSTNode] = self.get_metadata(cst.metadata.ParentNodeProvider, node)
            ast_parents: list[FunctionParent] = []
            while parents is not None:
                if isinstance(parents, (cst.FunctionDef, cst.ClassDef)):
                    ast_parents.append(
                        FunctionParent(parents.name.value, parents.__class__.__name__),
                    )
                parents = self.get_metadata(cst.metadata.ParentNodeProvider, parents, default=None)
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name.value,
                    file_path=self.file_path,
                    parents=list(reversed(ast_parents)),
                    starting_line=pos.start.line,
                    ending_line=pos.end.line,
                ),
            )


class FunctionWithReturnStatement(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.functions: List[FunctionToOptimize] = []
        self.ast_path: List[FunctionParent] = []
        self.file_path: str = file_path

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        # Check if the function has a return statement and add it to the list
        if function_has_return_statement(node):
            self.functions.append(
                FunctionToOptimize(
                    function_name=node.name,
                    file_path=self.file_path,
                    parents=self.ast_path[:],
                ),
            )
        # Continue visiting the body of the function to find nested functions
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.append(FunctionParent(node.name, node.__class__.__name__))
        super().generic_visit(node)
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            self.ast_path.pop()


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str


@dataclass(frozen=True, config=dict(arbitrary_types_allowed=True))
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
    file_path: str
    parents: List[FunctionParent]  # List[ClassDef | FunctionDef | AsyncFunctionDef]
    starting_line: Optional[int] = None
    ending_line: Optional[int] = None

    # # For "BubbleSort.sorter", returns "BubbleSort"
    # # For "sorter", returns "sorter"
    # # TODO does not support nested classes or functions
    @property
    def top_level_parent_name(self) -> str:
        return self.function_name if not self.parents else self.parents[0].name

    def __str__(self) -> str:
        return (
            f"{self.file_path}:{'.'.join([p.name for p in self.parents])}"
            f"{'.' if self.parents else ''}{self.function_name}"
        )

    @property
    def qualified_name(self):
        return self.function_name if self.parents == [] else f"{self.parents[0].name}.{self.function_name}"

    def qualified_name_with_modules_from_root(self, project_root_path: str) -> str:
        return f"{module_name_from_file_path(self.file_path, project_root_path)}.{self.qualified_name}"


def get_functions_to_optimize_by_file(
    optimize_all: Optional[str],
    file: Optional[str],
    only_get_this_function: Optional[str],
    test_cfg: TestConfig,
    ignore_paths: List[str],
    project_root: str,
    module_root: str,
) -> Tuple[Dict[str, List[FunctionToOptimize]], int]:
    if optimize_all:
        logging.info("Finding all functions in the module '%s' ...", optimize_all)
        functions: Dict[str, List[FunctionToOptimize]] = get_all_files_and_functions(optimize_all)
    elif file is not None:
        logging.info("Finding all functions in the file '%s' ...", file)
        functions = find_all_functions_in_file(file)
        if only_get_this_function is not None:
            split_function = only_get_this_function.split(".")
            if len(split_function) > 2:
                raise ValueError(
                    "Function name should be in the format 'function_name' or 'class_name.function_name'",
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
                raise ValueError(
                    f"Function {only_function_name} not found in file {file} or"
                    f" the function does not have a 'return' statement.",
                )
            functions[file] = [found_function]
    else:
        logging.info("Finding all functions modified in the current git diff ...")
        functions = get_functions_within_git_diff()
    filtered_modified_functions, functions_count = filter_functions(
        functions,
        test_cfg.tests_root,
        ignore_paths,
        project_root,
        module_root,
    )
    logging.info("Found %d functions to optimize", functions_count)
    return filtered_modified_functions, functions_count


def get_functions_within_git_diff() -> Dict[str, List[FunctionToOptimize]]:
    modified_lines: dict[str, list[int]] = get_git_diff(uncommitted_changes=False)
    modified_functions: Dict[str, List[FunctionToOptimize]] = {}
    for path in modified_lines:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf8") as f:
            file_content = f.read()
            try:
                wrapper = cst.metadata.MetadataWrapper(cst.parse_module(file_content))
            except Exception as e:
                logging.exception(e)
                continue
            function_lines = FunctionVisitor(file_path=path)
            wrapper.visit(function_lines)
            modified_functions[path] = [
                function_to_optimize
                for function_to_optimize in function_lines.functions
                if (start_line := function_to_optimize.starting_line) is not None
                and (end_line := function_to_optimize.ending_line) is not None
                and any(start_line <= line <= end_line for line in modified_lines[path])
            ]
    return modified_functions


def get_all_files_and_functions(module_root_path: str) -> Dict[str, List[FunctionToOptimize]]:
    functions: Dict[str, List[FunctionToOptimize]] = {}
    for root, dirs, files in os.walk(module_root_path):
        for file in files:
            if not file.endswith(".py"):
                continue
            file_path = os.path.join(root, file)

            # Find all the functions in the file
            functions.update(find_all_functions_in_file(file_path))
    # Randomize the order of the files to optimize to avoid optimizing the same file in the same order every time.
    # Helpful if an optimize-all run is stuck and we restart it.
    files_list = list(functions.items())
    random.shuffle(files_list)
    functions_shuffled = dict(files_list)
    return functions_shuffled


def find_all_functions_in_file(file_path: str) -> Dict[str, List[FunctionToOptimize]]:
    functions: Dict[str, List[FunctionToOptimize]] = {}
    with open(file_path, encoding="utf8") as f:
        try:
            ast_module = ast.parse(f.read())
        except Exception as e:
            logging.exception(e)
            return functions
        function_name_visitor = FunctionWithReturnStatement(file_path)
        function_name_visitor.visit(ast_module)
        functions[file_path] = function_name_visitor.functions
    return functions


def is_git_repo(file_path: str) -> bool:
    try:
        git.Repo(file_path, search_parent_directories=True)
        return True
    except git.InvalidGitRepositoryError:
        return False


@lru_cache(maxsize=None)
def ignored_submodule_paths(module_root: str) -> list[str]:
    if is_git_repo(module_root):
        git_repo = git.Repo(module_root, search_parent_directories=True)
        return [
            os.path.realpath(os.path.join(git_repo.working_tree_dir, submodule.path))
            for submodule in git_repo.submodules
        ]
    else:
        return []


class TopLevelFunctionOrMethodVisitor(ast.NodeVisitor):
    def __init__(
        self,
        file_name: str,
        function_or_method_name: str,
        class_name: Optional[str] = None,
    ) -> None:
        self.file_name = file_name
        self.class_name = class_name
        self.function_name = function_or_method_name
        self.is_top_level = False
        self.function_has_args = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.function_name:
            self.is_top_level = True
            self.function_has_args = any(
                (
                    bool(node.args.args),
                    bool(node.args.kwonlyargs),
                    bool(node.args.kwarg),
                    bool(node.args.posonlyargs),
                    bool(node.args.vararg),
                ),
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # iterate over the class methods
        if node.name == self.class_name:
            for body_node in node.body:
                if isinstance(body_node, ast.FunctionDef) and body_node.name == self.function_name:
                    self.is_top_level = True
                    return
        return


def inspect_top_level_functions_or_methods(
    file_name: str,
    function_or_method_name: str,
    class_name: Optional[str] = None,
) -> FunctionProperties:
    with open(file_name, encoding="utf8") as file:
        try:
            ast_module = ast.parse(file.read())
        except Exception as e:
            logging.exception(e)
            return False
    visitor = TopLevelFunctionOrMethodVisitor(
        file_name=file_name,
        function_or_method_name=function_or_method_name,
        class_name=class_name,
    )
    visitor.visit(ast_module)
    return FunctionProperties(is_top_level=visitor.is_top_level, has_args=visitor.function_has_args)


def filter_functions(
    modified_functions: Dict[str, List[FunctionToOptimize]],
    tests_root: str,
    ignore_paths: List[str],
    project_root: str,
    module_root: str,
    disable_logs: bool = False,
) -> Tuple[Dict[str, List[FunctionToOptimize]], int]:
    # Remove any function that we don't want to optimize

    # Ignore files with submodule path, cache the submodule paths
    submodule_paths = ignored_submodule_paths(module_root)

    filtered_modified_functions: Dict[str, List[FunctionToOptimize]] = {}
    functions_count: int = 0
    test_functions_removed_count: int = 0
    non_modules_removed_count: int = 0
    site_packages_removed_count: int = 0
    ignore_paths_removed_count: int = 0
    malformed_paths_count: int = 0
    submodule_ignored_paths_count: int = 0
    # We desperately need Python 3.10+ only support to make this code readable with structural pattern matching
    for file_path, functions in modified_functions.items():
        if file_path.startswith(tests_root + os.sep):
            test_functions_removed_count += len(functions)
            continue
        if file_path in ignore_paths or any(
            file_path.startswith(ignore_path + os.sep) for ignore_path in ignore_paths if ignore_path
        ):
            ignore_paths_removed_count += 1
            continue
        if file_path in submodule_paths or any(
            file_path.startswith(submodule_path + os.sep) for submodule_path in submodule_paths
        ):
            submodule_ignored_paths_count += 1
            continue
        if path_belongs_to_site_packages(file_path):
            site_packages_removed_count += len(functions)
            continue
        if not file_path.startswith(module_root + os.sep):
            non_modules_removed_count += len(functions)
            continue
        try:
            ast.parse(f"import {module_name_from_file_path(file_path, project_root)}")
        except SyntaxError:
            malformed_paths_count += 1
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
            logging.info(f"Ignoring:\n{log_string}")

    return {k: v for k, v in filtered_modified_functions.items() if v}, functions_count


def filter_files_optimized(
    file_path: str,
    tests_root: str,
    ignore_paths: List[str],
    module_root: str,
) -> bool:
    """Optimized version of the filter_functions function above.
    Takes in file paths and returns the count of files that are to be optimized.
    """
    submodule_paths = None
    if file_path.startswith(tests_root + os.sep):
        print(f"Ignoring file {file_path} because it is in the tests directory")
        return False
    if file_path in ignore_paths or any(
        file_path.startswith(ignore_path + os.sep) for ignore_path in ignore_paths
    ):
        print(f"Ignoring file {file_path} because it is in the ignored paths")
        return False
    if path_belongs_to_site_packages(file_path):
        print(f"Ignoring file {file_path} because it is in the site-packages directory")
        return False
    if not file_path.startswith(module_root + os.sep):
        print(f"Ignoring file {file_path} because it is outside the module-root")
        return False
    if submodule_paths is None:
        submodule_paths = ignored_submodule_paths(module_root)
    if file_path in submodule_paths or any(
        file_path.startswith(submodule_path + os.sep) for submodule_path in submodule_paths
    ):
        print(f"Ignoring file {file_path} because it is in the ignored submodules")
        return False

    return True


def function_has_return_statement(function_node: Union[FunctionDef, AsyncFunctionDef]) -> bool:
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return):
            return True
    return False
