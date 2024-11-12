from __future__ import annotations

import ast
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from pathlib import Path

    from libcst import FunctionDef

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CodeOptimizationContext, OptimizedCandidate, ValidCode

ASTNodeT = TypeVar("ASTNodeT", bound=ast.AST)


def normalize_node(node: ASTNodeT) -> ASTNodeT:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
        node.body = node.body[1:]
    if hasattr(node, "body"):
        node.body = [normalize_node(node) for node in node.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return node


@lru_cache(maxsize=3)
def normalize_code(code: str) -> str:
    return ast.unparse(normalize_node(ast.parse(code)))


class OptimFunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(
        self,
        function_name: str,
        class_name: str | None,
        contextual_functions: set[tuple[str, str]],
        preexisting_objects: list[tuple[str, list[FunctionParent]]] | None = None,
    ) -> None:
        super().__init__()
        if preexisting_objects is None:
            preexisting_objects = []
        self.function_name = function_name
        self.class_name = class_name
        self.optim_body: FunctionDef | None = None
        self.optim_new_class_functions: list[cst.FunctionDef] = []
        self.optim_new_functions: list[cst.FunctionDef] = []
        self.preexisting_objects = preexisting_objects
        self.contextual_functions = contextual_functions.union({(self.class_name, self.function_name)})

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
        parent2 = None
        try:
            if parent is not None and isinstance(parent, cst.Module):
                parent2 = self.get_metadata(cst.metadata.ParentNodeProvider, parent)
        except:
            pass
        if node.name.value == self.function_name:
            self.optim_body = node
        elif (
            self.preexisting_objects
            and (node.name.value, []) not in self.preexisting_objects
            and (isinstance(parent, cst.Module) or (parent2 is not None and not isinstance(parent2, cst.ClassDef)))
        ):
            self.optim_new_functions.append(node)

    def visit_ClassDef_body(self, node: cst.ClassDef) -> None:
        parents = [FunctionParent(name=node.name.value, type="ClassDef")]
        for child_node in node.body.body:
            if (
                self.preexisting_objects
                and isinstance(child_node, cst.FunctionDef)
                and (node.name.value, child_node.name.value) not in self.contextual_functions
                and (child_node.name.value, parents) not in self.preexisting_objects
            ):
                self.optim_new_class_functions.append(child_node)


class OptimFunctionReplacer(cst.CSTTransformer):
    def __init__(
        self,
        function_name: str,
        optim_body: cst.FunctionDef,
        optim_new_class_functions: list[cst.FunctionDef],
        optim_new_functions: list[cst.FunctionDef],
        class_name: str | None = None,
    ) -> None:
        super().__init__()
        self.function_name = function_name
        self.optim_body = optim_body
        self.optim_new_class_functions = optim_new_class_functions
        self.optim_new_functions = optim_new_functions
        self.class_name = class_name
        self.depth: int = 0
        self.in_class: bool = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.function_name and (self.depth == 0 or (self.depth == 1 and self.in_class)):
            return updated_node.with_changes(body=self.optim_body.body, decorators=self.optim_body.decorators)
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self.depth += 1
        if self.in_class:
            return False
        self.in_class = (self.depth == 1) and (node.name.value == self.class_name)
        return self.in_class

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.depth -= 1
        if self.in_class and (self.depth == 0) and (original_node.name.value == self.class_name):
            self.in_class = False
            return updated_node.with_changes(
                body=updated_node.body.with_changes(
                    body=(list(updated_node.body.body) + self.optim_new_class_functions)
                )
            )
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        node = updated_node
        max_function_index = None
        class_index = None
        for index, _node in enumerate(node.body):
            if isinstance(_node, cst.FunctionDef):
                max_function_index = index
            if isinstance(_node, cst.ClassDef):
                class_index = index
        if max_function_index is not None:
            node = node.with_changes(
                body=(
                    *node.body[: max_function_index + 1],
                    *self.optim_new_functions,
                    *node.body[max_function_index + 1 :],
                )
            )
        elif class_index is not None:
            node = node.with_changes(
                body=(*node.body[: class_index + 1], *self.optim_new_functions, *node.body[class_index + 1 :])
            )
        else:
            node = node.with_changes(body=(*self.optim_new_functions, *node.body))
        return node


def replace_functions_in_file(
    source_code: str,
    original_function_names: list[str],
    optimized_code: str,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
    contextual_functions: set[tuple[str, str]],
) -> str:
    parsed_function_names = []
    for original_function_name in original_function_names:
        if original_function_name.count(".") == 0:
            class_name, function_name = None, original_function_name
        elif original_function_name.count(".") == 1:
            class_name, function_name = original_function_name.split(".")
        else:
            msg = f"Unable to find {original_function_name}. Returning unchanged source code."
            logger.error(msg)
            return source_code
        parsed_function_names.append((function_name, class_name))

    module = cst.metadata.MetadataWrapper(cst.parse_module(optimized_code))

    for function_name, class_name in parsed_function_names:
        visitor = OptimFunctionCollector(function_name, class_name, contextual_functions, preexisting_objects)
        module.visit(visitor)

        if visitor.optim_body is None and not preexisting_objects:
            continue
        if visitor.optim_body is None:
            msg = f"Unable to find function {function_name} in optimized code. Returning unchanged source code."
            logger.error(msg)
            return source_code

        transformer = OptimFunctionReplacer(
            visitor.function_name,
            visitor.optim_body,
            visitor.optim_new_class_functions,
            visitor.optim_new_functions,
            class_name=class_name,
        )
        original_module = cst.parse_module(source_code)
        modified_tree = original_module.visit(transformer)
        source_code = modified_tree.code

    return source_code


def replace_functions_and_add_imports(
    source_code: str,
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: Path,
    module_abspath: Path,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
    contextual_functions: set[tuple[str, str]],
    project_root_path: Path,
) -> str:
    return add_needed_imports_from_module(
        optimized_code,
        replace_functions_in_file(
            source_code, function_names, optimized_code, preexisting_objects, contextual_functions
        ),
        file_path_of_module_with_function_to_optimize,
        module_abspath,
        project_root_path,
    )


def replace_function_definitions_in_module(
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: Path,
    module_abspath: Path,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
    contextual_functions: set[tuple[str, str]],
    project_root_path: Path,
) -> bool:
    source_code: str = module_abspath.read_text(encoding="utf8")
    new_code: str = replace_functions_and_add_imports(
        source_code,
        function_names,
        optimized_code,
        file_path_of_module_with_function_to_optimize,
        module_abspath,
        preexisting_objects,
        contextual_functions,
        project_root_path,
    )
    if is_zero_diff(source_code, new_code):
        return False
    module_abspath.write_text(new_code, encoding="utf8")
    return True


def is_zero_diff(original_code: str, new_code: str) -> bool:
    return normalize_code(original_code) == normalize_code(new_code)


def replace_optimized_code(
    callee_module_paths: set[Path],
    candidates: list[OptimizedCandidate],
    code_context: CodeOptimizationContext,
    function_to_optimize: FunctionToOptimize,
    validated_original_code: dict[Path, ValidCode],
    project_root: Path,
) -> tuple[set[Path], dict[str, dict[Path, str]]]:
    initial_optimized_code = {
        candidate.optimization_id: replace_functions_and_add_imports(
            validated_original_code[function_to_optimize.file_path].source_code,
            [function_to_optimize.qualified_name],
            candidate.source_code,
            function_to_optimize.file_path,
            function_to_optimize.file_path,
            code_context.preexisting_objects,
            code_context.contextual_dunder_methods,
            project_root,
        )
        for candidate in candidates
    }
    callee_original_code = {
        module_path: validated_original_code[module_path].source_code for module_path in callee_module_paths
    }
    intermediate_original_code: dict[str, dict[Path, str]] = {
        candidate.optimization_id: (
            callee_original_code | {function_to_optimize.file_path: initial_optimized_code[candidate.optimization_id]}
        )
        for candidate in candidates
    }
    module_paths = callee_module_paths | {function_to_optimize.file_path}
    optimized_code = {
        candidate.optimization_id: {
            module_path: replace_functions_and_add_imports(
                intermediate_original_code[candidate.optimization_id][module_path],
                (
                    [
                        callee.qualified_name
                        for callee in code_context.helper_functions
                        if callee.file_path == module_path and callee.jedi_definition.type != "class"
                    ]
                ),
                candidate.source_code,
                function_to_optimize.file_path,
                module_path,
                [],
                code_context.contextual_dunder_methods,
                project_root,
            )
            for module_path in module_paths
        }
        for candidate in candidates
    }
    return module_paths, optimized_code


def is_optimized_module_code_zero_diff(
    candidates: list[OptimizedCandidate],
    validated_original_code: dict[Path, ValidCode],
    optimized_code: dict[str, dict[Path, str]],
    module_paths: set[Path],
) -> dict[str, dict[Path, bool]]:
    return {
        candidate.optimization_id: {
            callee_module_path: normalize_code(optimized_code[candidate.optimization_id][callee_module_path])
            == validated_original_code[callee_module_path].normalized_code
            for callee_module_path in module_paths
        }
        for candidate in candidates
    }


def candidates_with_diffs(
    candidates: list[OptimizedCandidate],
    validated_original_code: ValidCode,
    optimized_code: dict[str, dict[Path, str]],
    module_paths: set[Path],
) -> list[OptimizedCandidate]:
    return [
        candidate
        for candidate in candidates
        if not all(
            is_optimized_module_code_zero_diff(candidates, validated_original_code, optimized_code, module_paths)[
                candidate.optimization_id
            ].values()
        )
    ]


def replace_optimized_code_in_worktrees(
    optimized_code: dict[str, dict[Path, str]],
    candidates: list[OptimizedCandidate],  # Should be candidates_with_diffs
    worktrees: list[Path],
    git_root: Path,  # Handle None case
) -> None:
    for candidate, worktree in zip(candidates, worktrees[1:]):
        for module_path in optimized_code[candidate.optimization_id]:
            (worktree / module_path.relative_to(git_root)).write_text(
                optimized_code[candidate.optimization_id][module_path], encoding="utf8"
            )  # Check with is_optimized_module_code_zero_diff


def function_to_optimize_original_worktree_fqn(
    function_to_optimize: FunctionToOptimize, worktrees: list[Path], git_root: Path
) -> str:
    return (
        str(worktrees[0].name / function_to_optimize.file_path.relative_to(git_root).with_suffix("")).replace("/", ".")
        + "."
        + function_to_optimize.qualified_name
    )
