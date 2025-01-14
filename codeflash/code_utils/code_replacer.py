from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module
from codeflash.code_utils.code_utils import cst_to_code, get_only_code_content
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CodeOptimizationContext, OptimizedCandidate, ValidCode

ASTNodeT = TypeVar("ASTNodeT", bound=ast.AST)


def normalize_node(node: ASTNodeT) -> ASTNodeT:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
        node.body = node.body[1:]
    if hasattr(node, "body"):
        node.body = [normalize_node(n) for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return node


@lru_cache(maxsize=3)
def normalize_code(code: str) -> str:
    return ast.unparse(normalize_node(ast.parse(code)))


class OptimFunctionCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(
        self,
        preexisting_objects: list[tuple[str, list[FunctionParent]]] | None = None,
        function_names: set[tuple[str | None, str]] | None = None,
    ) -> None:
        super().__init__()
        self.preexisting_objects = preexisting_objects if preexisting_objects is not None else []

        self.function_names = function_names  # set of (class_name, function_name)
        self.modified_functions: dict[
            tuple[str | None, str], cst.FunctionDef
        ] = {}  # keys are (class_name, function_name)
        self.new_functions: list[cst.FunctionDef] = []
        self.new_class_functions: dict[str, list[cst.FunctionDef]] = defaultdict(list)
        self.current_class = None
        self.modified_init_functions: dict[str, cst.FunctionDef] = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if (self.current_class, node.name.value) in self.function_names:
            self.modified_functions[(self.current_class, node.name.value)] = node
        elif self.current_class and node.name.value == "__init__":
            self.modified_init_functions[self.current_class] = node
        elif (
            self.preexisting_objects
            and (node.name.value, []) not in self.preexisting_objects
            and self.current_class is None
        ):
            self.new_functions.append(node)
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self.current_class:
            return False  # If already in a class, do not recurse deeper
        self.current_class = node.name.value

        parents = [FunctionParent(name=node.name.value, type="ClassDef")]
        for child_node in node.body.body:
            if (
                self.preexisting_objects
                and isinstance(child_node, cst.FunctionDef)
                and (child_node.name.value, parents) not in self.preexisting_objects
            ):
                self.new_class_functions[node.name.value].append(child_node)

        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        if self.current_class:
            self.current_class = None


class OptimFunctionReplacer(cst.CSTTransformer):
    def __init__(
        self,
        modified_functions: dict[tuple[str | None, str], cst.FunctionDef] = None,
        new_functions: list[cst.FunctionDef] = None,
        new_class_functions: dict[str, list[cst.FunctionDef]] = None,
        modified_init_functions: dict[str, cst.FunctionDef] = None,
    ) -> None:
        super().__init__()
        self.modified_functions = modified_functions if modified_functions is not None else {}
        self.new_functions = new_functions if new_functions is not None else []
        self.new_class_functions = new_class_functions if new_class_functions is not None else defaultdict(list)
        self.modified_init_functions: dict[str, cst.FunctionDef] = (
            modified_init_functions if modified_init_functions is not None else {}
        )
        self.current_class = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if (self.current_class, original_node.name.value) in self.modified_functions:
            node = self.modified_functions[(self.current_class, original_node.name.value)]
            if get_only_code_content(cst_to_code(original_node)) == get_only_code_content(cst_to_code(node)):
                return original_node  # Code was unchanged, so don't modify docstrings / comments
            return updated_node.with_changes(body=node.body, decorators=node.decorators)
        if original_node.name.value == "__init__" and self.current_class in self.modified_init_functions:
            if get_only_code_content(cst_to_code(original_node)) == get_only_code_content(
                cst_to_code(self.modified_init_functions[self.current_class])
            ):
                return original_node  # Code was unchanged, so don't modify docstrings / comments
            return merge_init_functions(updated_node, self.modified_init_functions[self.current_class])

        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        if self.current_class:
            return False  # If already in a class, do not recurse deeper
        self.current_class = node.name.value
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        if self.current_class and self.current_class == original_node.name.value:
            self.current_class = None
            if original_node.name.value in self.new_class_functions:
                return updated_node.with_changes(
                    body=updated_node.body.with_changes(
                        body=(list(updated_node.body.body) + list(self.new_class_functions[original_node.name.value]))
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
                body=(*node.body[: max_function_index + 1], *self.new_functions, *node.body[max_function_index + 1 :])
            )
        elif class_index is not None:
            node = node.with_changes(
                body=(*node.body[: class_index + 1], *self.new_functions, *node.body[class_index + 1 :])
            )
        else:
            node = node.with_changes(body=(*self.new_functions, *node.body))
        return node


class AttributeCollector(cst.CSTVisitor):
    """Collects all self.attribute mentions in a CST."""

    def __init__(self):
        super().__init__()
        self.attributes: set[str] = set()

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        """Record any self.attribute access."""
        if isinstance(node.value, cst.Name) and node.value.value == "self":
            self.attributes.add(node.attr.value)
        return True


class AssignmentCollector(cst.CSTVisitor):
    """Collects attributes being assigned to in a CST."""

    def __init__(self):
        super().__init__()
        self.assigned_attrs: set[str] = set()

    def visit_Assign(self, node: cst.Assign) -> bool:
        """Check regular assignments like self.x = ..."""
        for target in node.targets:
            if (
                isinstance(target.target, cst.Attribute)
                and isinstance(target.target.value, cst.Name)
                and target.target.value.value == "self"
            ):
                self.assigned_attrs.add(target.target.attr.value)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        """Check annotated assignments like self.x: str = ..."""
        if (
            isinstance(node.target, cst.Attribute)
            and isinstance(node.target.value, cst.Name)
            and node.target.value.value == "self"
        ):
            self.assigned_attrs.add(node.target.attr.value)
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        """Check augmented assignments like self.x += ..."""
        if (
            isinstance(node.target, cst.Attribute)
            and isinstance(node.target.value, cst.Name)
            and node.target.value.value == "self"
        ):
            self.assigned_attrs.add(node.target.attr.value)
        return True


def merge_init_functions(original_init: cst.FunctionDef, new_init: cst.FunctionDef) -> cst.FunctionDef:
    """Merges two __init__ function definitions. Collects all self.attribute mentions
    from the original init, then filters out statements from the new init that
    assign to those attributes (but allows reading them).

    Args:
        original_init: The original __init__ function to preserve
        new_init: The new __init__ function whose body will be filtered and appended

    Returns:
        A merged FunctionDef

    """
    # Collect all self.attribute mentions from original init
    collector = AttributeCollector()
    original_init.visit(collector)
    existing_attrs = collector.attributes
    # Get set of existing statements as strings. # This should just be in terms of code, not comments?
    original_stmts = {cst.Module([stmt]).code for stmt in original_init.body.body}
    # Filter new init body statements
    filtered_body = []
    for stmt in new_init.body.body:
        if cst.Module([stmt]).code in original_stmts:
            continue
        # Check for assignments to existing attributes
        assign_collector = AssignmentCollector()
        stmt.visit(assign_collector)

        # Keep statement if it doesn't assign to any existing attributes
        if not assign_collector.assigned_attrs.intersection(existing_attrs):
            filtered_body.append(stmt)

    # Merge bodies using with_changes
    return original_init.with_changes(
        body=original_init.body.with_changes(body=original_init.body.body + tuple(filtered_body))
    )


def replace_functions_in_file(
    source_code: str,
    original_function_names: list[str],
    optimized_code: str,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
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
        parsed_function_names.append((class_name, function_name))

    # Collect functions we want to modify from the optimized code
    module = cst.metadata.MetadataWrapper(cst.parse_module(optimized_code))
    visitor = OptimFunctionCollector(preexisting_objects, set(parsed_function_names))
    module.visit(visitor)

    # Replace these functions in the original code
    transformer = OptimFunctionReplacer(
        modified_functions=visitor.modified_functions,
        new_functions=visitor.new_functions,
        new_class_functions=visitor.new_class_functions,
        modified_init_functions=visitor.modified_init_functions,
    )
    original_module = cst.parse_module(source_code)
    modified_tree = original_module.visit(transformer)
    return modified_tree.code


def replace_functions_and_add_imports(
    source_code: str,
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: Path,
    module_abspath: Path,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
    project_root_path: Path,
) -> str:
    return add_needed_imports_from_module(
        optimized_code,
        replace_functions_in_file(source_code, function_names, optimized_code, preexisting_objects),
        module_abspath,
        module_abspath,
        project_root_path,
    )


def replace_function_definitions_in_module(
    function_names: list[str],
    optimized_code: str,
    file_path_of_module_with_function_to_optimize: Path,
    module_abspath: Path,
    preexisting_objects: list[tuple[str, list[FunctionParent]]],
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
