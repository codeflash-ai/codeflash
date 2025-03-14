from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, GatherImportsVisitor, RemoveImportsVisitor
from libcst.helpers import calculate_module_and_package

from codeflash.cli_cmds.console import logger
from codeflash.models.models import FunctionParent, FunctionSource

if TYPE_CHECKING:
    from libcst.helpers import ModuleNameAndPackage

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class FutureAliasedImportTransformer(cst.CSTTransformer):
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        if (
            (updated_node_module := updated_node.module)
            and updated_node_module.value == "__future__"
            and all(m.matches(name, m.ImportAlias()) for name in updated_node.names)
        ):
            if names := [name for name in updated_node.names if name.asname is None]:
                return updated_node.with_changes(names=names)
            return cst.RemoveFromParent()
        return updated_node


def delete___future___aliased_imports(module_code: str) -> str:
    return cst.parse_module(module_code).visit(FutureAliasedImportTransformer()).code


def add_needed_imports_from_module(
    src_module_code: str,
    dst_module_code: str,
    src_path: Path,
    dst_path: Path,
    project_root: Path,
    helper_functions: list[FunctionSource] | None = None,
    helper_functions_fqn: set[str] | None = None,
) -> str:
    """Add all needed and used source module code imports to the destination module code, and return it."""
    src_module_code = delete___future___aliased_imports(src_module_code)
    if not helper_functions_fqn:
        helper_functions_fqn = {f.fully_qualified_name for f in (helper_functions or [])}

    src_module_and_package: ModuleNameAndPackage = calculate_module_and_package(project_root, src_path)
    dst_module_and_package: ModuleNameAndPackage = calculate_module_and_package(project_root, dst_path)

    dst_context: CodemodContext = CodemodContext(
        filename=src_path.name,
        full_module_name=dst_module_and_package.name,
        full_package_name=dst_module_and_package.package,
    )
    gatherer: GatherImportsVisitor = GatherImportsVisitor(
        CodemodContext(
            filename=src_path.name,
            full_module_name=src_module_and_package.name,
            full_package_name=src_module_and_package.package,
        )
    )
    cst.parse_module(src_module_code).visit(gatherer)
    try:
        for mod in gatherer.module_imports:
            AddImportsVisitor.add_needed_import(dst_context, mod)
            RemoveImportsVisitor.remove_unused_import(dst_context, mod)
        for mod, obj_seq in gatherer.object_mapping.items():
            for obj in obj_seq:
                if f"{mod}.{obj}" in helper_functions_fqn:
                    continue  # Skip adding imports for helper functions already in the context
                AddImportsVisitor.add_needed_import(dst_context, mod, obj)
                RemoveImportsVisitor.remove_unused_import(dst_context, mod, obj)
    except Exception as e:
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_module_code
    for mod, asname in gatherer.module_aliases.items():
        AddImportsVisitor.add_needed_import(dst_context, mod, asname=asname)
        RemoveImportsVisitor.remove_unused_import(dst_context, mod, asname=asname)
    for mod, alias_pairs in gatherer.alias_mapping.items():
        for alias_pair in alias_pairs:
            if f"{mod}.{alias_pair[0]}" in helper_functions_fqn:
                continue
            AddImportsVisitor.add_needed_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])
            RemoveImportsVisitor.remove_unused_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])

    try:
        parsed_module = cst.parse_module(dst_module_code)
    except cst.ParserSyntaxError as e:
        logger.exception(f"Syntax error in destination module code: {e}")
        return dst_module_code  # Return the original code if there's a syntax error
    try:
        transformed_module = AddImportsVisitor(dst_context).transform_module(parsed_module)
        transformed_module = RemoveImportsVisitor(dst_context).transform_module(transformed_module)
        return transformed_module.code.lstrip("\n")
    except Exception as e:
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_module_code


def get_code(functions_to_optimize: list[FunctionToOptimize]) -> tuple[str | None, set[tuple[str, str]]]:
    """Return the code for a function or methods in a Python module. functions_to_optimize is either a singleton
    FunctionToOptimize instance, which represents either a function at the module level or a method of a class at the
    module level, or it represents a list of methods of the same class.
    """
    if (
        not functions_to_optimize
        or (functions_to_optimize[0].parents and functions_to_optimize[0].parents[0].type != "ClassDef")
        or (
            len(functions_to_optimize[0].parents) > 1
            or ((len(functions_to_optimize) > 1) and len({fn.parents[0] for fn in functions_to_optimize}) != 1)
        )
    ):
        return None, set()

    file_path: Path = functions_to_optimize[0].file_path
    class_skeleton: set[tuple[int, int | None]] = set()
    contextual_dunder_methods: set[tuple[str, str]] = set()
    target_code: str = ""

    def find_target(node_list: list[ast.stmt], name_parts: tuple[str, str] | tuple[str]) -> ast.AST | None:
        target: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Assign | ast.AnnAssign | None = None
        node: ast.stmt
        for node in node_list:
            if (
                # The many mypy issues will be fixed once this code moves to the backend,
                # using Type Guards as we move to 3.10+.
                # We will cover the Type Alias case on the backend since it's a 3.12 feature.
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name_parts[0]
            ):
                target = node
                break
                # The next two cases cover type aliases in pre-3.12 syntax, where only single assignment is allowed.
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name_parts[0]
            ) or (isinstance(node, ast.AnnAssign) and hasattr(node.target, "id") and node.target.id == name_parts[0]):
                if class_skeleton:
                    break
                target = node
                break

        if target is None or len(name_parts) == 1:
            return target

        if not isinstance(target, ast.ClassDef):
            return None
        class_skeleton.add((target.lineno, target.body[0].lineno - 1))
        cbody = target.body
        if isinstance(cbody[0], ast.expr):  # Is a docstring
            class_skeleton.add((cbody[0].lineno, cbody[0].end_lineno))
            cbody = cbody[1:]
            cnode: ast.stmt
        for cnode in cbody:
            # Collect all dunder methods.
            cnode_name: str
            if (
                isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef))
                and len(cnode_name := cnode.name) > 4
                and cnode_name != name_parts[1]
                and cnode_name.isascii()
                and cnode_name.startswith("__")
                and cnode_name.endswith("__")
            ):
                contextual_dunder_methods.add((target.name, cnode_name))
                class_skeleton.add((cnode.lineno, cnode.end_lineno))

        return find_target(target.body, name_parts[1:])

    with open(file_path, encoding="utf8") as file:
        source_code: str = file.read()
    try:
        module_node: ast.Module = ast.parse(source_code)
    except SyntaxError:
        logger.exception("get_code - Syntax error while parsing code")
        return None, set()
    # Get the source code lines for the target node
    lines: list[str] = source_code.splitlines(keepends=True)
    if len(functions_to_optimize[0].parents) == 1:
        if (
            functions_to_optimize[0].parents[0].type == "ClassDef"
        ):  # All functions_to_optimize functions are methods of the same class.
            qualified_name_parts_list: list[tuple[str, str] | tuple[str]] = [
                (fto.parents[0].name, fto.function_name) for fto in functions_to_optimize
            ]

        else:
            logger.error(f"Error: get_code does not support inner functions: {functions_to_optimize[0].parents}")
            return None, set()
    elif len(functions_to_optimize[0].parents) == 0:
        qualified_name_parts_list = [(functions_to_optimize[0].function_name,)]
    else:
        logger.error(
            "Error: get_code does not support more than one level of nesting for now. "
            f"Parents: {functions_to_optimize[0].parents}"
        )
        return None, set()
    for qualified_name_parts in qualified_name_parts_list:
        target_node: ast.AST | None = find_target(module_node.body, qualified_name_parts)
        if target_node is None:
            continue

        if (
            isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and target_node.decorator_list
        ):
            target_code += "".join(lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno])
        else:
            target_code += "".join(lines[target_node.lineno - 1 : target_node.end_lineno])
    if not target_code:
        return None, set()
    class_list: list[tuple[int, int | None]] = sorted(class_skeleton)
    class_code = "".join(["".join(lines[s_lineno - 1 : e_lineno]) for (s_lineno, e_lineno) in class_list])
    return class_code + target_code, contextual_dunder_methods


def extract_code(functions_to_optimize: list[FunctionToOptimize]) -> tuple[str | None, set[tuple[str, str]]]:
    edited_code, contextual_dunder_methods = get_code(functions_to_optimize)
    if edited_code is None:
        return None, set()
    try:
        compile(edited_code, "edited_code", "exec")
    except SyntaxError as e:
        logger.exception(f"extract_code - Syntax error in extracted optimization candidate code: {e}")
        return None, set()
    return edited_code, contextual_dunder_methods


def find_preexisting_objects(source_code: str) -> set[tuple[str, tuple[FunctionParent, ...]]]:
    """Find all preexisting functions, classes or class methods in the source code."""
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = set()
    try:
        module_node: ast.Module = ast.parse(source_code)
    except SyntaxError:
        logger.exception("find_preexisting_objects - Syntax error while parsing code")
        return preexisting_objects
    for node in module_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            preexisting_objects.add((node.name, ()))
        elif isinstance(node, ast.ClassDef):
            preexisting_objects.add((node.name, ()))
            for cnode in node.body:
                if isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    preexisting_objects.add((cnode.name, (FunctionParent(node.name, "ClassDef"),)))
    return preexisting_objects
