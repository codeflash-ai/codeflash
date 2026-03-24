from __future__ import annotations

from typing import TYPE_CHECKING

import libcst as cst

from codeflash_python.context.types import CodeContextType
from codeflash_python.context.unused_definition_remover import (
    collect_top_level_defs_with_usages,
    get_section_names,
    is_assignment_used,
    recurse_sections,
)

if TYPE_CHECKING:
    from codeflash_python.context.unused_definition_remover import UsageInfo


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    """Removes the docstring from an indented block if it exists."""
    if not isinstance(indented_block.body[0], cst.SimpleStatementLine):
        return indented_block
    first_stmt = indented_block.body[0].body[0]
    if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
        return indented_block.with_changes(body=indented_block.body[1:])
    return indented_block


def parse_code_and_prune_cst(
    code: str,
    code_context_type: CodeContextType,
    target_functions: set[str],
    helpers_of_helper_functions: set[str] | None = None,
    remove_docstrings: bool = False,
) -> cst.Module:
    """Parse and filter the code CST, returning the pruned Module."""
    if helpers_of_helper_functions is None:
        helpers_of_helper_functions = set()
    module = cst.parse_module(code)
    defs_with_usages = collect_top_level_defs_with_usages(module, target_functions | helpers_of_helper_functions)

    if code_context_type == CodeContextType.READ_WRITABLE:
        filtered_node, found_target = prune_cst(
            module, target_functions, defs_with_usages=defs_with_usages, keep_class_init=True
        )
    elif code_context_type == CodeContextType.READ_ONLY:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=False,
            include_dunder_methods=True,
        )
    elif code_context_type == CodeContextType.TESTGEN:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_dunder_methods=True,
            include_init_dunder=True,
        )
    elif code_context_type == CodeContextType.HASHING:
        filtered_node, found_target = prune_cst(
            module, target_functions, remove_docstrings=True, exclude_init_from_targets=True
        )
    else:
        raise ValueError(f"Unknown code_context_type: {code_context_type}")  # noqa: EM102

    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return filtered_node
    raise ValueError("Pruning produced no module")


def prune_cst(
    node: cst.CSTNode,
    target_functions: set[str],
    prefix: str = "",
    *,
    defs_with_usages: dict[str, UsageInfo] | None = None,
    helpers: set[str] | None = None,
    remove_docstrings: bool = False,
    include_target_in_output: bool = True,
    exclude_init_from_targets: bool = False,
    keep_class_init: bool = False,
    include_dunder_methods: bool = False,
    include_init_dunder: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    """Unified function to prune CST nodes based on various filtering criteria.

    Args:
        node: The CST node to filter
        target_functions: Set of qualified function names that are targets
        prefix: Current qualified name prefix (for class methods)
        defs_with_usages: Dict of definitions with usage info (for READ_WRITABLE mode)
        helpers: Set of helper function qualified names (for READ_ONLY/TESTGEN modes)
        remove_docstrings: Whether to remove docstrings from output
        include_target_in_output: Whether to include target functions in output
        exclude_init_from_targets: Whether to exclude __init__ from targets (HASHING mode)
        keep_class_init: Whether to keep __init__ methods in classes (READ_WRITABLE mode)
        include_dunder_methods: Whether to include dunder methods (READ_ONLY/TESTGEN modes)
        include_init_dunder: Whether to include __init__ in dunder methods

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # Check if it's a helper function (higher priority than target)
        if helpers and qualified_name in helpers:
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), True
            return node, True

        # Check if it's a target function
        if qualified_name in target_functions:
            # Handle exclude_init_from_targets for HASHING mode
            if exclude_init_from_targets and node.name.value == "__init__":
                return None, False

            if include_target_in_output:
                if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                    return node.with_changes(body=remove_docstring_from_body(node.body)), True
                return node, True
            return None, True

        # Handle class __init__ for READ_WRITABLE mode
        if keep_class_init and node.name.value == "__init__":
            return node, False

        # Handle dunder methods for READ_ONLY/TESTGEN modes
        if (
            include_dunder_methods
            and len(node.name.value) > 4
            and node.name.value.startswith("__")
            and node.name.value.endswith("__")
        ):
            if not include_init_dunder and node.name.value == "__init__":
                return None, False
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), False
            return node, False

        return None, False

    if isinstance(node, cst.ClassDef):
        if prefix:
            return None, False
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = node.name.value
        class_name = node.name.value

        # Handle dependency classes for READ_WRITABLE mode
        if defs_with_usages:
            # Check if this class contains any target functions
            has_target_functions = any(
                isinstance(stmt, cst.FunctionDef) and f"{class_prefix}.{stmt.name.value}" in target_functions
                for stmt in node.body.body
            )

            # If the class is used as a dependency (not containing target functions), keep it entirely
            if (
                not has_target_functions
                and class_name in defs_with_usages
                and defs_with_usages[class_name].used_by_qualified_function
            ):
                return node, True

        # Recursively filter each statement in the class body
        new_class_body: list[cst.CSTNode] = []
        found_in_class = False

        for stmt in node.body.body:
            filtered, found_target = prune_cst(
                stmt,
                target_functions,
                class_prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            )
            found_in_class |= found_target
            if filtered:
                new_class_body.append(filtered)

        if not found_in_class:
            return None, False

        # Apply docstring removal to class if needed
        if remove_docstrings and new_class_body:
            updated_body = node.body.with_changes(body=new_class_body)
            assert isinstance(updated_body, cst.IndentedBlock)
            return node.with_changes(body=remove_docstring_from_body(updated_body)), True

        return node.with_changes(body=node.body.with_changes(body=new_class_body)) if new_class_body else None, True

    # Handle assignments for READ_WRITABLE mode
    if defs_with_usages is not None:
        if isinstance(node, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
            if is_assignment_used(node, defs_with_usages):
                return node, True
            return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    if helpers is not None:
        return recurse_sections(
            node,
            section_names,
            lambda child: prune_cst(
                child,
                target_functions,
                prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            ),
            keep_non_target_children=True,
        )
    return recurse_sections(
        node,
        section_names,
        lambda child: prune_cst(
            child,
            target_functions,
            prefix,
            defs_with_usages=defs_with_usages,
            helpers=helpers,
            remove_docstrings=remove_docstrings,
            include_target_in_output=include_target_in_output,
            exclude_init_from_targets=exclude_init_from_targets,
            keep_class_init=keep_class_init,
            include_dunder_methods=include_dunder_methods,
            include_init_dunder=include_init_dunder,
        ),
    )
