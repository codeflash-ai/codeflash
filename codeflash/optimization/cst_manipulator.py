from __future__ import annotations

import libcst as cst


def is_dunder_method(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Returns the section attribute names (e.g., body, orelse) for a given node if they exist."""
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


def prune_cst_for_read_writable_code(
    node: cst.CSTNode, target_functions: set[str], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node and its children to keep nodes that lead to target functions."""
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        if qualified_name in target_functions:
            return node, True
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")
        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value
        new_body = []
        found_target = False

        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                qualified_name = f"{class_prefix}.{stmt.name.value}"
                if qualified_name in target_functions:
                    new_body.append(stmt)
                    found_target = True

        # If no target functions found, remove the class entirely
        if not new_body:
            return None, False

        return node.with_changes(body=cst.IndentedBlock(body=new_body)), found_target

    # For other nodes, we preserve them only if they contain target functions in their children.
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_read_writable_code(child, target_functions, prefix)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_writable_code(original_content, target_functions, prefix)
            if found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered

    if not found_any_target:
        return None, False

    return (node.with_changes(**updates) if updates else node), True


def get_read_writable_code(code: str, target_functions: set[str]) -> str:
    """Creates a read-writable code string by parsing and filtering the code to keep only
    target functions and the minimal surrounding structure.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_writable_code(module, target_functions)
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""


def prune_cst_for_read_only_code(
    node: cst.CSTNode, target_functions: set[str], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for read-only context:

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        # If it's a target function, remove it but mark found_target = True
        if qualified_name in target_functions:
            return None, True
        # Keep only dunder methods
        if is_dunder_method(node.name.value):
            return node, False
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")

        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # First pass: detect if there is a target function in the class
        found_in_class = False
        new_body = []
        for stmt in node.body.body:
            filtered, found_target = prune_cst_for_read_only_code(stmt, target_functions, class_prefix)
            found_in_class |= found_target

            if isinstance(filtered, cst.FunctionDef):
                # Check if it's a target or non-dunder method
                qname = f"{class_prefix}.{filtered.name.value}"
                if qname in target_functions or not is_dunder_method(filtered.name.value):
                    continue
            if filtered:
                new_body.append(filtered)

        if not found_in_class:
            return None, False

        return node.with_changes(body=cst.IndentedBlock(body=new_body)) if new_body else None, True

    # For other nodes, keep the node and recursively filter children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_read_only_code(child, target_functions, prefix)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_only_code(original_content, target_functions, prefix)
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered

    if updates:
        return (node.with_changes(**updates), found_any_target)

    return node, found_any_target


def get_read_only_code(code: str, target_functions: set[str]) -> str:
    """Creates a read-only version of the code by parsing and filtering the code to keep only
    class contextual information, and other module scoped variables.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_only_code(module, target_functions)
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""
