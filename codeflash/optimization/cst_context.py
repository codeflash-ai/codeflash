from __future__ import annotations

import libcst as cst
from pydantic import BaseModel, Field


class CSTContextNode(BaseModel):
    """A node in the context tree, representing a single node in the CST. This context tree is a part of the main CST tree that contains nodes that lead to a target function.
    The corresponding cst_node is stored so that the cst can be rebuilt flexibly, based on whatever information is needed.
    In the future, this tree can be used as the code replacer, since we know where the target functions are located in the tree.
    """

    cst_node: cst.CSTNode | None
    children: dict[str, list[CSTContextNode] | CSTContextNode] = Field(default_factory=dict)
    is_target_function: bool = False
    target_functions: set[str] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True

    def add_child(self, section: str, child: CSTContextNode):
        """Add a child node to the specified section. sections are either body, orelse, finalbody, or handlers."""
        original_section = getattr(self.cst_node, section, None)
        if isinstance(original_section, (list, tuple)):
            if section not in self.children:
                self.children[section] = []
            self.children[section].append(child)
        else:
            self.children[section] = child


def build_context_tree(context_node: CSTContextNode, prefix: str = "") -> bool:
    """Recursively builds a context tree from a CST, tracking target functions and their containing structures.

    Args:
        context_node: Current node in the context tree
        prefix: Prefix to add to the target function names to create qualified names

    Returns:
        bool: True if a target function was found in this branch

    """

    def process_node(node: cst.CSTNode, section: str) -> bool:
        if isinstance(node, cst.ClassDef):
            if prefix:  # Don't go into nested classes
                return False
            class_node = CSTContextNode(cst_node=node, target_functions=context_node.target_functions)
            if build_context_tree(class_node, f"{prefix}.{node.name.value}" if prefix else node.name.value):
                context_node.add_child(section, class_node)
                return True

        elif isinstance(node, cst.FunctionDef):
            qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
            if qualified_name in context_node.target_functions:
                func_node = CSTContextNode(
                    cst_node=node, is_target_function=True, target_functions=context_node.target_functions
                )
                context_node.add_child(section, func_node)
                return True

        elif isinstance(node, cst.CSTNode):
            other_node = CSTContextNode(cst_node=node, target_functions=context_node.target_functions)
            if build_context_tree(other_node, prefix):
                context_node.add_child(section, other_node)
                return True

        return False

    has_target = False
    node = context_node.cst_node

    # Check each section directly
    for section_name in ["body", "orelse", "finalbody", "handlers"]:
        section_content = getattr(node, section_name, None)
        if section_content is not None:
            if isinstance(section_content, (list, tuple)):
                has_target_list = [process_node(section_node, section_name) for section_node in section_content]
                has_target |= any(has_target_list)
            else:
                has_target |= process_node(section_content, section_name)

    return has_target


def find_containing_classes(code: str, target_functions: set[str]) -> CSTContextNode:
    """Parse the code and find all class definitions containing the target functions."""
    root = CSTContextNode(cst_node=cst.parse_module(code), target_functions=target_functions)
    if not build_context_tree(root):
        raise ValueError("No target functions found in the provided code")
    return root


def create_read_write_context(context_node: CSTContextNode) -> str:
    """Rebuilds a CST tree to create our read-write context"""

    def rebuild_node(node: CSTContextNode) -> cst.CSTNode:
        if node.is_target_function:
            return node.cst_node

        updates = {}
        for section_name, children in node.children.items():
            if isinstance(children, list):
                updates[section_name] = [rebuild_node(child) for child in children]
            else:
                updates[section_name] = rebuild_node(children)

        return node.cst_node.with_changes(**updates)

    if not isinstance(context_node.cst_node, cst.Module):
        raise ValueError("Root of context tree must be a Module node")

    rebuilt_module = rebuild_node(context_node)
    return rebuilt_module.code


def create_read_only_context(context_node: CSTContextNode) -> str:
    """Creates a read-only version of the context tree where:
    - Global variables and typing information are preserved
    - Class definitions preserve all information (variables, docstrings, etc.)
    - Only dunder methods are preserved
    - All other methods (including target functions) are removed completely
    """

    def rebuild_non_context_node(node: cst.CSTNode) -> cst.CSTNode | None:
        """Recursively rebuild non-context nodes, preserving all statements except class and function definitions."""
        if isinstance(node, (cst.ClassDef, cst.FunctionDef, cst.Import, cst.ImportFrom)):
            return None

        updates = {}
        for section_name in ["body", "orelse", "finalbody", "handlers"]:
            section_content = getattr(node, section_name, None)
            if section_content is not None:
                if isinstance(section_content, (list, tuple)):
                    rebuilt_children = []
                    for child in section_content:
                        rebuilt_child = rebuild_non_context_node(child)
                        if rebuilt_child is not None:
                            rebuilt_children.append(rebuilt_child)
                    if rebuilt_children:
                        updates[section_name] = rebuilt_children
                else:
                    rebuilt_child = rebuild_non_context_node(section_content)
                    if rebuilt_child is not None:
                        updates[section_name] = rebuilt_child

        if not updates:
            return node

        return node.with_changes(**updates)

    def rebuild_node(node: CSTContextNode) -> cst.CSTNode | None:
        # Remove target functions completely
        if node.is_target_function:
            return None

        # For regular functions, remove them
        if isinstance(node.cst_node, cst.FunctionDef):
            return None

        # For class definitions, preserve structure but remove non-dunder methods
        if isinstance(node.cst_node, cst.ClassDef):
            return rebuild_class(node)

        updates = {}
        for section_name in ["body", "orelse", "finalbody", "handlers"]:
            section_content = getattr(node.cst_node, section_name, None)
            if section_content is not None:
                context_children = node.children.get(section_name, [])
                context_children = [context_children] if not isinstance(context_children, list) else context_children

                if isinstance(section_content, (list, tuple)):
                    rebuilt_children = []
                    # Create a map of context children positions
                    context_positions = {id(child.cst_node): i for i, child in enumerate(context_children)}

                    for i, child_node in enumerate(section_content):
                        # If this node is in context, use rebuild_node
                        if id(child_node) in context_positions:
                            context_idx = context_positions[id(child_node)]
                            rebuilt_child = rebuild_node(context_children[context_idx])
                        else:
                            # Otherwise use rebuild_non_context_node
                            rebuilt_child = rebuild_non_context_node(child_node)

                        if rebuilt_child is not None:
                            rebuilt_children.append(rebuilt_child)

                    if rebuilt_children:
                        updates[section_name] = rebuilt_children
                else:
                    # Single node case
                    if context_children:  # If we have a context child
                        rebuilt_child = rebuild_node(context_children[0])
                    else:
                        rebuilt_child = rebuild_non_context_node(section_content)

                    if rebuilt_child is not None:
                        updates[section_name] = rebuilt_child

        if not updates:
            return None

        return node.cst_node.with_changes(**updates)

    def is_dunder_method(func_node: cst.FunctionDef) -> bool:
        """Check if a function is a dunder method."""
        name = func_node.name.value
        return name.startswith("__") and name.endswith("__")

    def rebuild_class(node: CSTContextNode) -> cst.ClassDef:
        """Rebuilds a class definition, preserving only structure, variables, and dunder methods."""
        class_node = node.cst_node
        new_body = []

        if isinstance(class_node.body, cst.IndentedBlock):
            body_statements = class_node.body.body
        else:
            body_statements = [class_node.body]

        for stmt in body_statements:
            if isinstance(stmt, cst.FunctionDef):
                if is_dunder_method(stmt):
                    if f"{class_node.name.value}.{stmt.name.value}" in node.target_functions:
                        # target function is a dunder method already shown in read-write context
                        continue
                    # Preserve only dunder methods
                    new_body.append(stmt)
            else:
                # Keep all other class contents (variables, docstrings, etc.)
                new_body.append(stmt)

        if not new_body:
            return None
        return class_node.with_changes(body=cst.IndentedBlock(new_body))

    if not isinstance(context_node.cst_node, cst.Module):
        raise ValueError("Root of context tree must be a Module node")

    rebuilt_module = rebuild_node(context_node)
    if rebuilt_module is None:
        return ""
    return rebuilt_module.code


def print_tree(node: CSTContextNode, level: int = 0):
    """Helper function to visualize the full CST node structure recursively"""
    indent = "  " * level
    print(f"\n{indent}CSTContextNode:")
    print(f"{indent}  is_target_function: {node.is_target_function}")
    print(f"{indent}  cst_node type: {type(node.cst_node)}")
    print(f"{indent}  children:")

    for section_name, children in node.children.items():
        print(f"{indent}    {section_name}:")
        if isinstance(children, list):
            for child in children:
                print_tree(child, level + 3)
        else:
            print_tree(children, level + 3)
