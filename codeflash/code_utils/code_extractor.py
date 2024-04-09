from __future__ import annotations

import ast
import logging
from collections import deque

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def get_code(
    functions_to_optimize: list[FunctionToOptimize],
) -> tuple[str | None, set[tuple[str, str]]]:
    """Return the code for a class or functions in a file."""
    file_path: str = functions_to_optimize[0].file_path
    class_skeleton: set[tuple[int, int | None]] = set()
    contextual_dunder_methods: set[tuple[str, str]] = set()
    target_code: str = ""

    def find_target(
        node_list: list[ast.stmt],
        name_parts: tuple[str, str] | tuple[str],
    ) -> ast.AST | None:
        target: (
            ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.Assign
            | ast.AnnAssign
            | None
        ) = None
        node: ast.stmt
        for node in node_list:
            if (
                # The many mypy issues will be fixed once this code moves to the backend,
                # using Type Guards as we move to 3.10+.
                # We will cover the Type Alias case on the backend since it's a 3.12 feature.
                (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == name_parts[0]
                )
                # The next two cases cover type aliases in pre-3.12 syntax, where only single assignment is allowed.
                or (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == name_parts[0]
                )
                or (
                    isinstance(node, ast.AnnAssign)
                    and hasattr(node.target, "id")
                    and node.target.id == name_parts[0]
                )
            ):
                target = node
                break

        if target is None or len(name_parts) == 1:
            return target

        if isinstance(target, ast.ClassDef):
            class_skeleton.add((target.lineno, target.lineno))
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

        return None

    with open(file_path, encoding="utf8") as file:
        source_code = file.read()
    try:
        module_node = ast.parse(source_code)
    except SyntaxError as e:
        logging.exception(f"get_code - Syntax error in code: {e}")
        return None, set()
    # Get the source code lines for the target node
    lines = source_code.splitlines(keepends=True)
    if len(functions_to_optimize[0].parents) == 1:
        if (
            functions_to_optimize[0].parents[0].type == "ClassDef"
        ):  # All functions_to_optimize functions are methods of the same class.
            qualified_name_parts_list: list[tuple[str, str] | tuple[str]] = [
                (fto.parents[0].name, fto.function_name) for fto in functions_to_optimize
            ]

        else:
            logging.error(
                f"Error: get_code does not support nesting function in functions: {functions_to_optimize[0].parents}",
            )
            return None, set()
    elif len(functions_to_optimize[0].parents) == 0:
        qualified_name_parts_list = [(functions_to_optimize[0].function_name,)]
    else:
        logging.error(
            "Error: get_code does not support more than one level of nesting for now. "
            f"Parents: {functions_to_optimize[0].parents}",
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
            target_code += "".join(
                lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno],
            )
        else:
            target_code += "".join(lines[target_node.lineno - 1 : target_node.end_lineno])
    class_list: list[tuple[int, int | None]] = sorted(class_skeleton)
    class_code = "".join(
        ["".join(lines[s_lineno - 1 : e_lineno]) for (s_lineno, e_lineno) in class_list],
    )
    return class_code + target_code, contextual_dunder_methods


def get_code_no_skeleton(file_path: str, target_name: str) -> str | None:
    """Return the code for a function in a file, irrespective of class skeleton."""
    with open(file_path, encoding="utf8") as file:
        source_code = file.read()

    try:
        module_node = ast.parse(source_code)
    except SyntaxError as e:
        logging.exception(f"get_code_no_skeleton - Syntax error in code: {e}")
        return None

    name_parts = target_name.split(".")
    target_node = None
    stack: deque[ast.AST] = deque([module_node])

    while stack:
        node = stack.pop()
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name_parts[-1]
        ):
            target_node = node
            break
        stack.extend(list(ast.iter_child_nodes(node)))

    if target_node is None:
        return None

    # Get the source code lines for the target node
    lines = source_code.splitlines(keepends=True)
    if hasattr(target_node, "decorator_list") and target_node.decorator_list:
        target_code = "".join(
            lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno],
        )
    else:
        target_code = "".join(lines[target_node.lineno - 1 : target_node.end_lineno])

    return target_code


def extract_code(
    functions_to_optimize: list[FunctionToOptimize],
) -> tuple[str | None, set[tuple[str, str]]]:
    edited_code, contextual_dunder_methods = get_code(functions_to_optimize)
    if edited_code is None:
        return None, set()
    try:
        compile(edited_code, "edited_code", "exec")
    except SyntaxError as e:
        logging.exception(
            f"extract_code - Syntax error in extracted optimization candidate code: {e}",
        )
        return None, set()
    return edited_code, contextual_dunder_methods
