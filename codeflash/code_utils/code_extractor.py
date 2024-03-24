import ast
import logging
from collections import deque
from typing import Optional

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def get_code(function_to_optimize: FunctionToOptimize) -> Optional[str]:
    """Returns the code for a class or function in a file."""
    file_path = function_to_optimize.file_path
    class_skeleton = []

    def find_target(node_list: list[ast.AST], name_parts: list[str]) -> Optional[ast.AST]:
        target_node = None
        for node in node_list:
            if (
                (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name == name_parts[0]
                )
                or (
                    isinstance(node, ast.Assign)
                    and hasattr(node.targets[0], "id")
                    and node.targets[0].id == name_parts[0]
                )
                or (
                    isinstance(node, ast.AnnAssign)
                    and hasattr(node.target, "id")
                    and node.target.id == name_parts[0]
                )
            ):
                target_node = node
                break

        if target_node is None or len(name_parts) == 1:
            return target_node

        if isinstance(target_node, ast.ClassDef):
            class_skeleton.append([node.lineno, node.lineno])
            cbody = target_node.body
            if isinstance(cbody[0], ast.expr):  # Is a docstring
                class_skeleton.append([cbody[0].lineno, cbody[0].end_lineno])
                cbody = cbody[1:]
            for cnode in cbody:
                if hasattr(cnode, "name") and cnode.name == "__init__":
                    class_skeleton.append([cnode.lineno, cnode.end_lineno])

            return find_target(target_node.body, name_parts[1:])

        return None

    with open(file_path, "r", encoding="utf8") as file:
        source_code = file.read()
    try:
        module_node = ast.parse(source_code)
    except SyntaxError as e:
        logging.error(f"get_code - Syntax error in code: {e}")
        return None
    if len(function_to_optimize.parents) == 1:
        if function_to_optimize.parents[0].type == "ClassDef":
            name_parts = [function_to_optimize.parents[0].name, function_to_optimize.function_name]
        else:
            logging.error(
                f"Error: get_code does not support nesting function in functions: {function_to_optimize.parents}"
            )
            return None
    elif len(function_to_optimize.parents) == 0:
        name_parts = [function_to_optimize.function_name]
    else:
        logging.error(
            f"Error: get_code does not support more than one level of nesting for now. Parents: {function_to_optimize.parents}"
        )
        return None
    target_node = find_target(module_node.body, name_parts)
    if target_node is None:
        return None

    # Get the source code lines for the target node
    lines = source_code.splitlines(keepends=True)
    class_code = "".join(
        ["".join(lines[s_lineno - 1 : e_lineno]) for (s_lineno, e_lineno) in class_skeleton]
    )
    if hasattr(target_node, "decorator_list") and len(target_node.decorator_list) > 0:
        target_code = "".join(
            lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno]
        )
    else:
        target_code = "".join(lines[target_node.lineno - 1 : target_node.end_lineno])

    return class_code + target_code


def get_code_no_skeleton(file_path: str, target_name: str) -> Optional[str]:
    """Returns the code for a function in a file. Irrespective of class skeleton."""

    with open(file_path, "r", encoding="utf8") as file:
        source_code = file.read()

    try:
        module_node = ast.parse(source_code)
    except SyntaxError as e:
        logging.error(f"get_code_no_skeleton - Syntax error in code: {e}")
        return None

    name_parts = target_name.split(".")
    target_node = None
    stack = deque([module_node])

    while stack:
        node = stack.pop()
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name_parts[-1]
        ):
            target_node = node
            break
        stack.extend(ast.iter_child_nodes(node))

    if target_node is None:
        return None

    # Get the source code lines for the target node
    lines = source_code.splitlines(keepends=True)
    if hasattr(target_node, "decorator_list") and target_node.decorator_list:
        target_code = "".join(
            lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno]
        )
    else:
        target_code = "".join(lines[target_node.lineno - 1 : target_node.end_lineno])

    return target_code
