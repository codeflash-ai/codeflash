import ast
import logging
from collections import deque
from typing import Optional, Union

from codeflash.discovery.functions_to_optimize import FunctionToOptimize


def get_code(function_to_optimize: FunctionToOptimize) -> Optional[str]:
    """Returns the code for a class or function in a file."""
    file_path: str = function_to_optimize.file_path
    class_skeleton: list[tuple[int, int]] = []

    def find_target(
        node_list: list[ast.stmt], name_parts: Union[tuple[str, str], tuple[str]]
    ) -> Optional[ast.AST]:
        target: Optional[
            Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign]
        ] = None
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign]

        for node in node_list:
            if (
                # The many mypy issues will be fixed once this code moves to the backend,
                # using Type Guards as we move to 3.10+.
                # We will cover the Type Alias case on the backend since it's a 3.12 feature.
                (
                    isinstance(node, Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef])
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
            class_skeleton.append((target.lineno, target.lineno))
            cbody = target.body
            if isinstance(cbody[0], ast.expr):  # Is a docstring
                class_skeleton.append((cbody[0].lineno, cbody[0].end_lineno))
                cbody = cbody[1:]
                cnode: Union[ast.FunctionDef, ast.AsyncFunctionDef]
            for cnode in cbody:
                # Collect all dunder methods.
                cnode_name: str
                if (
                    isinstance(cnode, Union[ast.FunctionDef, ast.AsyncFunctionDef])
                    and len(cnode_name := cnode.name) > 4
                    and cnode_name.isascii()
                    and cnode_name.startswith("__")
                    and cnode_name.endswith("__")
                ):
                    class_skeleton.append((cnode.lineno, cnode.end_lineno))

            return find_target(target.body, name_parts[1:])

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
            qualified_name_parts: Union[tuple[str, str], tuple[str]] = (
                function_to_optimize.parents[0].name,
                function_to_optimize.function_name,
            )
        else:
            logging.error(
                f"Error: get_code does not support nesting function in functions: {function_to_optimize.parents}"
            )
            return None
    elif len(function_to_optimize.parents) == 0:
        qualified_name_parts = (function_to_optimize.function_name,)
    else:
        logging.error(
            "Error: get_code does not support more than one level of nesting for now. "
            f"Parents: {function_to_optimize.parents}"
        )
        return None
    target_node = find_target(module_node.body, qualified_name_parts)
    if target_node is None:
        return None

    # Get the source code lines for the target node
    lines = source_code.splitlines(keepends=True)
    class_code = "".join(
        ["".join(lines[s_lineno - 1 : e_lineno]) for (s_lineno, e_lineno) in class_skeleton]
    )
    if (
        isinstance(target_node, Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef])
        and target_node.decorator_list
    ):
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
    stack: deque[ast.AST] = deque([module_node])

    while stack:
        node = stack.pop()
        if (
            isinstance(node, Union[ast.FunctionDef, ast.AsyncFunctionDef])
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
            lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno]
        )
    else:
        target_code = "".join(lines[target_node.lineno - 1 : target_node.end_lineno])

    return target_code
