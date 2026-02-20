from __future__ import annotations

import ast
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from codeflash.models.models import FunctionParent

ASTNodeT = TypeVar("ASTNodeT", bound=ast.AST)
ObjectDefT = TypeVar("ObjectDefT", ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class FunctionKind(Enum):
    FUNCTION = 0
    STATIC_METHOD = 1
    CLASS_METHOD = 2
    INSTANCE_METHOD = 3


def normalize_node(node: ASTNodeT) -> ASTNodeT:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
        node.body = node.body[1:]
    if hasattr(node, "body"):
        node.body = [normalize_node(n) for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return node


@lru_cache(maxsize=3)
def normalize_code(code: str) -> str:
    return ast.unparse(normalize_node(ast.parse(code)))


def get_attribute_parts(node: ast.expr) -> list[str] | None:
    """Get all parts of an attribute/name chain (e.g., ['np', 'array'] from np.array)."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            parts.reverse()
            return parts
    return None


def resolve_relative_name(module: str | None, level: int, current_module: str) -> str | None:
    if level == 0:
        return module
    current_parts = current_module.split(".")
    if level > len(current_parts):
        return None
    base_parts = current_parts[:-level]
    if module:
        base_parts.extend(module.split("."))
    return ".".join(base_parts)


def parse_imports(code: str) -> list[ast.Import | ast.ImportFrom]:
    return [node for node in ast.walk(ast.parse(code)) if isinstance(node, (ast.Import, ast.ImportFrom))]


def get_module_full_name(node: ast.Import | ast.ImportFrom, current_module: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base_module = resolve_relative_name(node.module, node.level, current_module)
    if base_module is None:
        return []
    if node.module is None and node.level > 0:
        return [f"{base_module}.{alias.name}" for alias in node.names]
    return [base_module]


def get_first_top_level_object_def_ast(
    object_name: str, object_type: type[ObjectDefT], node: ast.AST
) -> ObjectDefT | None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, object_type) and child.name == object_name:
            return child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if descendant := get_first_top_level_object_def_ast(object_name, object_type, child):
            return descendant
    return None


def get_first_top_level_function_or_method_ast(
    function_name: str, parents: list[FunctionParent], node: ast.AST
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if not parents:
        result = get_first_top_level_object_def_ast(function_name, ast.FunctionDef, node)
        if result is not None:
            return result
        return get_first_top_level_object_def_ast(function_name, ast.AsyncFunctionDef, node)
    if parents[0].type == "ClassDef" and (
        class_node := get_first_top_level_object_def_ast(parents[0].name, ast.ClassDef, node)
    ):
        result = get_first_top_level_object_def_ast(function_name, ast.FunctionDef, class_node)
        if result is not None:
            return result
        return get_first_top_level_object_def_ast(function_name, ast.AsyncFunctionDef, class_node)
    return None


def function_kind(node: ast.FunctionDef | ast.AsyncFunctionDef, parents: list[FunctionParent]) -> FunctionKind | None:
    if not parents or parents[0].type in ["FunctionDef", "AsyncFunctionDef"]:
        return FunctionKind.FUNCTION
    if parents[0].type == "ClassDef":
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "classmethod":
                    return FunctionKind.CLASS_METHOD
                if decorator.id == "staticmethod":
                    return FunctionKind.STATIC_METHOD
        return FunctionKind.INSTANCE_METHOD
    return None


def has_typed_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef, parents: list[FunctionParent]) -> bool:
    kind = function_kind(node, parents)
    if kind in [FunctionKind.FUNCTION, FunctionKind.STATIC_METHOD]:
        return all(arg.annotation for arg in node.args.args)
    if kind in [FunctionKind.CLASS_METHOD, FunctionKind.INSTANCE_METHOD]:
        return all(arg.annotation for arg in node.args.args[1:])
    return False
