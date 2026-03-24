from __future__ import annotations

import ast
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.models.models import CodeStringsMarkdown


def parse_and_collect_imports(code_context: CodeStringsMarkdown) -> tuple[ast.Module, dict[str, str]] | None:
    all_code = "\n".join(cs.code for cs in code_context.code_strings)
    try:
        tree = ast.parse(all_code)
    except SyntaxError:
        return None
    imported_names: dict[str, str] = {}

    # Directly iterate over the module body and nested structures instead of ast.walk
    # This avoids traversing every single node in the tree
    def collect_imports(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        imported_name = alias.asname if alias.asname else alias.name
                        imported_names[imported_name] = node.module
            # Recursively check nested structures (function defs, class defs, if statements, etc.)
            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.ExceptHandler,
                ),
            ):
                if hasattr(node, "body"):
                    collect_imports(node.body)
                if hasattr(node, "orelse"):
                    collect_imports(node.orelse)  # type: ignore[arg-type]
                if hasattr(node, "finalbody"):
                    collect_imports(node.finalbody)  # type: ignore[arg-type]
                if hasattr(node, "handlers"):
                    for handler in node.handlers:  # type: ignore[attr-defined]
                        collect_imports(handler.body)
            # Handle match/case statements (Python 3.10+)
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                for case in node.cases:  # type: ignore[attr-defined]
                    collect_imports(case.body)

    collect_imports(tree.body)
    return tree, imported_names


def collect_existing_class_names(tree: ast.Module) -> set[str]:
    class_names = set()
    stack = list(tree.body)

    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            stack.extend(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.body)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            stack.extend(node.body)
            if hasattr(node, "orelse"):
                stack.extend(node.orelse)  # type: ignore[arg-type]
        elif isinstance(node, ast.Try):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
            for handler in node.handlers:
                stack.extend(handler.body)

    return class_names


BUILTIN_AND_TYPING_NAMES = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "type",
        "object",
        "None",
        "NoneType",
        "Ellipsis",
        "NotImplemented",
        "memoryview",
        "range",
        "slice",
        "property",
        "classmethod",
        "staticmethod",
        "super",
        "Optional",
        "Union",
        "Any",
        "List",
        "Dict",
        "Set",
        "FrozenSet",
        "Tuple",
        "Type",
        "Callable",
        "Iterator",
        "Generator",
        "Coroutine",
        "AsyncGenerator",
        "AsyncIterator",
        "Iterable",
        "AsyncIterable",
        "Sequence",
        "MutableSequence",
        "Mapping",
        "MutableMapping",
        "Collection",
        "Awaitable",
        "Literal",
        "Final",
        "ClassVar",
        "TypeVar",
        "TypeAlias",
        "ParamSpec",
        "Concatenate",
        "Annotated",
        "TypeGuard",
        "Self",
        "Unpack",
        "TypeVarTuple",
        "Never",
        "NoReturn",
        "SupportsInt",
        "SupportsFloat",
        "SupportsComplex",
        "SupportsBytes",
        "SupportsAbs",
        "SupportsRound",
        "IO",
        "TextIO",
        "BinaryIO",
        "Pattern",
        "Match",
    }
)


def collect_type_names_from_annotation(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Subscript):
        names = collect_type_names_from_annotation(node.value)
        names |= collect_type_names_from_annotation(node.slice)
        return names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return collect_type_names_from_annotation(node.left) | collect_type_names_from_annotation(node.right)
    if isinstance(node, ast.Tuple):
        names = set[str]()
        for elt in node.elts:
            names |= collect_type_names_from_annotation(elt)
        return names
    return set()


MAX_RAW_PROJECT_CLASS_BODY_ITEMS = 8
MAX_RAW_PROJECT_CLASS_LINES = 40


def get_expr_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None

    parts: list[str] = []
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            continue
        if isinstance(current, ast.Call):
            current = current.func
            continue
        if isinstance(current, ast.Name):
            base_name = current.id
        else:
            base_name = None
        break

    if not parts:
        return base_name

    parts.reverse()
    if base_name is not None:
        parts.insert(0, base_name)
    return ".".join(parts)


def collect_import_aliases(module_tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname if alias.asname else alias.name.split(".")[0]
                aliases[bound_name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bound_name = alias.asname if alias.asname else alias.name
                aliases[bound_name] = f"{node.module}.{alias.name}"
    return aliases


def find_class_node_by_name(class_name: str, module_tree: ast.Module) -> ast.ClassDef | None:
    stack: list[ast.AST] = [module_tree]
    while stack:
        node = stack.pop()
        body = getattr(node, "body", None)
        if body:
            for item in body:
                if isinstance(item, ast.ClassDef):
                    if item.name == class_name:
                        return item
                    stack.append(item)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    stack.append(item)
    return None


def expr_matches_name(node: ast.AST | None, import_aliases: dict[str, str], suffix: str) -> bool:
    expr_name = get_expr_name(node)
    if expr_name is None:
        return False

    suffix_dot = "." + suffix
    if expr_name == suffix or expr_name.endswith(suffix_dot):
        return True
    resolved_name = import_aliases.get(expr_name)
    return resolved_name is not None and (resolved_name == suffix or resolved_name.endswith(suffix_dot))


def get_node_source(node: ast.AST | None, module_source: str, fallback: str = "...") -> str:
    if node is None:
        return fallback
    source_segment = ast.get_source_segment(module_source, node)
    if source_segment is not None:
        return source_segment
    try:
        return ast.unparse(node)
    except Exception:
        return fallback


def bool_literal(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def is_namedtuple_class(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> bool:
    for base in class_node.bases:  # noqa: SIM110
        if expr_matches_name(base, import_aliases, "NamedTuple"):
            return True
    return False


def get_dataclass_config(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> tuple[bool, bool, bool]:
    for decorator in class_node.decorator_list:
        if not expr_matches_name(decorator, import_aliases, "dataclass"):
            continue
        init_enabled = True
        kw_only = False
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                literal_value = bool_literal(keyword.value)
                if literal_value is None:
                    continue
                if keyword.arg == "init":
                    init_enabled = literal_value
                elif keyword.arg == "kw_only":
                    kw_only = literal_value
        return True, init_enabled, kw_only
    return False, False, False


def is_classvar_annotation(annotation: ast.expr, import_aliases: dict[str, str]) -> bool:
    annotation_root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    return expr_matches_name(annotation_root, import_aliases, "ClassVar")


def is_project_subpath(module_path: Path, project_root_path: Path) -> bool:
    return str(module_path.resolve()).startswith(str(project_root_path.resolve()) + os.sep)


def get_class_start_line(class_node: ast.ClassDef) -> int:
    if class_node.decorator_list:
        return min(d.lineno for d in class_node.decorator_list)
    return class_node.lineno


def class_has_explicit_init(class_node: ast.ClassDef) -> bool:
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
            return True
    return False
