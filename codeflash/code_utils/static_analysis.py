from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from codeflash.models.models import FunctionParent


ObjectDefT = TypeVar("ObjectDefT", ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class ImportedInternalModuleAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    full_name: str
    file_path: Path

    @field_validator("name")
    @classmethod
    def name_is_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            msg = "must be an identifier"
            raise ValueError(msg)
        return v

    @field_validator("full_name")
    @classmethod
    def full_name_is_dotted_identifier(cls, v: str) -> str:
        if any(not s or not s.isidentifier() for s in v.split(".")):
            msg = "must be a dotted identifier"
            raise ValueError(msg)
        return v

    @field_validator("file_path")
    @classmethod
    def file_path_exists(cls, v: Path | None) -> Path | None:
        if v and not v.exists():
            msg = "must be an existing path"
            raise ValueError(msg)
        return v


class FunctionKind(Enum):
    FUNCTION = 0
    STATIC_METHOD = 1
    CLASS_METHOD = 2
    INSTANCE_METHOD = 3


def parse_imports(code: str) -> list[ast.Import | ast.ImportFrom]:
    return [node for node in ast.walk(ast.parse(code)) if isinstance(node, (ast.Import, ast.ImportFrom))]


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


def get_module_full_name(node: ast.Import | ast.ImportFrom, current_module: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base_module = resolve_relative_name(node.module, node.level, current_module)
    if base_module is None:
        return []
    if node.module is None and node.level > 0:
        return [f"{base_module}.{alias.name}" for alias in node.names]
    return [base_module]


def is_internal_module(module_name: str, project_root: Path) -> bool:
    module_path = module_name.replace(".", "/")
    possible_paths = [project_root / f"{module_path}.py", project_root / module_path / "__init__.py"]
    return any(path.exists() for path in possible_paths)


def get_module_file_path(module_name: str, project_root: Path) -> Path | None:
    module_path = module_name.replace(".", "/")
    possible_paths = [project_root / f"{module_path}.py", project_root / module_path / "__init__.py"]
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    return None


def analyze_imported_modules(
    code_str: str, module_file_path: Path, project_root: Path
) -> list[ImportedInternalModuleAnalysis]:
    """Statically finds and analyzes all imported internal modules."""
    module_rel_path = module_file_path.relative_to(project_root).with_suffix("")
    current_module = ".".join(module_rel_path.parts)
    imports = parse_imports(code_str)
    module_names: set[str] = set()
    for node in imports:
        module_names.update(get_module_full_name(node, current_module))
    internal_modules = {module_name for module_name in module_names if is_internal_module(module_name, project_root)}
    return [
        ImportedInternalModuleAnalysis(name=str(mod_name).split(".")[-1], full_name=mod_name, file_path=file_path)
        for mod_name in internal_modules
        if (file_path := get_module_file_path(mod_name, project_root)) is not None
    ]


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
) -> ast.FunctionDef | None:
    if not parents:
        return get_first_top_level_object_def_ast(function_name, ast.FunctionDef, node)
    if parents[0].type == "ClassDef" and (
        class_node := get_first_top_level_object_def_ast(parents[0].name, ast.ClassDef, node)
    ):
        return get_first_top_level_object_def_ast(function_name, ast.FunctionDef, class_node)
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
