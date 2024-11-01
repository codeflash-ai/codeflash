from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, field_validator


class ImportedInternalModuleAnalysis(BaseModel, frozen=True):
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


def parse_imports(code: str) -> Iterator[ast.AST]:
    return (node for node in ast.walk(ast.parse(code)) if isinstance(node, (ast.Import, ast.ImportFrom)))


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


def get_module_full_name(node: ast.Import | ast.ImportFrom, current_module: str) -> Iterator[str]:
    if isinstance(node, ast.Import):
        return (alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        base_module = resolve_relative_name(node.module, node.level, current_module)
        if base_module is None:
            return iter(())
        if node.module is None and node.level > 0:
            # Relative import with no module specified, e.g., from . import mymodule
            return (f"{base_module}.{alias.name}" for alias in node.names)
        # For absolute imports or relative imports with module specified
        return iter([base_module])
    return iter(())


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


def analyze_imported_internal_modules(
    code_str: str, module_file_path: Path, project_root: Path
) -> list[ImportedInternalModuleAnalysis]:
    """Analyzes a Python module's code to find all imported internal modules."""
    module_rel_path = module_file_path.relative_to(project_root).with_suffix("")
    current_module = ".".join(module_rel_path.parts)

    imports = parse_imports(code_str)
    module_names = set()
    for node in imports:
        module_names.update(get_module_full_name(node, current_module))

    internal_modules = filter(is_internal_module, (module_names, project_root))

    return [
        ImportedInternalModuleAnalysis(name=mod_name.split(".")[-1], full_name=mod_name, file_path=file_path)
        for mod_name in internal_modules
        if (file_path := get_module_file_path(mod_name, project_root)) is not None
    ]
