from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import site
import sys
import sysconfig
from pathlib import Path

from pydantic import BaseModel, field_validator


class ImportedModuleAnalysis(BaseModel, frozen=True):
    name: str  # TODO Crosshair: Validate that this is the basename of full_name.
    origin: str  # TODO Crosshair: Make this an enum. Validate what the prefix of file_path must be depending on origin.
    full_name: str  # TODO Crosshair: Validate that if file_path exists, it can transform into its suffix.
    file_path: Path | None  # TODO Crosshair: Validate that it transforms into full_name, can be None only for std lib.

    # TODO CROSSHAIR Add clone of libcst path to qualified name and package function, add package info to model.

    @field_validator("name")
    @classmethod
    def name_is_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError("must be an identifier")
        return v

    # TODO CROSSHAIR Make this into a standalone function.
    @field_validator("full_name")
    @classmethod
    def full_name_is_dotted_identifier(cls, v: str) -> str:
        if any(not s or not s.isidentifier() for s in v.split(".")):
            raise ValueError("must be a dotted identifier")
        return v

    @field_validator("file_path")
    @classmethod
    def file_path_exists(cls, v: Path | None) -> Path | None:
        if v and not v.exists():
            raise ValueError("must be an existing path")
        return v


def resolve_module_name(module: str | None, level: int, name: str, current_module_name: str) -> str | None:
    if level == 0:
        if module:
            return module  # Absolute import, return module name
        return name  # Edge case
    current_module_parts = current_module_name.split(".")
    if level > len(current_module_parts):
        return None  # Invalid relative import
    base_parts = current_module_parts[:-level]
    if module:
        base_parts.extend(module.split("."))
    if name:
        base_parts.append(name)
    return ".".join(base_parts)


def collect_imports(node: ast.AST, module_name: str) -> set[str]:
    imports: set[str] = set()

    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        module = node.module
        level = node.level
        if module is None and level > 0:
            # Relative import with names
            for alias in node.names:
                name = alias.name
                resolved_module = resolve_module_name(module, level, name, module_name)
                if resolved_module:
                    imports.add(resolved_module)
        else:
            # Absolute import, collect the module without imported names
            resolved_module = resolve_module_name(module, level, "", module_name)
            if resolved_module:
                imports.add(resolved_module)
    else:
        for child in ast.iter_child_nodes(node):
            imports.update(collect_imports(child, module_name))

    return imports


def categorize_module(module_name: str, project_root: Path, module_file_path: Path) -> ImportedModuleAnalysis | None:
    if not module_name:
        return None

    # Default module search paths
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ImportError):
        spec = None

    if spec is None:
        # Internal modules
        custom_path_list = [str(module_file_path.parent), str(project_root), *sys.path.copy()]

        try:
            spec = importlib.machinery.PathFinder.find_spec(module_name, path=custom_path_list)
        except (ModuleNotFoundError, ImportError):
            spec = None

    if spec is None:
        # Module not found
        origin = "unknown"
        file_path = None
    else:
        spec_origin = spec.origin
        if spec_origin is None or spec_origin in ("built-in", "frozen"):
            origin = "standard library"
            file_path = None
        else:
            file_path_path: Path = Path(spec_origin).resolve()
            stdlib_paths: list[Path] = [
                Path(p).resolve() for key, p in sysconfig.get_paths().items() if "stdlib" in key or key == "purelib"
            ]
            site_packages_paths: list[Path] = [Path(p).resolve() for p in site.getsitepackages()]

            if user_site_packages := site.getusersitepackages():
                site_packages_paths.append(Path(user_site_packages).resolve())

            # Determine the origin based on the file path
            if any(file_path_path.is_relative_to(p) for p in stdlib_paths):
                origin = "standard library"
            elif any(file_path_path.is_relative_to(p) for p in site_packages_paths):
                origin = "third party"
            elif file_path_path.is_relative_to(project_root.resolve()):
                origin = "internal"
            else:
                origin = "unknown"

            file_path = file_path_path

    return ImportedModuleAnalysis(
        name=module_name.split(".")[-1], origin=origin, full_name=module_name, file_path=file_path
    )


def analyze_imported_modules(code_str: str, module_file_path: Path, project_root: Path) -> list[ImportedModuleAnalysis]:
    rel_parts: list[str] = list(module_file_path.resolve().relative_to(project_root.resolve()).with_suffix("").parts)
    if rel_parts and rel_parts[-1] == "__init__":
        rel_parts = rel_parts[:-1]

    imported_modules = []
    for module in collect_imports(ast.parse(code_str), ".".join(rel_parts)):
        info = categorize_module(module, project_root, module_file_path)
        if info and info not in imported_modules:
            imported_modules.append(info)

    return imported_modules
