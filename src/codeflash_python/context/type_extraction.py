"""Parameter type constructor extraction and import analysis for class context."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.models.models import CodeString, CodeStringsMarkdown
from codeflash_python.code_utils.code_utils import path_belongs_to_site_packages
from codeflash_python.context.ast_helpers import (
    BUILTIN_AND_TYPING_NAMES,
    collect_import_aliases,
    collect_type_names_from_annotation,
    find_class_node_by_name,
    is_project_subpath,
)
from codeflash_python.context.class_extraction import (
    append_project_class_context,
    collect_synthetic_constructor_type_names,
    extract_init_stub_from_class,
    get_module_source_and_tree,
    should_use_raw_project_class_context,
)
from codeflash_python.context.jedi_helpers import get_jedi_project

if TYPE_CHECKING:
    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


def collect_type_names_from_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef, tree: ast.Module, class_name: str | None
) -> set[str]:
    type_names: set[str] = set()
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        type_names |= collect_type_names_from_annotation(arg.annotation)
    if func_node.args.vararg:
        type_names |= collect_type_names_from_annotation(func_node.args.vararg.annotation)
    if func_node.args.kwarg:
        type_names |= collect_type_names_from_annotation(func_node.args.kwarg.annotation)
    for body_node in ast.walk(func_node):
        if (
            isinstance(body_node, ast.Call)
            and isinstance(body_node.func, ast.Name)
            and body_node.func.id == "isinstance"
        ):
            if len(body_node.args) >= 2:
                second_arg = body_node.args[1]
                if isinstance(second_arg, ast.Name):
                    type_names.add(second_arg.id)
                elif isinstance(second_arg, ast.Tuple):
                    for elt in second_arg.elts:
                        if isinstance(elt, ast.Name):
                            type_names.add(elt.id)
        elif isinstance(body_node, ast.Compare):
            if (
                isinstance(body_node.left, ast.Call)
                and isinstance(body_node.left.func, ast.Name)
                and body_node.left.func.id == "type"
            ):
                for comparator in body_node.comparators:
                    if isinstance(comparator, ast.Name):
                        type_names.add(comparator.id)
    if class_name is not None:
        for top_node in ast.walk(tree):
            if isinstance(top_node, ast.ClassDef) and top_node.name == class_name:
                for base in top_node.bases:
                    if isinstance(base, ast.Name):
                        type_names.add(base.id)
                break
    return type_names


def build_import_from_map(tree: ast.Module) -> dict[str, str]:
    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                import_map[alias.asname if alias.asname else alias.name] = node.module
    return import_map


def extract_parameter_type_constructors(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, existing_class_names: set[str]
) -> CodeStringsMarkdown:
    import jedi

    try:
        source = function_to_optimize.file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return CodeStringsMarkdown(code_strings=[])

    func_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_to_optimize.function_name
        ):
            if function_to_optimize.starting_line is not None and node.lineno != function_to_optimize.starting_line:
                continue
            func_node = node
            break
    if func_node is None:
        return CodeStringsMarkdown(code_strings=[])

    type_names = collect_type_names_from_function(func_node, tree, function_to_optimize.class_name)
    type_names -= BUILTIN_AND_TYPING_NAMES
    type_names -= existing_class_names
    if not type_names:
        return CodeStringsMarkdown(code_strings=[])

    import_map = build_import_from_map(tree)

    code_strings: list[CodeString] = []
    module_cache: dict[Path, tuple[str, ast.Module]] = {}
    emitted_classes: set[tuple[Path, str]] = set()
    emitted_class_names: set[str] = set()

    def append_type_context(type_name: str, module_name: str, *, transitive: bool = False) -> None:
        try:
            script_code = f"from {module_name} import {type_name}"
            script = jedi.Script(script_code, project=get_jedi_project(str(project_root_path)))
            definitions = script.goto(1, len(f"from {module_name} import ") + len(type_name), follow_imports=True)
            if not definitions:
                return

            module_path = definitions[0].module_path
            if not module_path:
                return
            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = is_project_subpath(module_path, project_root_path)
            is_third_party = "site-packages" in module_str
            if transitive and not is_project and not is_third_party:
                return

            module_result = get_module_source_and_tree(module_path, module_cache)
            if module_result is None:
                return
            mod_source, mod_tree = module_result

            class_key = (module_path, type_name)
            if class_key in emitted_classes or type_name in existing_class_names:
                return

            class_node = find_class_node_by_name(type_name, mod_tree)
            if class_node is not None and is_project:
                import_aliases = collect_import_aliases(mod_tree)
                if should_use_raw_project_class_context(class_node, import_aliases):
                    if append_project_class_context(
                        type_name,
                        module_path,
                        project_root_path,
                        module_cache,
                        existing_class_names,
                        emitted_classes,
                        emitted_class_names,
                        code_strings,
                    ):
                        return

            stub = extract_init_stub_from_class(type_name, mod_source, mod_tree)
            if stub:
                code_strings.append(CodeString(code=stub, file_path=module_path))
                emitted_classes.add(class_key)
                emitted_class_names.add(type_name)
        except Exception:
            if transitive:
                logger.debug("Error extracting transitive constructor stub for %s from %s", type_name, module_name)
            else:
                logger.debug("Error extracting constructor stub for %s from %s", type_name, module_name)

    for type_name in sorted(type_names):
        module_name = import_map.get(type_name)
        if not module_name:
            continue
        append_type_context(type_name, module_name)

    # Transitive extraction (one level): for each extracted stub, find __init__ param types and extract their stubs
    transitive_import_map = dict(import_map)
    for _, cached_tree in module_cache.values():
        for name, module in build_import_from_map(cached_tree).items():
            transitive_import_map.setdefault(name, module)

    emitted_names = type_names | existing_class_names | emitted_class_names | BUILTIN_AND_TYPING_NAMES
    transitive_type_names: set[str] = set()
    for cs in code_strings:
        try:
            stub_tree = ast.parse(cs.code)
        except SyntaxError:
            continue
        import_aliases = collect_import_aliases(stub_tree)
        for stub_node in ast.walk(stub_tree):
            if isinstance(stub_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and stub_node.name in (
                "__init__",
                "__post_init__",
            ):
                for arg in stub_node.args.args + stub_node.args.posonlyargs + stub_node.args.kwonlyargs:
                    transitive_type_names |= collect_type_names_from_annotation(arg.annotation)
            elif isinstance(stub_node, ast.ClassDef):
                transitive_type_names |= collect_synthetic_constructor_type_names(stub_node, import_aliases)
    transitive_type_names -= emitted_names
    for type_name in sorted(transitive_type_names):
        module_name = transitive_import_map.get(type_name)
        if not module_name:
            continue
        append_type_context(type_name, module_name, transitive=True)

    return CodeStringsMarkdown(code_strings=code_strings)


def is_project_module_cached(module_name: str, project_root_path: Path, cache: dict[str, bool]) -> bool:
    cached = cache.get(module_name)
    if cached is not None:
        return cached
    is_project = is_project_module(module_name, project_root_path)
    cache[module_name] = is_project
    return is_project


def is_project_path(module_path: Path | None, project_root_path: Path) -> bool:
    if module_path is None:
        return False
    # site-packages must be checked first because .venv/site-packages is under project root
    if path_belongs_to_site_packages(module_path):
        return False
    try:
        module_path.resolve().relative_to(project_root_path.resolve())
        return True
    except ValueError:
        return False


def is_project_module(module_name: str, project_root_path: Path) -> bool:
    """Check if a module is part of the project (not external/stdlib)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    else:
        if spec is None or spec.origin is None:
            return False
        return is_project_path(Path(spec.origin), project_root_path)
