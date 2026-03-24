from __future__ import annotations

import ast
import logging
import os
from typing import TYPE_CHECKING

from codeflash_python.context.ast_helpers import (
    MAX_RAW_PROJECT_CLASS_BODY_ITEMS,
    MAX_RAW_PROJECT_CLASS_LINES,
    bool_literal,
    collect_existing_class_names,
    collect_import_aliases,
    collect_type_names_from_annotation,
    expr_matches_name,
    find_class_node_by_name,
    get_class_start_line,
    get_dataclass_config,
    get_expr_name,
    get_node_source,
    is_classvar_annotation,
    is_namedtuple_class,
    is_project_subpath,
    parse_and_collect_imports,
)
from codeflash_python.context.jedi_helpers import get_jedi_project
from codeflash_python.models.models import CodeString, CodeStringsMarkdown

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("codeflash_python")


def collect_synthetic_constructor_type_names(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> set[str]:
    is_dataclass, dataclass_init_enabled, _ = get_dataclass_config(class_node, import_aliases)
    if not is_namedtuple_class(class_node, import_aliases) and not is_dataclass:
        return set()
    if is_dataclass and not dataclass_init_enabled:
        return set()

    names = set[str]()
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name) or item.annotation is None:
            continue
        if is_classvar_annotation(item.annotation, import_aliases):
            continue

        include_in_init = True
        if isinstance(item.value, ast.Call) and expr_matches_name(item.value.func, import_aliases, "field"):
            for keyword in item.value.keywords:
                if keyword.arg != "init":
                    continue
                literal_value = bool_literal(keyword.value)
                if literal_value is not None:
                    include_in_init = literal_value
                break

        if include_in_init:
            names |= collect_type_names_from_annotation(item.annotation)

    return names


def extract_synthetic_init_parameters(
    class_node: ast.ClassDef, module_source: str, import_aliases: dict[str, str], *, kw_only_by_default: bool
) -> list[tuple[str, str, str | None, bool]]:
    parameters: list[tuple[str, str, str | None, bool]] = []
    for item in class_node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        if is_classvar_annotation(item.annotation, import_aliases):
            continue

        include_in_init = True
        kw_only = kw_only_by_default
        default_value: str | None = None
        if item.value is not None:
            if isinstance(item.value, ast.Call) and expr_matches_name(item.value.func, import_aliases, "field"):
                for keyword in item.value.keywords:
                    if keyword.arg == "init":
                        literal_value = bool_literal(keyword.value)
                        if literal_value is not None:
                            include_in_init = literal_value
                    elif keyword.arg == "kw_only":
                        literal_value = bool_literal(keyword.value)
                        if literal_value is not None:
                            kw_only = literal_value
                    elif keyword.arg == "default":
                        default_value = get_node_source(keyword.value, module_source)
                    elif keyword.arg == "default_factory":
                        default_value = "..."
            else:
                default_value = get_node_source(item.value, module_source)

        if not include_in_init:
            continue

        parameters.append(
            (item.target.id, get_node_source(item.annotation, module_source, "Any"), default_value, kw_only)
        )
    return parameters


def build_synthetic_init_stub(
    class_node: ast.ClassDef, module_source: str, import_aliases: dict[str, str]
) -> str | None:
    is_namedtuple = is_namedtuple_class(class_node, import_aliases)
    is_dataclass, dataclass_init_enabled, dataclass_kw_only = get_dataclass_config(class_node, import_aliases)
    if not is_namedtuple and not is_dataclass:
        return None
    if is_dataclass and not dataclass_init_enabled:
        return None

    parameters = extract_synthetic_init_parameters(
        class_node, module_source, import_aliases, kw_only_by_default=dataclass_kw_only
    )
    if not parameters:
        return None

    signature_parts = ["self"]
    inserted_kw_only_marker = False
    for param_name, annotation_source, default_value, kw_only in parameters:
        if kw_only and not inserted_kw_only_marker:
            signature_parts.append("*")
            inserted_kw_only_marker = True
        part = f"{param_name}: {annotation_source}"
        if default_value is not None:
            part += f" = {default_value}"
        signature_parts.append(part)

    signature = ", ".join(signature_parts)
    return f"    def __init__({signature}):\n        ..."


def extract_function_stub_snippet(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, module_lines: list[str]) -> str:
    start_line = min(d.lineno for d in fn_node.decorator_list) if fn_node.decorator_list else fn_node.lineno
    return "\n".join(module_lines[start_line - 1 : fn_node.end_lineno])


def extract_raw_class_context(class_node: ast.ClassDef, module_source: str, module_tree: ast.Module) -> str:
    class_source = "\n".join(module_source.splitlines()[get_class_start_line(class_node) - 1 : class_node.end_lineno])
    needed_imports = extract_imports_for_class(module_tree, class_node, module_source)
    if needed_imports:
        return f"{needed_imports}\n\n{class_source}"
    return class_source


def has_non_property_method_decorator(
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef, import_aliases: dict[str, str]
) -> bool:
    for decorator in fn_node.decorator_list:
        if expr_matches_name(decorator, import_aliases, "property"):
            continue
        decorator_name = get_expr_name(decorator)
        if decorator_name and decorator_name.endswith((".setter", ".deleter")):
            continue
        return True
    return False


def should_use_raw_project_class_context(class_node: ast.ClassDef, import_aliases: dict[str, str]) -> bool:
    if class_node.decorator_list:
        return True

    if is_namedtuple_class(class_node, import_aliases):
        return True
    is_dataclass, _, _ = get_dataclass_config(class_node, import_aliases)
    if is_dataclass:
        return True

    start_line = get_class_start_line(class_node)
    assert class_node.end_lineno is not None
    class_line_count = class_node.end_lineno - start_line + 1
    is_small = (
        class_line_count <= MAX_RAW_PROJECT_CLASS_LINES and len(class_node.body) <= MAX_RAW_PROJECT_CLASS_BODY_ITEMS
    )

    has_explicit_init = False

    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                has_explicit_init = True
                if is_small:
                    return True
            if has_non_property_method_decorator(item, import_aliases):
                return True
        elif isinstance(item, (ast.Assign, ast.AnnAssign)) and isinstance(item.value, ast.Call):
            return True

    return False


def extract_init_stub_from_class(class_name: str, module_source: str, module_tree: ast.Module) -> str | None:
    class_node = find_class_node_by_name(class_name, module_tree)

    if class_node is None:
        return None

    lines = module_source.splitlines()
    import_aliases = collect_import_aliases(module_tree)
    explicit_init_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    support_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name == "__init__":
                explicit_init_nodes.append(item)
                support_nodes.append(item)
                continue
            if item.name == "__post_init__":
                support_nodes.append(item)
                continue
            for d in item.decorator_list:
                if (isinstance(d, ast.Name) and d.id == "property") or (
                    isinstance(d, ast.Attribute) and d.attr == "property"
                ):
                    support_nodes.append(item)
                    break

    snippets: list[str] = []
    if not explicit_init_nodes:
        synthetic_init = build_synthetic_init_stub(class_node, module_source, import_aliases)
        if synthetic_init is not None:
            snippets.append(synthetic_init)
    for fn_node in support_nodes:
        snippets.append(extract_function_stub_snippet(fn_node, lines))

    if not snippets:
        return None

    return f"class {class_name}:\n" + "\n".join(snippets)


def get_module_source_and_tree(
    module_path: Path, module_cache: dict[Path, tuple[str, ast.Module]]
) -> tuple[str, ast.Module] | None:
    if module_path in module_cache:
        return module_cache[module_path]
    try:
        module_source = module_path.read_text(encoding="utf-8")
        module_tree = ast.parse(module_source)
    except Exception:
        return None
    module_cache[module_path] = (module_source, module_tree)
    return module_source, module_tree


def resolve_imported_class_reference(
    base_expr_name: str,
    current_module_tree: ast.Module,
    current_module_path: Path,
    project_root_path: Path,
    module_cache: dict[Path, tuple[str, ast.Module]],
) -> tuple[str, Path] | None:
    import jedi

    import_aliases = collect_import_aliases(current_module_tree)
    class_name = base_expr_name.rsplit(".", 1)[-1]
    if "." not in base_expr_name and find_class_node_by_name(class_name, current_module_tree) is not None:
        return class_name, current_module_path

    resolved_name = base_expr_name
    if base_expr_name in import_aliases:
        resolved_name = import_aliases[base_expr_name]
    elif "." in base_expr_name:
        head, tail = base_expr_name.split(".", 1)
        if head in import_aliases:
            resolved_name = f"{import_aliases[head]}.{tail}"

    if "." not in resolved_name:
        return None

    module_name, class_name = resolved_name.rsplit(".", 1)
    try:
        script_code = f"from {module_name} import {class_name}"
        script = jedi.Script(script_code, project=get_jedi_project(str(project_root_path)))
        definitions = script.goto(1, len(f"from {module_name} import ") + len(class_name), follow_imports=True)
    except Exception:
        return None

    if not definitions or definitions[0].module_path is None:
        return None
    module_path = definitions[0].module_path
    if not is_project_subpath(module_path, project_root_path):
        return None
    if get_module_source_and_tree(module_path, module_cache) is None:
        return None
    return class_name, module_path


def append_project_class_context(
    class_name: str,
    module_path: Path,
    project_root_path: Path,
    module_cache: dict[Path, tuple[str, ast.Module]],
    existing_class_names: set[str],
    emitted_classes: set[tuple[Path, str]],
    emitted_class_names: set[str],
    code_strings: list[CodeString],
) -> bool:
    module_result = get_module_source_and_tree(module_path, module_cache)
    if module_result is None:
        return False
    module_source, module_tree = module_result
    class_node = find_class_node_by_name(class_name, module_tree)
    if class_node is None:
        return False

    class_key = (module_path, class_name)
    if class_key in emitted_classes or class_name in existing_class_names:
        return True

    for base in class_node.bases:
        base_expr_name = get_expr_name(base)
        if base_expr_name is None:
            continue
        resolved = resolve_imported_class_reference(
            base_expr_name, module_tree, module_path, project_root_path, module_cache
        )
        if resolved is None:
            continue
        base_name, base_module_path = resolved
        if base_name in existing_class_names:
            continue
        append_project_class_context(
            base_name,
            base_module_path,
            project_root_path,
            module_cache,
            existing_class_names,
            emitted_classes,
            emitted_class_names,
            code_strings,
        )

    code_strings.append(
        CodeString(code=extract_raw_class_context(class_node, module_source, module_tree), file_path=module_path)
    )
    emitted_classes.add(class_key)
    emitted_class_names.add(class_name)
    return True


def resolve_instance_class_name(name: str, module_tree: ast.Module) -> str | None:
    for node in module_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Call):
                        func = value.func
                        if isinstance(func, ast.Name):
                            return func.id
                        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            return func.value.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            ann = node.annotation
            if isinstance(ann, ast.Name):
                return ann.id
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                return ann.value.id
    return None


def enrich_testgen_context(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    import jedi

    result = parse_and_collect_imports(code_context)
    if result is None:
        return CodeStringsMarkdown(code_strings=[])
    tree, imported_names = result

    if not imported_names:
        return CodeStringsMarkdown(code_strings=[])

    existing_classes = collect_existing_class_names(tree)

    code_strings: list[CodeString] = []
    emitted_class_names: set[str] = set()

    # --- Step 1: Project class definitions (jedi resolution + recursive base extraction) ---
    extracted_classes: set[tuple[Path, str]] = set()
    module_cache: dict[Path, tuple[str, ast.Module]] = {}

    def get_module_source_and_tree(module_path: Path) -> tuple[str, ast.Module] | None:
        if module_path in module_cache:
            return module_cache[module_path]
        try:
            module_source = module_path.read_text(encoding="utf-8")
            module_tree = ast.parse(module_source)
        except Exception:
            return None
        else:
            module_cache[module_path] = (module_source, module_tree)
            return module_source, module_tree

    def extract_class_and_bases(
        class_name: str, module_path: Path, module_source: str, module_tree: ast.Module
    ) -> None:
        if (module_path, class_name) in extracted_classes:
            return

        class_node = None
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            return

        for base in class_node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                continue

            if base_name and base_name not in existing_classes:
                extract_class_and_bases(base_name, module_path, module_source, module_tree)

        if (module_path, class_name) in extracted_classes:
            return

        lines = module_source.split("\n")
        start_line = class_node.lineno
        if class_node.decorator_list:
            start_line = min(d.lineno for d in class_node.decorator_list)
        class_source = "\n".join(lines[start_line - 1 : class_node.end_lineno])

        full_source = class_source

        code_strings.append(CodeString(code=full_source, file_path=module_path))
        extracted_classes.add((module_path, class_name))
        emitted_class_names.add(class_name)

    for name, module_name in imported_names.items():
        if name in existing_classes or module_name == "__future__":
            continue
        try:
            test_code = f"import {module_name}"
            script = jedi.Script(test_code, project=get_jedi_project(str(project_root_path)))
            completions = script.goto(1, len(test_code))

            if not completions:
                continue

            module_path = completions[0].module_path
            if not module_path:
                continue

            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = module_str.startswith(str(project_root_path.resolve()) + os.sep)
            is_third_party = "site-packages" in module_str
            if not is_project and not is_third_party:
                continue

            mod_result = get_module_source_and_tree(module_path)
            if mod_result is None:
                continue
            module_source, module_tree = mod_result

            if is_project:
                extract_class_and_bases(name, module_path, module_source, module_tree)
                if (module_path, name) not in extracted_classes:
                    resolved_class = resolve_instance_class_name(name, module_tree)
                    if resolved_class and resolved_class not in existing_classes:
                        extract_class_and_bases(resolved_class, module_path, module_source, module_tree)
            elif is_third_party:
                target_name = name
                if not any(isinstance(n, ast.ClassDef) and n.name == name for n in ast.walk(module_tree)):
                    resolved_class = resolve_instance_class_name(name, module_tree)
                    if resolved_class:
                        target_name = resolved_class
                if target_name not in emitted_class_names:
                    stub = extract_init_stub_from_class(target_name, module_source, module_tree)
                    if stub:
                        code_strings.append(CodeString(code=stub, file_path=module_path))
                        emitted_class_names.add(target_name)

        except Exception:
            logger.debug("Error extracting class definition for %s from %s", name, module_name)
            continue

    return CodeStringsMarkdown(code_strings=code_strings)


def extract_imports_for_class(module_tree: ast.Module, class_node: ast.ClassDef, module_source: str) -> str:
    """Extract import statements needed for a class definition.

    This extracts imports for base classes, decorators, and type annotations.
    """
    needed_names: set[str] = set()

    # Get base class names
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            needed_names.add(base.id)
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # For things like abc.ABC, we need the module name
            needed_names.add(base.value.id)

    # Get decorator names (e.g., dataclass, field)
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name):
            needed_names.add(decorator.id)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                needed_names.add(decorator.func.id)
            elif isinstance(decorator.func, ast.Attribute) and isinstance(decorator.func.value, ast.Name):
                needed_names.add(decorator.func.value.id)

    # Get type annotation names from class body (for dataclass fields)
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and item.annotation:
            collect_names_from_annotation(item.annotation, needed_names)
        # Also check for field() calls which are common in dataclasses
        elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
            if isinstance(item.value.func, ast.Name):
                needed_names.add(item.value.func.id)

    # Find imports that provide these names
    import_lines: list[str] = []
    source_lines = module_source.split("\n")
    added_imports: set[int] = set()  # Track line numbers to avoid duplicates

    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break

    return "\n".join(import_lines)


def collect_names_from_annotation(node: ast.expr, names: set[str]) -> None:
    """Recursively collect type annotation names from an AST node."""
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Subscript):
        collect_names_from_annotation(node.value, names)
        collect_names_from_annotation(node.slice, names)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            collect_names_from_annotation(elt, names)
    elif isinstance(node, ast.BinOp):  # For Union types with | syntax
        collect_names_from_annotation(node.left, names)
        collect_names_from_annotation(node.right, names)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        names.add(node.value.id)
