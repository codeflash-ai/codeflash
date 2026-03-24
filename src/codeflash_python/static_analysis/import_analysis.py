from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, GatherImportsVisitor, RemoveImportsVisitor
from libcst.helpers import calculate_module_and_package

if TYPE_CHECKING:
    from pathlib import Path

    from libcst.helpers import ModuleNameAndPackage

    from codeflash.models.models import FunctionSource

logger = logging.getLogger("codeflash_python")


class DottedImportCollector(cst.CSTVisitor):
    """Collects all top-level imports from a Python module in normalized dotted format, including top-level conditional imports like `if TYPE_CHECKING:`.

    Examples
    --------
        import os                                                                  ==> "os"
        import dbt.adapters.factory                                                ==> "dbt.adapters.factory"
        from pathlib import Path                                                   ==> "pathlib.Path"
        from recce.adapter.base import BaseAdapter                                 ==> "recce.adapter.base.BaseAdapter"
        from typing import Any, List, Optional                                     ==> "typing.Any", "typing.List", "typing.Optional"
        from recce.util.lineage import ( build_column_key, filter_dependency_maps) ==> "recce.util.lineage.build_column_key", "recce.util.lineage.filter_dependency_maps"

    """

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.depth = 0  # top-level

    def get_full_dotted_name(self, expr: cst.BaseExpression) -> str:
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return f"{self.get_full_dotted_name(expr.value)}.{expr.attr.value}"
        return ""

    def collect_imports_from_block(self, block: cst.IndentedBlock | cst.Module) -> None:
        for statement in block.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for child in statement.body:
                    if isinstance(child, cst.Import):
                        for alias in child.names:
                            module = self.get_full_dotted_name(alias.name)
                            asname = alias.asname.name.value if alias.asname else alias.name.value  # type: ignore[attr-defined]
                            if isinstance(asname, cst.Attribute):
                                self.imports.add(module)
                            else:
                                self.imports.add(module if module == asname else f"{module}.{asname}")

                    elif isinstance(child, cst.ImportFrom):
                        if child.module is None:
                            continue
                        module = self.get_full_dotted_name(child.module)
                        if isinstance(child.names, cst.ImportStar):
                            continue
                        for alias in child.names:
                            if isinstance(alias, cst.ImportAlias):
                                name = alias.name.value
                                asname = alias.asname.name.value if alias.asname else name  # type: ignore[attr-defined]
                                self.imports.add(f"{module}.{asname}")

    def visit_Module(self, node: cst.Module) -> None:
        self.depth = 0
        self.collect_imports_from_block(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.depth -= 1

    def visit_If(self, node: cst.If) -> None:
        if self.depth == 0 and isinstance(node.body, (cst.IndentedBlock, cst.Module)):
            self.collect_imports_from_block(node.body)

    def visit_Try(self, node: cst.Try) -> None:
        if self.depth == 0 and isinstance(node.body, (cst.IndentedBlock, cst.Module)):
            self.collect_imports_from_block(node.body)


class FutureAliasedImportTransformer(cst.CSTTransformer):
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        import libcst.matchers as m

        if (
            (updated_node_module := updated_node.module)
            and updated_node_module.value == "__future__"
            and not isinstance(updated_node.names, cst.ImportStar)
            and all(m.matches(name, m.ImportAlias()) for name in updated_node.names)
        ):
            if names := [name for name in updated_node.names if name.asname is None]:
                return updated_node.with_changes(names=names)
            return cst.RemoveFromParent()
        return updated_node


def delete___future___aliased_imports(module_code: str) -> str:
    return cst.parse_module(module_code).visit(FutureAliasedImportTransformer()).code


def resolve_star_import(module_name: str, project_root: Path) -> set[str]:
    try:
        module_path = module_name.replace(".", "/")
        possible_paths = [project_root / f"{module_path}.py", project_root / f"{module_path}/__init__.py"]

        module_file = None
        for path in possible_paths:
            if path.exists():
                module_file = path
                break

        if module_file is None:
            logger.warning("Could not find module file for %s, skipping star import resolution", module_name)
            return set()

        with module_file.open(encoding="utf8") as f:
            module_code = f.read()

        tree = ast.parse(module_code)

        all_names = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
            ):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    all_names = []
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            all_names.append(elt.value)
                        elif isinstance(elt, ast.Str):  # type: ignore[deprecated]  # Python < 3.8 compatibility
                            all_names.append(elt.s)
                break

        if all_names is not None:
            return set(all_names)

        public_names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    public_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        public_names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                    public_names.add(node.target.id)
            elif isinstance(node, ast.Import) or (
                isinstance(node, ast.ImportFrom) and not any(alias.name == "*" for alias in node.names)
            ):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if not name.startswith("_"):
                        public_names.add(name)

        return public_names

    except Exception as e:
        logger.warning("Error resolving star import for %s: %s", module_name, e)
        return set()


def add_needed_imports_from_module(
    src_module_code: str,
    dst_module_code: str | cst.Module,
    src_path: Path,
    dst_path: Path,
    project_root: Path,
    helper_functions: list[FunctionSource] | None = None,
    helper_functions_fqn: set[str] | None = None,
) -> str:
    """Add all needed and used source module code imports to the destination module code, and return it."""
    src_module_code = delete___future___aliased_imports(src_module_code)
    if not helper_functions_fqn:
        helper_functions_fqn = {f.fully_qualified_name for f in (helper_functions or [])}

    dst_code_fallback = dst_module_code if isinstance(dst_module_code, str) else dst_module_code.code

    src_module_and_package: ModuleNameAndPackage = calculate_module_and_package(project_root, src_path)
    dst_module_and_package: ModuleNameAndPackage = calculate_module_and_package(project_root, dst_path)

    dst_context: CodemodContext = CodemodContext(
        filename=src_path.name,
        full_module_name=dst_module_and_package.name,
        full_package_name=dst_module_and_package.package,
    )
    gatherer: GatherImportsVisitor = GatherImportsVisitor(
        CodemodContext(
            filename=src_path.name,
            full_module_name=src_module_and_package.name,
            full_package_name=src_module_and_package.package,
        )
    )
    try:
        src_module = cst.parse_module(src_module_code)
        # Exclude function/class bodies so GatherImportsVisitor only sees module-level imports.
        # Nested imports (inside functions) are part of function logic and must not be
        # scheduled for add/remove — RemoveImportsVisitor would strip them as "unused".
        module_level_only = src_module.with_changes(
            body=[stmt for stmt in src_module.body if not isinstance(stmt, (cst.FunctionDef, cst.ClassDef))]
        )
        module_level_only.visit(gatherer)
    except Exception as e:
        logger.exception("Error parsing source module code: %s", e)
        return dst_code_fallback

    dotted_import_collector = DottedImportCollector()
    if isinstance(dst_module_code, cst.Module):
        parsed_dst_module = dst_module_code
        parsed_dst_module.visit(dotted_import_collector)
    else:
        try:
            parsed_dst_module = cst.parse_module(dst_module_code)
            parsed_dst_module.visit(dotted_import_collector)
        except cst.ParserSyntaxError as e:
            logger.exception("Syntax error in destination module code: %s", e)
            return dst_code_fallback

    try:
        for mod in gatherer.module_imports:
            # Skip __future__ imports as they cannot be imported directly
            # __future__ imports should only be imported with specific objects i.e from __future__ import annotations
            if mod == "__future__":
                continue
            if mod not in dotted_import_collector.imports:
                AddImportsVisitor.add_needed_import(dst_context, mod)
            RemoveImportsVisitor.remove_unused_import(dst_context, mod)
        aliased_objects = set()
        for mod, alias_pairs in gatherer.alias_mapping.items():
            for alias_pair in alias_pairs:
                if alias_pair[0] and alias_pair[1]:  # Both name and alias exist
                    aliased_objects.add(f"{mod}.{alias_pair[0]}")

        for mod, obj_seq in gatherer.object_mapping.items():
            for obj in obj_seq:
                if (
                    f"{mod}.{obj}" in helper_functions_fqn or dst_context.full_module_name == mod  # avoid circular deps
                ):
                    continue  # Skip adding imports for helper functions already in the context

                if f"{mod}.{obj}" in aliased_objects:
                    continue

                # Handle star imports by resolving them to actual symbol names
                if obj == "*":
                    resolved_symbols = resolve_star_import(mod, project_root)
                    logger.debug("Resolved star import from %s: %s", mod, resolved_symbols)

                    for symbol in resolved_symbols:
                        if (
                            f"{mod}.{symbol}" not in helper_functions_fqn
                            and f"{mod}.{symbol}" not in dotted_import_collector.imports
                        ):
                            AddImportsVisitor.add_needed_import(dst_context, mod, symbol)
                        RemoveImportsVisitor.remove_unused_import(dst_context, mod, symbol)
                else:
                    if f"{mod}.{obj}" not in dotted_import_collector.imports:
                        AddImportsVisitor.add_needed_import(dst_context, mod, obj)
                    RemoveImportsVisitor.remove_unused_import(dst_context, mod, obj)
    except Exception as e:
        logger.exception("Error adding imports to destination module code: %s", e)
        return dst_code_fallback

    for mod, asname in gatherer.module_aliases.items():
        if not asname:
            continue
        if f"{mod}.{asname}" not in dotted_import_collector.imports:
            AddImportsVisitor.add_needed_import(dst_context, mod, asname=asname)
        RemoveImportsVisitor.remove_unused_import(dst_context, mod, asname=asname)

    for mod, alias_pairs in gatherer.alias_mapping.items():
        for alias_pair in alias_pairs:
            if f"{mod}.{alias_pair[0]}" in helper_functions_fqn:
                continue

            if not alias_pair[0] or not alias_pair[1]:
                continue

            if f"{mod}.{alias_pair[1]}" not in dotted_import_collector.imports:
                AddImportsVisitor.add_needed_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])
            RemoveImportsVisitor.remove_unused_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])

    try:
        add_imports_visitor = AddImportsVisitor(dst_context)
        transformed_module = add_imports_visitor.transform_module(parsed_dst_module)
        transformed_module = RemoveImportsVisitor(dst_context).transform_module(transformed_module)
        return transformed_module.code.lstrip("\n")
    except Exception as e:
        logger.exception("Error adding imports to destination module code: %s", e)
        return dst_code_fallback
