from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import jedi
import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, GatherImportsVisitor, RemoveImportsVisitor
from libcst.helpers import calculate_module_and_package

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_consts import MAX_CONTEXT_LEN_REVIEW
from codeflash.models.models import CodePosition, FunctionParent

if TYPE_CHECKING:
    from libcst.helpers import ModuleNameAndPackage

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionSource


class GlobalAssignmentCollector(cst.CSTVisitor):
    """Collects all global assignment statements."""

    def __init__(self) -> None:
        super().__init__()
        self.assignments: dict[str, cst.Assign] = {}
        self.assignment_order: list[str] = []
        # Track scope depth to identify global assignments
        self.scope_depth = 0
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:  # noqa: ARG002
        self.scope_depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:  # noqa: ARG002
        self.scope_depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:  # noqa: ARG002
        self.scope_depth += 1
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: ARG002
        self.scope_depth -= 1

    def visit_If(self, node: cst.If) -> Optional[bool]:  # noqa: ARG002
        self.if_else_depth += 1
        return True

    def leave_If(self, original_node: cst.If) -> None:  # noqa: ARG002
        self.if_else_depth -= 1

    def visit_Else(self, node: cst.Else) -> Optional[bool]:  # noqa: ARG002
        # Else blocks are already counted as part of the if statement
        return True

    def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
        # Only process global assignments (not inside functions, classes, etc.)
        if self.scope_depth == 0 and self.if_else_depth == 0:  # We're at module level
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    name = target.target.value
                    self.assignments[name] = node
                    if name not in self.assignment_order:
                        self.assignment_order.append(name)
        return True


class GlobalAssignmentTransformer(cst.CSTTransformer):
    """Transforms global assignments in the original file with those from the new file."""

    def __init__(self, new_assignments: dict[str, cst.Assign], new_assignment_order: list[str]) -> None:
        super().__init__()
        self.new_assignments = new_assignments
        self.new_assignment_order = new_assignment_order
        self.processed_assignments: set[str] = set()
        self.scope_depth = 0
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: ARG002
        self.scope_depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:  # noqa: ARG002
        self.scope_depth -= 1
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: ARG002
        self.scope_depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:  # noqa: ARG002
        self.scope_depth -= 1
        return updated_node

    def visit_If(self, node: cst.If) -> None:  # noqa: ARG002
        self.if_else_depth += 1

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:  # noqa: ARG002
        self.if_else_depth -= 1
        return updated_node

    def visit_Else(self, node: cst.Else) -> None:
        # Else blocks are already counted as part of the if statement
        pass

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        if self.scope_depth > 0 or self.if_else_depth > 0:
            return updated_node

        # Check if this is a global assignment we need to replace
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                name = target.target.value
                if name in self.new_assignments:
                    self.processed_assignments.add(name)
                    return self.new_assignments[name]

        return updated_node

    def _find_insertion_index(self, updated_node: cst.Module) -> int:
        """Find the position of the last import statement in the top-level of the module."""
        insert_index = 0
        for i, stmt in enumerate(updated_node.body):
            is_top_level_import = isinstance(stmt, cst.SimpleStatementLine) and any(
                isinstance(child, (cst.Import, cst.ImportFrom)) for child in stmt.body
            )

            is_conditional_import = isinstance(stmt, cst.If) and all(
                isinstance(inner, cst.SimpleStatementLine)
                and all(isinstance(child, (cst.Import, cst.ImportFrom)) for child in inner.body)
                for inner in stmt.body.body
            )

            if is_top_level_import or is_conditional_import:
                insert_index = i + 1

            # Stop scanning once we reach a class or function definition.
            # Imports are supposed to be at the top of the file, but they can technically appear anywhere, even at the bottom of the file.
            # Without this check, a stray import later in the file
            # would incorrectly shift our insertion index below actual code definitions.
            if isinstance(stmt, (cst.ClassDef, cst.FunctionDef)):
                break

        return insert_index

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: ARG002
        # Add any new assignments that weren't in the original file
        new_statements = list(updated_node.body)

        # Find assignments to append
        assignments_to_append = [
            self.new_assignments[name]
            for name in self.new_assignment_order
            if name not in self.processed_assignments and name in self.new_assignments
        ]

        if assignments_to_append:
            # after last top-level imports
            insert_index = self._find_insertion_index(updated_node)

            assignment_lines = [
                cst.SimpleStatementLine([assignment], leading_lines=[cst.EmptyLine()])
                for assignment in assignments_to_append
            ]

            new_statements = list(chain(new_statements[:insert_index], assignment_lines, new_statements[insert_index:]))

            # Add a blank line after the last assignment if needed
            after_index = insert_index + len(assignment_lines)
            if after_index < len(new_statements):
                next_stmt = new_statements[after_index]
                # If there's no empty line, add one
                has_empty = any(isinstance(line, cst.EmptyLine) for line in next_stmt.leading_lines)
                if not has_empty:
                    new_statements[after_index] = next_stmt.with_changes(
                        leading_lines=[cst.EmptyLine(), *next_stmt.leading_lines]
                    )

        return updated_node.with_changes(body=new_statements)


class GlobalStatementCollector(cst.CSTVisitor):
    """Visitor that collects all global statements (excluding imports and functions/classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.global_statements = []
        self.in_function_or_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:  # noqa: ARG002
        # Don't visit inside classes
        self.in_function_or_class = True
        return False

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:  # noqa: ARG002
        self.in_function_or_class = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:  # noqa: ARG002
        # Don't visit inside functions
        self.in_function_or_class = True
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:  # noqa: ARG002
        self.in_function_or_class = False

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        if not self.in_function_or_class:
            for statement in node.body:
                # Skip imports
                if not isinstance(statement, (cst.Import, cst.ImportFrom, cst.Assign)):
                    self.global_statements.append(node)
                    break


class LastImportFinder(cst.CSTVisitor):
    """Finds the position of the last import statement in the module."""

    def __init__(self) -> None:
        super().__init__()
        self.last_import_line = 0
        self.current_line = 0

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        self.current_line += 1
        for statement in node.body:
            if isinstance(statement, (cst.Import, cst.ImportFrom)):
                self.last_import_line = self.current_line


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

    def _collect_imports_from_block(self, block: cst.IndentedBlock) -> None:
        for statement in block.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for child in statement.body:
                    if isinstance(child, cst.Import):
                        for alias in child.names:
                            module = self.get_full_dotted_name(alias.name)
                            asname = alias.asname.name.value if alias.asname else alias.name.value
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
                                asname = alias.asname.name.value if alias.asname else name
                                self.imports.add(f"{module}.{asname}")

    def visit_Module(self, node: cst.Module) -> None:
        self.depth = 0
        self._collect_imports_from_block(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: ARG002
        self.depth += 1

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:  # noqa: ARG002
        self.depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: ARG002
        self.depth += 1

    def leave_ClassDef(self, node: cst.ClassDef) -> None:  # noqa: ARG002
        self.depth -= 1

    def visit_If(self, node: cst.If) -> None:
        if self.depth == 0:
            self._collect_imports_from_block(node.body)

    def visit_Try(self, node: cst.Try) -> None:
        if self.depth == 0:
            self._collect_imports_from_block(node.body)


class ImportInserter(cst.CSTTransformer):
    """Transformer that inserts global statements after the last import."""

    def __init__(self, global_statements: list[cst.SimpleStatementLine], last_import_line: int) -> None:
        super().__init__()
        self.global_statements = global_statements
        self.last_import_line = last_import_line
        self.current_line = 0
        self.inserted = False

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,  # noqa: ARG002
        updated_node: cst.SimpleStatementLine,
    ) -> cst.Module:
        self.current_line += 1

        # If we're right after the last import and haven't inserted yet
        if self.current_line == self.last_import_line and not self.inserted:
            self.inserted = True
            return cst.Module(body=[updated_node, *self.global_statements])

        return cst.Module(body=[updated_node])

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:  # noqa: ARG002
        # If there were no imports, add at the beginning of the module
        if self.last_import_line == 0 and not self.inserted:
            updated_body = list(updated_node.body)
            for stmt in reversed(self.global_statements):
                updated_body.insert(0, stmt)
            return updated_node.with_changes(body=updated_body)
        return updated_node


def extract_global_statements(source_code: str) -> tuple[cst.Module, list[cst.SimpleStatementLine]]:
    """Extract global statements from source code."""
    module = cst.parse_module(source_code)
    collector = GlobalStatementCollector()
    module.visit(collector)
    return module, collector.global_statements


def find_last_import_line(target_code: str) -> int:
    """Find the line number of the last import statement."""
    module = cst.parse_module(target_code)
    finder = LastImportFinder()
    module.visit(finder)
    return finder.last_import_line


class FutureAliasedImportTransformer(cst.CSTTransformer):
    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,  # noqa: ARG002
        updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        import libcst.matchers as m

        if (
            (updated_node_module := updated_node.module)
            and updated_node_module.value == "__future__"
            and all(m.matches(name, m.ImportAlias()) for name in updated_node.names)
        ):
            if names := [name for name in updated_node.names if name.asname is None]:
                return updated_node.with_changes(names=names)
            return cst.RemoveFromParent()
        return updated_node


def delete___future___aliased_imports(module_code: str) -> str:
    return cst.parse_module(module_code).visit(FutureAliasedImportTransformer()).code


def add_global_assignments(src_module_code: str, dst_module_code: str) -> str:
    src_module, new_added_global_statements = extract_global_statements(src_module_code)
    dst_module, existing_global_statements = extract_global_statements(dst_module_code)

    unique_global_statements = []
    for stmt in new_added_global_statements:
        if any(
            stmt is existing_stmt or stmt.deep_equals(existing_stmt) for existing_stmt in existing_global_statements
        ):
            continue
        unique_global_statements.append(stmt)

    mod_dst_code = dst_module_code
    # Insert unique global statements if any
    if unique_global_statements:
        last_import_line = find_last_import_line(dst_module_code)
        # Reuse already-parsed dst_module
        transformer = ImportInserter(unique_global_statements, last_import_line)
        # Use visit inplace, don't parse again
        modified_module = dst_module.visit(transformer)
        mod_dst_code = modified_module.code
        # Parse the code after insertion
        original_module = cst.parse_module(mod_dst_code)
    else:
        # No new statements to insert, reuse already-parsed dst_module
        original_module = dst_module

    # Parse the src_module_code once only (already done above: src_module)
    # Collect assignments from the new file
    new_collector = GlobalAssignmentCollector()
    src_module.visit(new_collector)
    # Only create transformer if there are assignments to insert/transform
    if not new_collector.assignments:  # nothing to transform
        return mod_dst_code

    # Transform the original destination module
    transformer = GlobalAssignmentTransformer(new_collector.assignments, new_collector.assignment_order)
    transformed_module = original_module.visit(transformer)

    return transformed_module.code


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
            logger.warning(f"Could not find module file for {module_name}, skipping star import resolution")
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
                        elif isinstance(elt, ast.Str):  # Python < 3.8 compatibility
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

        return public_names  # noqa: TRY300

    except Exception as e:
        logger.warning(f"Error resolving star import for {module_name}: {e}")
        return set()


def add_needed_imports_from_module(
    src_module_code: str,
    dst_module_code: str,
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
        cst.parse_module(src_module_code).visit(gatherer)
    except Exception as e:
        logger.error(f"Error parsing source module code: {e}")
        return dst_module_code

    dotted_import_collector = DottedImportCollector()
    try:
        parsed_dst_module = cst.parse_module(dst_module_code)
        parsed_dst_module.visit(dotted_import_collector)
    except cst.ParserSyntaxError as e:
        logger.exception(f"Syntax error in destination module code: {e}")
        return dst_module_code  # Return the original code if there's a syntax error

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
                    logger.debug(f"Resolved star import from {mod}: {resolved_symbols}")

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
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_module_code

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
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_module_code


def get_code(functions_to_optimize: list[FunctionToOptimize]) -> tuple[str | None, set[tuple[str, str]]]:
    """Return the code for a function or methods in a Python module.

    functions_to_optimize is either a singleton FunctionToOptimize instance, which represents either a function at the
    module level or a method of a class at the module level, or it represents a list of methods of the same class.
    """
    if (
        not functions_to_optimize
        or (functions_to_optimize[0].parents and functions_to_optimize[0].parents[0].type != "ClassDef")
        or (
            len(functions_to_optimize[0].parents) > 1
            or ((len(functions_to_optimize) > 1) and len({fn.parents[0] for fn in functions_to_optimize}) != 1)
        )
    ):
        return None, set()

    file_path: Path = functions_to_optimize[0].file_path
    class_skeleton: set[tuple[int, int | None]] = set()
    contextual_dunder_methods: set[tuple[str, str]] = set()
    target_code: str = ""

    def find_target(node_list: list[ast.stmt], name_parts: tuple[str, str] | tuple[str]) -> ast.AST | None:
        target: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Assign | ast.AnnAssign | None = None
        node: ast.stmt
        for node in node_list:
            if (
                # The many mypy issues will be fixed once this code moves to the backend,
                # using Type Guards as we move to 3.10+.
                # We will cover the Type Alias case on the backend since it's a 3.12 feature.
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name_parts[0]
            ):
                target = node
                break
                # The next two cases cover type aliases in pre-3.12 syntax, where only single assignment is allowed.
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name_parts[0]
            ) or (isinstance(node, ast.AnnAssign) and hasattr(node.target, "id") and node.target.id == name_parts[0]):
                if class_skeleton:
                    break
                target = node
                break

        if target is None or len(name_parts) == 1:
            return target

        if not isinstance(target, ast.ClassDef):
            return None
        class_skeleton.add((target.lineno, target.body[0].lineno - 1))
        cbody = target.body
        if isinstance(cbody[0], ast.expr):  # Is a docstring
            class_skeleton.add((cbody[0].lineno, cbody[0].end_lineno))
            cbody = cbody[1:]
            cnode: ast.stmt
        for cnode in cbody:
            # Collect all dunder methods.
            cnode_name: str
            if (
                isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef))
                and len(cnode_name := cnode.name) > 4
                and cnode_name != name_parts[1]
                and cnode_name.isascii()
                and cnode_name.startswith("__")
                and cnode_name.endswith("__")
            ):
                contextual_dunder_methods.add((target.name, cnode_name))
                class_skeleton.add((cnode.lineno, cnode.end_lineno))

        return find_target(target.body, name_parts[1:])

    with file_path.open(encoding="utf8") as file:
        source_code: str = file.read()
    try:
        module_node: ast.Module = ast.parse(source_code)
    except SyntaxError:
        logger.exception("get_code - Syntax error while parsing code")
        return None, set()
    # Get the source code lines for the target node
    lines: list[str] = source_code.splitlines(keepends=True)
    if len(functions_to_optimize[0].parents) == 1:
        if (
            functions_to_optimize[0].parents[0].type == "ClassDef"
        ):  # All functions_to_optimize functions are methods of the same class.
            qualified_name_parts_list: list[tuple[str, str] | tuple[str]] = [
                (fto.parents[0].name, fto.function_name) for fto in functions_to_optimize
            ]

        else:
            logger.error(f"Error: get_code does not support inner functions: {functions_to_optimize[0].parents}")
            return None, set()
    elif len(functions_to_optimize[0].parents) == 0:
        qualified_name_parts_list = [(functions_to_optimize[0].function_name,)]
    else:
        logger.error(
            "Error: get_code does not support more than one level of nesting for now. "
            f"Parents: {functions_to_optimize[0].parents}"
        )
        return None, set()
    for qualified_name_parts in qualified_name_parts_list:
        target_node: ast.AST | None = find_target(module_node.body, qualified_name_parts)
        if target_node is None:
            continue

        if (
            isinstance(target_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and target_node.decorator_list
        ):
            target_code += "".join(lines[target_node.decorator_list[0].lineno - 1 : target_node.end_lineno])
        else:
            target_code += "".join(lines[target_node.lineno - 1 : target_node.end_lineno])
    if not target_code:
        return None, set()
    class_list: list[tuple[int, int | None]] = sorted(class_skeleton)
    class_code = "".join(["".join(lines[s_lineno - 1 : e_lineno]) for (s_lineno, e_lineno) in class_list])
    return class_code + target_code, contextual_dunder_methods


def extract_code(functions_to_optimize: list[FunctionToOptimize]) -> tuple[str | None, set[tuple[str, str]]]:
    edited_code, contextual_dunder_methods = get_code(functions_to_optimize)
    if edited_code is None:
        return None, set()
    try:
        compile(edited_code, "edited_code", "exec")
    except SyntaxError as e:
        logger.exception(f"extract_code - Syntax error in extracted optimization candidate code: {e}")
        return None, set()
    return edited_code, contextual_dunder_methods


def find_preexisting_objects(source_code: str) -> set[tuple[str, tuple[FunctionParent, ...]]]:
    """Find all preexisting functions, classes or class methods in the source code."""
    preexisting_objects: set[tuple[str, tuple[FunctionParent, ...]]] = set()
    try:
        module_node: ast.Module = ast.parse(source_code)
    except SyntaxError:
        logger.exception("find_preexisting_objects - Syntax error while parsing code")
        return preexisting_objects
    for node in module_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            preexisting_objects.add((node.name, ()))
        elif isinstance(node, ast.ClassDef):
            preexisting_objects.add((node.name, ()))
            for cnode in node.body:
                if isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    preexisting_objects.add((cnode.name, (FunctionParent(node.name, "ClassDef"),)))
    return preexisting_objects


@dataclass
class FunctionCallLocation:
    """Represents a location where the target function is called."""

    calling_function: str
    line: int
    column: int


@dataclass
class FunctionDefinitionInfo:
    """Contains information about a function definition."""

    name: str
    node: ast.FunctionDef
    source_code: str
    start_line: int
    end_line: int
    is_method: bool
    class_name: Optional[str] = None


class FunctionCallFinder(ast.NodeVisitor):
    """AST visitor that finds all function definitions that call a specific qualified function.

    Args:
        target_function_name: The qualified name of the function to find (e.g., "module.function" or "function")
        target_filepath: The filepath where the target function is defined

    """

    def __init__(self, target_function_name: str, target_filepath: str, source_lines: list[str]) -> None:
        self.target_function_name = target_function_name
        self.target_filepath = target_filepath
        self.source_lines = source_lines  # Store original source lines for extraction

        # Parse the target function name into parts
        self.target_parts = target_function_name.split(".")
        self.target_base_name = self.target_parts[-1]

        # Track current context
        self.current_function_stack: list[tuple[str, ast.FunctionDef]] = []
        self.current_class_stack: list[str] = []

        # Track imports to resolve qualified names
        self.imports: dict[str, str] = {}  # Maps imported names to their full paths

        # Results
        self.function_calls: list[FunctionCallLocation] = []
        self.calling_functions: set[str] = set()
        self.function_definitions: dict[str, FunctionDefinitionInfo] = {}

        # Track if we found calls in the current function
        self.found_call_in_current_function = False
        self.functions_with_nested_calls: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Track regular imports."""
        for alias in node.names:
            if alias.asname:
                # import module as alias
                self.imports[alias.asname] = alias.name
            else:
                # import module
                self.imports[alias.name.split(".")[-1]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            for alias in node.names:
                if alias.name == "*":
                    # from module import *
                    self.imports["*"] = node.module
                elif alias.asname:
                    # from module import name as alias
                    self.imports[alias.asname] = f"{node.module}.{alias.name}"
                else:
                    # from module import name
                    self.imports[alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track when entering a class definition."""
        self.current_class_stack.append(node.name)
        self.generic_visit(node)
        self.current_class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track when entering a function definition."""
        self._visit_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track when entering an async function definition."""
        self._visit_function_def(node)

    def _visit_function_def(self, node: ast.FunctionDef) -> None:
        """Track when entering a function definition."""
        func_name = node.name

        # Build the full qualified name including class if applicable
        full_name = f"{'.'.join(self.current_class_stack)}.{func_name}" if self.current_class_stack else func_name

        self.current_function_stack.append((full_name, node))
        self.found_call_in_current_function = False

        # Visit the function body
        self.generic_visit(node)

        # Process the function after visiting its body
        if self.found_call_in_current_function and full_name not in self.function_definitions:
            # Extract function source code
            source_code = self._extract_source_code(node)

            self.function_definitions[full_name] = FunctionDefinitionInfo(
                name=full_name,
                node=node,
                source_code=source_code,
                start_line=node.lineno,
                end_line=node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
                is_method=bool(self.current_class_stack),
                class_name=self.current_class_stack[-1] if self.current_class_stack else None,
            )

        # Handle nested functions - mark parent as containing nested calls
        if self.found_call_in_current_function and len(self.current_function_stack) > 1:
            parent_name = self.current_function_stack[-2][0]
            self.functions_with_nested_calls.add(parent_name)

            # Also store the parent function if not already stored
            if parent_name not in self.function_definitions:
                parent_node = self.current_function_stack[-2][1]
                parent_source = self._extract_source_code(parent_node)

                # Check if parent is a method (excluding current level)
                parent_class_context = self.current_class_stack if len(self.current_function_stack) == 2 else []

                self.function_definitions[parent_name] = FunctionDefinitionInfo(
                    name=parent_name,
                    node=parent_node,
                    source_code=parent_source,
                    start_line=parent_node.lineno,
                    end_line=parent_node.end_lineno if hasattr(parent_node, "end_lineno") else parent_node.lineno,
                    is_method=bool(parent_class_context),
                    class_name=parent_class_context[-1] if parent_class_context else None,
                )

        self.current_function_stack.pop()

        # Reset flag for parent function
        if self.current_function_stack:
            parent_name = self.current_function_stack[-1][0]
            self.found_call_in_current_function = parent_name in self.calling_functions

    def visit_Call(self, node: ast.Call) -> None:
        """Check if this call matches our target function."""
        if not self.current_function_stack:
            # Not inside a function, skip
            self.generic_visit(node)
            return

        if self._is_target_function_call(node):
            current_func_name = self.current_function_stack[-1][0]

            call_location = FunctionCallLocation(
                calling_function=current_func_name, line=node.lineno, column=node.col_offset
            )

            self.function_calls.append(call_location)
            self.calling_functions.add(current_func_name)
            self.found_call_in_current_function = True

        self.generic_visit(node)

    def _is_target_function_call(self, node: ast.Call) -> bool:
        """Determine if this call node is calling our target function."""
        call_name = self._get_call_name(node.func)
        if not call_name:
            return False

        # Check if it matches directly
        if call_name == self.target_function_name:
            return True

        # Check if it's just the base name matching
        if call_name == self.target_base_name:
            # Could be imported with a different name, check imports
            if call_name in self.imports:
                imported_path = self.imports[call_name]
                if imported_path == self.target_function_name or imported_path.endswith(
                    f".{self.target_function_name}"
                ):
                    return True
            # Could also be a direct call if we're in the same file
            return True

        # Check for qualified calls with imports
        call_parts = call_name.split(".")
        if call_parts[0] in self.imports:
            # Resolve the full path using imports
            base_import = self.imports[call_parts[0]]
            full_path = f"{base_import}.{'.'.join(call_parts[1:])}" if len(call_parts) > 1 else base_import

            if full_path == self.target_function_name or full_path.endswith(f".{self.target_function_name}"):
                return True

        return False

    def _get_call_name(self, func_node) -> Optional[str]:  # noqa: ANN001
        """Extract the name being called from a function node."""
        # Fast path short-circuit for ast.Name nodes
        if isinstance(func_node, ast.Name):
            return func_node.id

        # Fast attribute chain extraction (speed: append, loop, join, NO reversed)
        if isinstance(func_node, ast.Attribute):
            parts = []
            current = func_node
            # Unwind attribute chain as tight as possible (checked at each loop iteration)
            while True:
                parts.append(current.attr)
                val = current.value
                if isinstance(val, ast.Attribute):
                    current = val
                    continue
                if isinstance(val, ast.Name):
                    parts.append(val.id)
                    # Join in-place backwards via slice instead of reversed for slight speedup
                    return ".".join(parts[::-1])
                break
        return None

    def _extract_source_code(self, node: ast.FunctionDef) -> str:
        """Extract source code for a function node using original source lines."""
        if not self.source_lines or not hasattr(node, "lineno"):
            # Fallback to ast.unparse if available (Python 3.9+)
            try:
                return ast.unparse(node)
            except AttributeError:
                return f"# Source code extraction not available for {node.name}"

        # Get the lines for this function
        start_line = node.lineno - 1  # Convert to 0-based index
        end_line = node.end_lineno if hasattr(node, "end_lineno") else len(self.source_lines)

        # Extract the function lines
        func_lines = self.source_lines[start_line:end_line]

        # Find the minimum indentation (excluding empty lines)
        min_indent = float("inf")
        for line in func_lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # If this is a method (inside a class), preserve one level of indentation
        if self.current_class_stack:
            # Keep 4 spaces of indentation for methods
            dedent_amount = max(0, min_indent - 4)
            result_lines = []
            for line in func_lines:
                if line.strip():  # Only dedent non-empty lines
                    result_lines.append(line[dedent_amount:] if len(line) > dedent_amount else line)
                else:
                    result_lines.append(line)
        else:
            # For top-level functions, remove all leading indentation
            result_lines = []
            for line in func_lines:
                if line.strip():  # Only dedent non-empty lines
                    result_lines.append(line[min_indent:] if len(line) > min_indent else line)
                else:
                    result_lines.append(line)

        return "".join(result_lines).rstrip()

    def get_results(self) -> dict[str, str]:
        """Get the results of the analysis.

        Returns:
            A dictionary mapping qualified function names to their source code definitions.

        """
        return {info.name: info.source_code for info in self.function_definitions.values()}


def find_function_calls(source_code: str, target_function_name: str, target_filepath: str) -> dict[str, str]:
    """Find all function definitions that call a specific target function.

    Args:
        source_code: The Python source code to analyze
        target_function_name: The qualified name of the function to find (e.g., "module.function")
        target_filepath: The filepath where the target function is defined

    Returns:
        A dictionary mapping qualified function names to their source code definitions.
        Example: {"function_a": "def function_a():    ...", "MyClass.method_one": "def method_one(self):    ..."}

    """
    # Parse the source code
    tree = ast.parse(source_code)

    # Split source into lines for source extraction
    source_lines = source_code.splitlines(keepends=True)

    # Create and run the visitor
    visitor = FunctionCallFinder(target_function_name, target_filepath, source_lines)
    visitor.visit(tree)

    return visitor.get_results()


def find_occurances(
    qualified_name: str, file_path: str, fn_matches: list[Path], project_root: Path, tests_root: Path
) -> list[str]:  # max chars for context
    context_len = 0
    fn_call_context = ""
    for cur_file in fn_matches:
        if context_len > MAX_CONTEXT_LEN_REVIEW:
            break
        cur_file_path = Path(cur_file)
        # exclude references in tests
        try:
            if cur_file_path.relative_to(tests_root):
                continue
        except ValueError:
            pass
        with cur_file_path.open(encoding="utf8") as f:
            file_content = f.read()
        results = find_function_calls(file_content, target_function_name=qualified_name, target_filepath=file_path)
        if results:
            try:
                path_relative_to_project_root = cur_file_path.relative_to(project_root)
            except Exception as e:
                # shouldn't happen but ensuring we don't crash
                logger.debug(f"investigate {e}")
                continue
            fn_call_context += f"```python:{path_relative_to_project_root}\n"
            for (
                fn_definition
            ) in results.values():  # multiple functions in the file might be calling the desired function
                fn_call_context += f"{fn_definition}\n"
                context_len += len(fn_definition)
            fn_call_context += "```\n"
    return fn_call_context


def find_specific_function_in_file(
    source_code: str, filepath: Union[str, Path], target_function: str, target_class: str | None
) -> Optional[tuple[int, int]]:
    """Find a specific function definition in a Python file and return its location.

    Stops searching once the target is found (optimized for performance).

    Args:
        source_code: Source code string
        filepath: Path to the Python file
        target_function: Function Name of the function to find
        target_class: Class name of the function to find

    Returns:
        Tuple of (line_number, column_offset) if found, None otherwise

    """
    script = jedi.Script(code=source_code, path=filepath)
    names = script.get_names(all_scopes=True, definitions=True)
    for name in names:
        if name.type == "function" and name.name == target_function:
            # If class name specified, check parent
            if target_class:
                parent = name.parent()
                if parent and parent.name == target_class and parent.type == "class":
                    return CodePosition(line_no=name.line, col_no=name.column)
            else:
                # Top-level function match
                return CodePosition(line_no=name.line, col_no=name.column)

    return None  # Function not found


def get_fn_references_jedi(
    source_code: str, file_path: Path, project_root: Path, target_function: str, target_class: str | None
) -> list[Path]:
    function_position: CodePosition = find_specific_function_in_file(
        source_code, file_path, target_function, target_class
    )
    try:
        script = jedi.Script(code=source_code, path=file_path, project=jedi.Project(path=project_root))
        # Get references to the function
        references = script.get_references(line=function_position.line_no, column=function_position.col_no)
        # Collect unique file paths where references are found
        reference_files = set()
        for ref in references:
            if ref.module_path:
                # Convert to string and normalize path
                ref_path = str(ref.module_path)
                # Skip the definition itself
                if not (ref_path == file_path and ref.line == function_position.line_no):
                    reference_files.add(ref_path)
        return sorted(reference_files)
    except Exception as e:
        print(f"Error during Jedi analysis: {e}")
        return []


def get_opt_review_metrics(
    source_code: str, file_path: Path, qualified_name: str, project_root: Path, tests_root: Path
) -> str:
    try:
        qualified_name_split = qualified_name.rsplit(".", maxsplit=1)
        if len(qualified_name_split) == 1:
            target_function, target_class = qualified_name_split[0], None
        else:
            target_function, target_class = qualified_name_split[1], qualified_name_split[0]
        matches = get_fn_references_jedi(
            source_code, file_path, project_root, target_function, target_class
        )  # jedi is not perfect, it doesn't capture aliased references
        calling_fns_details = find_occurances(qualified_name, str(file_path), matches, project_root, tests_root)
    except Exception as e:
        calling_fns_details = ""
        logger.debug(f"Investigate {e}")
    return calling_fns_details
