from __future__ import annotations

import ast
import time
from importlib.util import find_spec
from itertools import chain
from typing import TYPE_CHECKING, Optional

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, GatherImportsVisitor, RemoveImportsVisitor
from libcst.helpers import calculate_module_and_package

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_consts import MAX_CONTEXT_LEN_REVIEW
from codeflash.languages.base import Language
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from pathlib import Path

    from libcst.helpers import ModuleNameAndPackage

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import FunctionSource


class GlobalFunctionCollector(cst.CSTVisitor):
    """Collects all module-level function definitions (not inside classes or other functions)."""

    def __init__(self) -> None:
        super().__init__()
        self.functions: dict[str, cst.FunctionDef] = {}
        self.function_order: list[str] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        name = node.name.value
        self.functions[name] = node
        if name not in self.function_order:
            self.function_order.append(name)
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        return False


class GlobalFunctionTransformer(cst.CSTTransformer):
    """Transforms/adds module-level functions from the new file to the original file."""

    def __init__(self, new_functions: dict[str, cst.FunctionDef], new_function_order: list[str]) -> None:
        super().__init__()
        self.new_functions = new_functions
        self.new_function_order = new_function_order
        self.processed_functions: set[str] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        name = original_node.name.value
        if name in self.new_functions:
            self.processed_functions.add(name)
            return self.new_functions[name]
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Add any new functions that weren't in the original file
        new_statements = list(updated_node.body)

        functions_to_append = [
            self.new_functions[name]
            for name in self.new_function_order
            if name not in self.processed_functions and name in self.new_functions
        ]

        if functions_to_append:
            # Find the position of the last function or class definition
            insert_index = find_insertion_index_after_imports(updated_node)
            for i, stmt in enumerate(new_statements):
                if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
                    insert_index = i + 1

            # Add empty line before each new function
            function_nodes = []
            for func in functions_to_append:
                func_with_empty_line = func.with_changes(leading_lines=[cst.EmptyLine(), *func.leading_lines])
                function_nodes.append(func_with_empty_line)

            new_statements = list(chain(new_statements[:insert_index], function_nodes, new_statements[insert_index:]))

        return updated_node.with_changes(body=new_statements)


def collect_referenced_names(node: cst.CSTNode) -> set[str]:
    """Collect all names referenced in a CST node using recursive traversal."""
    names: set[str] = set()

    def _collect(n: cst.CSTNode) -> None:
        if isinstance(n, cst.Name):
            names.add(n.value)
        # Recursively process all children
        for child in n.children:
            _collect(child)

    _collect(node)
    return names


class GlobalAssignmentCollector(cst.CSTVisitor):
    """Collects all global assignment statements."""

    def __init__(self) -> None:
        super().__init__()
        self.assignments: dict[str, cst.Assign | cst.AnnAssign] = {}
        self.assignment_order: list[str] = []
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        return False

    def visit_If(self, node: cst.If) -> Optional[bool]:
        self.if_else_depth += 1
        return True

    def leave_If(self, original_node: cst.If) -> None:
        self.if_else_depth -= 1

    def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
        if self.if_else_depth == 0:
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    name = target.target.value
                    self.assignments[name] = node
                    if name not in self.assignment_order:
                        self.assignment_order.append(name)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> Optional[bool]:
        if self.if_else_depth == 0 and isinstance(node.target, cst.Name) and node.value is not None:
            name = node.target.value
            self.assignments[name] = node
            if name not in self.assignment_order:
                self.assignment_order.append(name)
        return True


def find_insertion_index_after_imports(node: cst.Module) -> int:
    """Find the position of the last import statement in the top-level of the module."""
    insert_index = 0
    for i, stmt in enumerate(node.body):
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


class GlobalAssignmentTransformer(cst.CSTTransformer):
    """Transforms global assignments in the original file with those from the new file."""

    def __init__(self, new_assignments: dict[str, cst.Assign | cst.AnnAssign], new_assignment_order: list[str]) -> None:
        super().__init__()
        self.new_assignments = new_assignments
        self.new_assignment_order = new_assignment_order
        self.processed_assignments: set[str] = set()
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def visit_If(self, node: cst.If) -> None:
        self.if_else_depth += 1

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        self.if_else_depth -= 1
        return updated_node

    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.CSTNode:
        if self.if_else_depth > 0:
            return updated_node

        # Check if this is a global assignment we need to replace
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                name = target.target.value
                if name in self.new_assignments:
                    self.processed_assignments.add(name)
                    return self.new_assignments[name]

        return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.CSTNode:
        if self.if_else_depth > 0:
            return updated_node

        # Check if this is a global annotated assignment we need to replace
        if isinstance(original_node.target, cst.Name):
            name = original_node.target.value
            if name in self.new_assignments:
                self.processed_assignments.add(name)
                return self.new_assignments[name]

        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Add any new assignments that weren't in the original file
        new_statements = list(updated_node.body)

        # Find assignments to append
        assignments_to_append = [
            (name, self.new_assignments[name])
            for name in self.new_assignment_order
            if name not in self.processed_assignments and name in self.new_assignments
        ]

        if not assignments_to_append:
            return updated_node.with_changes(body=new_statements)

        # Collect all class and function names defined in the module
        # These are the names that assignments might reference
        module_defined_names: set[str] = set()
        for stmt in new_statements:
            if isinstance(stmt, (cst.ClassDef, cst.FunctionDef)):
                module_defined_names.add(stmt.name.value)

        # Partition assignments: those that reference module definitions go at the end,
        # those that don't can go right after imports
        assignments_after_imports: list[tuple[str, cst.Assign | cst.AnnAssign]] = []
        assignments_after_definitions: list[tuple[str, cst.Assign | cst.AnnAssign]] = []

        for name, assignment in assignments_to_append:
            # Get the value being assigned
            if isinstance(assignment, (cst.Assign, cst.AnnAssign)) and assignment.value is not None:
                value_node = assignment.value
            else:
                # No value to analyze, safe to place after imports
                assignments_after_imports.append((name, assignment))
                continue

            # Collect names referenced in the assignment value
            referenced_names = collect_referenced_names(value_node)

            # Check if any referenced names are module-level definitions
            if referenced_names & module_defined_names:
                # This assignment references a class/function, place it after definitions
                assignments_after_definitions.append((name, assignment))
            else:
                # Safe to place right after imports
                assignments_after_imports.append((name, assignment))

        # Insert assignments that don't depend on module definitions right after imports
        if assignments_after_imports:
            insert_index = find_insertion_index_after_imports(updated_node)
            assignment_lines = [
                cst.SimpleStatementLine([assignment], leading_lines=[cst.EmptyLine()])
                for _, assignment in assignments_after_imports
            ]
            new_statements = list(chain(new_statements[:insert_index], assignment_lines, new_statements[insert_index:]))

        # Insert assignments that depend on module definitions after all class/function definitions
        if assignments_after_definitions:
            # Find the position after the last function or class definition
            insert_index = find_insertion_index_after_imports(cst.Module(body=new_statements))
            for i, stmt in enumerate(new_statements):
                if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
                    insert_index = i + 1

            assignment_lines = [
                cst.SimpleStatementLine([assignment], leading_lines=[cst.EmptyLine()])
                for _, assignment in assignments_after_definitions
            ]
            new_statements = list(chain(new_statements[:insert_index], assignment_lines, new_statements[insert_index:]))

        return updated_node.with_changes(body=new_statements)


class GlobalStatementTransformer(cst.CSTTransformer):
    """Appends global statements at the end of the module. Run LAST after other transformers."""

    def __init__(self, global_statements: list[cst.SimpleStatementLine]) -> None:
        super().__init__()
        self.global_statements = global_statements

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self.global_statements:
            return updated_node

        new_statements = list(updated_node.body)

        # Add empty line before each statement for readability
        statement_lines = [
            stmt.with_changes(leading_lines=[cst.EmptyLine(), *stmt.leading_lines]) for stmt in self.global_statements
        ]

        # Append statements at the end of the module
        # This ensures they come after all functions, classes, and assignments
        new_statements.extend(statement_lines)

        return updated_node.with_changes(body=new_statements)


class GlobalStatementCollector(cst.CSTVisitor):
    """Collects module-level statements (excluding imports, assignments, functions and classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.global_statements: list[cst.SimpleStatementLine] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        for statement in node.body:
            if not isinstance(statement, (cst.Import, cst.ImportFrom, cst.Assign, cst.AnnAssign)):
                self.global_statements.append(node)
                break


class DottedImportCollector(cst.CSTVisitor):
    """Collects top-level imports as normalized dotted strings (e.g. 'from pathlib import Path' -> 'pathlib.Path')."""

    def __init__(self) -> None:
        self.imports: set[str] = set()

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
        self._collect_imports_from_block(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        return False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def visit_If(self, node: cst.If) -> None:
        self._collect_imports_from_block(node.body)

    def visit_Try(self, node: cst.Try) -> None:
        self._collect_imports_from_block(node.body)


def extract_global_statements(source_code: str) -> tuple[cst.Module, list[cst.SimpleStatementLine]]:
    """Extract global statements from source code."""
    module = cst.parse_module(source_code)
    collector = GlobalStatementCollector()
    module.visit(collector)
    return module, collector.global_statements


class FutureAliasedImportTransformer(cst.CSTTransformer):
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
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

    new_assignment_collector = GlobalAssignmentCollector()
    src_module.visit(new_assignment_collector)

    # Collect module-level functions from both source and destination
    src_function_collector = GlobalFunctionCollector()
    src_module.visit(src_function_collector)

    dst_function_collector = GlobalFunctionCollector()
    dst_module.visit(dst_function_collector)

    # Filter out functions that already exist in the destination (only add truly new functions)
    new_functions = {
        name: func
        for name, func in src_function_collector.functions.items()
        if name not in dst_function_collector.functions
    }
    new_function_order = [name for name in src_function_collector.function_order if name in new_functions]

    if not new_assignment_collector.assignments and not new_functions and not unique_global_statements:
        return dst_module_code

    # Transform in order: functions, then assignments, then global statements (so each can reference the previous)
    if new_functions:
        dst_module = dst_module.visit(GlobalFunctionTransformer(new_functions, new_function_order))

    if new_assignment_collector.assignments:
        dst_module = dst_module.visit(
            GlobalAssignmentTransformer(new_assignment_collector.assignments, new_assignment_collector.assignment_order)
        )

    if unique_global_statements:
        dst_module = dst_module.visit(GlobalStatementTransformer(unique_global_statements))

    return dst_module.code


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
        logger.warning(f"Error resolving star import for {module_name}: {e}")
        return set()


def add_needed_imports_from_module(
    src_module_code: str | cst.Module,
    dst_module_code: str | cst.Module,
    src_path: Path,
    dst_path: Path,
    project_root: Path,
    helper_functions: list[FunctionSource] | None = None,
    helper_functions_fqn: set[str] | None = None,
) -> str:
    """Add all needed and used source module code imports to the destination module code, and return it."""
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
        if isinstance(src_module_code, cst.Module):
            src_module = src_module_code.visit(FutureAliasedImportTransformer())
        else:
            src_module = cst.parse_module(src_module_code).visit(FutureAliasedImportTransformer())
        # Exclude function/class bodies so GatherImportsVisitor only sees module-level imports.
        # Nested imports (inside functions) are part of function logic and must not be
        # scheduled for add/remove — RemoveImportsVisitor would strip them as "unused".
        module_level_only = src_module.with_changes(
            body=[stmt for stmt in src_module.body if not isinstance(stmt, (cst.FunctionDef, cst.ClassDef))]
        )
        module_level_only.visit(gatherer)
    except Exception as e:
        logger.error(f"Error parsing source module code: {e}")
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
            logger.exception(f"Syntax error in destination module code: {e}")
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
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_code_fallback


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

        if not isinstance(target, ast.ClassDef) or len(name_parts) < 2:
            return None
        # At this point, name_parts has at least 2 elements
        method_name: str = name_parts[1]  # type: ignore[misc]
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
                and cnode_name != method_name
                and cnode_name.isascii()
                and cnode_name.startswith("__")
                and cnode_name.endswith("__")
            ):
                contextual_dunder_methods.add((target.name, cnode_name))
                class_skeleton.add((cnode.lineno, cnode.end_lineno))

        return find_target(target.body, (method_name,))

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
        target_node = find_target(module_node.body, qualified_name_parts)
        if target_node is None:
            continue
        # find_target returns FunctionDef, AsyncFunctionDef, ClassDef, Assign, or AnnAssign - all have lineno/end_lineno
        if not isinstance(
            target_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
        ):
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


has_numba = find_spec("numba") is not None

NUMERICAL_MODULES = frozenset({"numpy", "torch", "numba", "jax", "tensorflow", "math", "scipy"})
# Modules that require numba to be installed for optimization
NUMBA_REQUIRED_MODULES = frozenset({"numpy", "math", "scipy"})


def _uses_numerical_names(node: ast.AST, numerical_names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in numerical_names for n in ast.walk(node))


def _collect_numerical_imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    numerical_names: set[str] = set()
    modules_used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in NUMERICAL_MODULES:
                    numerical_names.add(alias.asname if alias.asname else module_root)
                    modules_used.add(module_root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_root = node.module.split(".")[0]
            if module_root in NUMERICAL_MODULES:
                for alias in node.names:
                    if alias.name == "*":
                        numerical_names.add(module_root)
                    else:
                        numerical_names.add(alias.asname if alias.asname else alias.name)
                modules_used.add(module_root)
    return numerical_names, modules_used


def _find_function_node(tree: ast.Module, name_parts: list[str]) -> ast.FunctionDef | None:
    """Find a function node in the AST given its qualified name parts (e.g. ["ClassName", "method"] or ["func"])."""
    if not name_parts or len(name_parts) > 2:
        return None
    body: list[ast.stmt] = tree.body
    for part in name_parts[:-1]:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name == part:
                body = node.body
                break
        else:
            return None
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name_parts[-1]:
            return node
    return None


def is_numerical_code(code_string: str, function_name: str | None = None) -> bool:
    """Check if a function uses numerical computing libraries (numpy, torch, numba, jax, tensorflow, scipy, math).

    Returns False for math/numpy/scipy if numba is not installed.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return False

    # Collect names that reference numerical modules from imports
    numerical_names, modules_used = _collect_numerical_imports(tree)

    if not function_name:
        # Return True if modules used and (numba available or modules don't all require numba)
        return bool(modules_used) and (has_numba or not modules_used.issubset(NUMBA_REQUIRED_MODULES))

    # Split the function name to handle class methods
    name_parts = function_name.split(".")

    # Find the target function node
    target_function = _find_function_node(tree, name_parts)
    if target_function is None:
        return False

    if not _uses_numerical_names(target_function, numerical_names):
        return False

    # If numba is not installed and all modules used require numba for optimization,
    # return False since we can't optimize this code
    return not (not has_numba and modules_used.issubset(NUMBA_REQUIRED_MODULES))


def get_opt_review_metrics(
    source_code: str, file_path: Path, qualified_name: str, project_root: Path, tests_root: Path, language: Language
) -> str:
    """Get markdown-formatted calling function context for optimization review."""
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.registry import get_language_support
    from codeflash.models.models import FunctionParent

    start_time = time.perf_counter()

    try:
        # Get the language support
        lang_support = get_language_support(language)
        if lang_support is None:
            return ""

        # Parse qualified name to get function name and class name
        qualified_name_split = qualified_name.rsplit(".", maxsplit=1)
        if len(qualified_name_split) == 1:
            function_name, class_name = qualified_name_split[0], None
        else:
            function_name, class_name = qualified_name_split[1], qualified_name_split[0]

        # Create a FunctionToOptimize for the function
        # We don't have full line info here, so we'll use defaults
        parents: list[FunctionParent] = []
        if class_name:
            parents = [FunctionParent(name=class_name, type="ClassDef")]

        func_info = FunctionToOptimize(
            function_name=function_name,
            file_path=file_path,
            parents=parents,
            starting_line=1,
            ending_line=1,
            language=str(language),
        )

        # Find references using language support
        references = lang_support.find_references(func_info, project_root, tests_root, max_files=500)

        if not references:
            return ""

        # Format references as markdown code blocks
        calling_fns_details = _format_references_as_markdown(references, file_path, project_root, language)

    except Exception as e:
        logger.debug(f"Error getting function references: {e}")
        calling_fns_details = ""

    end_time = time.perf_counter()
    logger.debug(f"Got function references in {end_time - start_time:.2f} seconds")
    return calling_fns_details


def _format_references_as_markdown(references: list, file_path: Path, project_root: Path, language: Language) -> str:
    """Format references as markdown code blocks with calling function code."""
    # Group references by file
    refs_by_file: dict[Path, list] = {}
    for ref in references:
        # Exclude the source file's definition/import references
        if ref.file_path == file_path and ref.reference_type in ("import", "reexport"):
            continue

        if ref.file_path not in refs_by_file:
            refs_by_file[ref.file_path] = []
        refs_by_file[ref.file_path].append(ref)

    from codeflash.languages.registry import get_language_support

    try:
        lang_support = get_language_support(language)
    except Exception:
        lang_support = None

    fn_call_context = ""
    context_len = 0

    for ref_file, file_refs in refs_by_file.items():
        if context_len > MAX_CONTEXT_LEN_REVIEW:
            break

        try:
            path_relative = ref_file.relative_to(project_root)
        except ValueError:
            continue

        # Get syntax highlighting language
        ext = ref_file.suffix.lstrip(".")
        if language == Language.PYTHON:
            lang_hint = "python"
        elif ext in ("ts", "tsx"):
            lang_hint = "typescript"
        else:
            lang_hint = "javascript"

        # Read the file to extract calling function context
        try:
            file_content = ref_file.read_text(encoding="utf-8")
            lines = file_content.splitlines()
        except Exception:
            continue

        # Get unique caller functions from this file
        callers_seen: set[str] = set()
        caller_contexts: list[str] = []

        for ref in file_refs:
            caller = ref.caller_function or "<module>"
            if caller in callers_seen:
                continue
            callers_seen.add(caller)

            # Extract context around the reference
            if ref.caller_function:
                # Try to extract the full calling function
                func_code = None
                if lang_support is not None:
                    func_code = lang_support.extract_calling_function_source(
                        file_content, ref.caller_function, ref.line
                    )
                if func_code:
                    caller_contexts.append(func_code)
                    context_len += len(func_code)
            else:
                # Module-level call - show a few lines of context
                start_line = max(0, ref.line - 3)
                end_line = min(len(lines), ref.line + 2)
                context_code = "\n".join(lines[start_line:end_line])
                caller_contexts.append(context_code)
                context_len += len(context_code)

        if caller_contexts:
            fn_call_context += f"```{lang_hint}:{path_relative.as_posix()}\n"
            fn_call_context += "\n".join(caller_contexts)
            fn_call_context += "\n```\n"

    return fn_call_context
