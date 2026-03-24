from __future__ import annotations

from itertools import chain
from typing import cast

import libcst as cst


class GlobalFunctionCollector(cst.CSTVisitor):
    """Collects all module-level function definitions (not inside classes or other functions)."""

    def __init__(self) -> None:
        super().__init__()
        self.functions: dict[str, cst.FunctionDef] = {}
        self.function_order: list[str] = []
        self.scope_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if self.scope_depth == 0:
            # Module-level function
            name = node.name.value
            self.functions[name] = node
            if name not in self.function_order:
                self.function_order.append(name)
        self.scope_depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.scope_depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.scope_depth += 1
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.scope_depth -= 1


class GlobalFunctionTransformer(cst.CSTTransformer):
    """Transforms/adds module-level functions from the new file to the original file."""

    def __init__(self, new_functions: dict[str, cst.FunctionDef], new_function_order: list[str]) -> None:
        super().__init__()
        self.new_functions = new_functions
        self.new_function_order = new_function_order
        self.processed_functions: set[str] = set()
        self.scope_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.scope_depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        self.scope_depth -= 1
        if self.scope_depth > 0:
            return updated_node

        # Check if this is a module-level function we need to replace
        name = original_node.name.value
        if name in self.new_functions:
            self.processed_functions.add(name)
            return self.new_functions[name]
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.scope_depth -= 1
        return updated_node

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

    def collect(n: cst.CSTNode) -> None:
        if isinstance(n, cst.Name):
            names.add(n.value)
        # Recursively process all children
        for child in n.children:
            collect(child)

    collect(node)
    return names


class GlobalAssignmentCollector(cst.CSTVisitor):
    """Collects all global assignment statements."""

    def __init__(self) -> None:
        super().__init__()
        self.assignments: dict[str, cst.Assign | cst.AnnAssign] = {}
        self.assignment_order: list[str] = []
        # Track scope depth to identify global assignments
        self.scope_depth = 0
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.scope_depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.scope_depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.scope_depth += 1
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.scope_depth -= 1

    def visit_If(self, node: cst.If) -> bool | None:
        self.if_else_depth += 1
        return True

    def leave_If(self, original_node: cst.If) -> None:
        self.if_else_depth -= 1

    def visit_Else(self, node: cst.Else) -> bool | None:
        # Else blocks are already counted as part of the if statement
        return True

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        # Only process global assignments (not inside functions, classes, etc.)
        if self.scope_depth == 0 and self.if_else_depth == 0:  # We're at module level
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    name = target.target.value
                    self.assignments[name] = node
                    if name not in self.assignment_order:
                        self.assignment_order.append(name)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        # Handle annotated assignments like: _CACHE: Dict[str, int] = {}
        # Only process module-level annotated assignments with a value
        if (
            self.scope_depth == 0
            and self.if_else_depth == 0
            and isinstance(node.target, cst.Name)
            and node.value is not None
        ):
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
        self.scope_depth = 0
        self.if_else_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.scope_depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        self.scope_depth -= 1
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.scope_depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.scope_depth -= 1
        return updated_node

    def visit_If(self, node: cst.If) -> None:
        self.if_else_depth += 1

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        self.if_else_depth -= 1
        return updated_node

    def visit_Else(self, node: cst.Else) -> None:
        # Else blocks are already counted as part of the if statement
        pass

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.Assign | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        if self.scope_depth > 0 or self.if_else_depth > 0:
            return updated_node

        # Check if this is a global assignment we need to replace
        for target in original_node.targets:
            if isinstance(target.target, cst.Name):
                name = target.target.value
                if name in self.new_assignments:
                    self.processed_assignments.add(name)
                    return cast("cst.Assign", self.new_assignments[name])

        return updated_node

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.AnnAssign | cst.FlattenSentinel[cst.BaseSmallStatement] | cst.RemovalSentinel:
        if self.scope_depth > 0 or self.if_else_depth > 0:
            return updated_node

        # Check if this is a global annotated assignment we need to replace
        if isinstance(original_node.target, cst.Name):
            name = original_node.target.value
            if name in self.new_assignments:
                self.processed_assignments.add(name)
                return cast("cst.AnnAssign", self.new_assignments[name])

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
    """Transformer that appends global statements at the end of the module.

    This ensures that global statements (like function calls at module level) are placed
    after all functions, classes, and assignments they might reference, preventing NameError
    at module load time.

    This transformer should be run LAST after GlobalFunctionTransformer and
    GlobalAssignmentTransformer have already added their content.
    """

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
    """Visitor that collects all global statements (excluding imports and functions/classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.global_statements = []
        self.in_function_or_class = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Don't visit inside classes
        self.in_function_or_class = True
        return False

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.in_function_or_class = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        # Don't visit inside functions
        self.in_function_or_class = True
        return False

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.in_function_or_class = False

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        if not self.in_function_or_class:
            for statement in node.body:
                # Skip imports and assignments (both regular and annotated)
                if not isinstance(statement, (cst.Import, cst.ImportFrom, cst.Assign, cst.AnnAssign)):
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

    # Reuse already-parsed dst_module
    original_module = dst_module

    # Parse the src_module_code once only (already done above: src_module)
    # Collect assignments from the new file
    new_assignment_collector = GlobalAssignmentCollector()
    src_module.visit(new_assignment_collector)

    # Collect module-level functions from both source and destination
    src_function_collector = GlobalFunctionCollector()
    src_module.visit(src_function_collector)

    dst_function_collector = GlobalFunctionCollector()
    original_module.visit(dst_function_collector)

    # Filter out functions that already exist in the destination (only add truly new functions)
    new_functions = {
        name: func
        for name, func in src_function_collector.functions.items()
        if name not in dst_function_collector.functions
    }
    new_function_order = [name for name in src_function_collector.function_order if name in new_functions]

    # If there are no assignments, no new functions, and no global statements, return unchanged
    if not new_assignment_collector.assignments and not new_functions and not unique_global_statements:
        return dst_module_code

    # The order of transformations matters:
    # 1. Functions first - so assignments and statements can reference them
    # 2. Assignments second - so they come after functions but before statements
    # 3. Global statements last - so they can reference both functions and assignments

    # Transform functions if any
    if new_functions:
        function_transformer = GlobalFunctionTransformer(new_functions, new_function_order)
        original_module = original_module.visit(function_transformer)

    # Transform assignments if any
    if new_assignment_collector.assignments:
        transformer = GlobalAssignmentTransformer(
            new_assignment_collector.assignments, new_assignment_collector.assignment_order
        )
        original_module = original_module.visit(transformer)

    # Insert global statements (like function calls at module level) LAST,
    # after all functions and assignments are added, to ensure they can reference any
    # functions or variables defined in the module
    if unique_global_statements:
        statement_transformer = GlobalStatementTransformer(unique_global_statements)
        original_module = original_module.visit(statement_transformer)

    return original_module.code
