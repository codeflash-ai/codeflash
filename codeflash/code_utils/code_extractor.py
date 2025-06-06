# ruff: noqa: ARG002
from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Optional

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, GatherImportsVisitor, RemoveImportsVisitor
from libcst.helpers import calculate_module_and_package

from codeflash.cli_cmds.console import logger
from codeflash.models.models import FunctionParent

if TYPE_CHECKING:
    from pathlib import Path

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

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self.scope_depth += 1
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.scope_depth -= 1

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self.scope_depth += 1
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.scope_depth -= 1

    def visit_If(self, node: cst.If) -> Optional[bool]:
        self.if_else_depth += 1
        return True

    def leave_If(self, original_node: cst.If) -> None:
        self.if_else_depth -= 1

    def visit_Else(self, node: cst.Else) -> Optional[bool]:
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

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # Add any new assignments that weren't in the original file
        new_statements = list(updated_node.body)

        # Find assignments to append
        assignments_to_append = [
            self.new_assignments[name]
            for name in self.new_assignment_order
            if name not in self.processed_assignments and name in self.new_assignments
        ]

        if assignments_to_append:
            # Add a blank line before appending new assignments if needed
            if new_statements and not isinstance(new_statements[-1], cst.EmptyLine):
                new_statements.append(cst.SimpleStatementLine([cst.Pass()], leading_lines=[cst.EmptyLine()]))
                new_statements.pop()  # Remove the Pass statement but keep the empty line

            # Add the new assignments
            new_statements.extend(
                [
                    cst.SimpleStatementLine([assignment], leading_lines=[cst.EmptyLine()])
                    for assignment in assignments_to_append
                ]
            )

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


class ImportInserter(cst.CSTTransformer):
    """Transformer that inserts global statements after the last import."""

    def __init__(self, global_statements: list[cst.SimpleStatementLine], last_import_line: int) -> None:
        super().__init__()
        self.global_statements = global_statements
        self.last_import_line = last_import_line
        self.current_line = 0
        self.inserted = False

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.Module:
        self.current_line += 1

        # If we're right after the last import and haven't inserted yet
        if self.current_line == self.last_import_line and not self.inserted:
            self.inserted = True
            return cst.Module(body=[updated_node, *self.global_statements])

        return cst.Module(body=[updated_node])

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # If there were no imports, add at the beginning of the module
        if self.last_import_line == 0 and not self.inserted:
            updated_body = list(updated_node.body)
            for stmt in reversed(self.global_statements):
                updated_body.insert(0, stmt)
            return updated_node.with_changes(body=updated_body)
        return updated_node


def extract_global_statements(source_code: str) -> list[cst.SimpleStatementLine]:
    """Extract global statements from source code."""
    module = cst.parse_module(source_code)
    collector = GlobalStatementCollector()
    module.visit(collector)
    return collector.global_statements


def find_last_import_line(target_code: str) -> int:
    """Find the line number of the last import statement."""
    module = cst.parse_module(target_code)
    finder = LastImportFinder()
    module.visit(finder)
    return finder.last_import_line


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
    non_assignment_global_statements = extract_global_statements(src_module_code)

    # Find the last import line in target
    last_import_line = find_last_import_line(dst_module_code)

    # Parse the target code
    target_module = cst.parse_module(dst_module_code)

    # Create transformer to insert non_assignment_global_statements
    transformer = ImportInserter(non_assignment_global_statements, last_import_line)
    #
    # # Apply transformation
    modified_module = target_module.visit(transformer)
    dst_module_code = modified_module.code

    # Parse the code
    original_module = cst.parse_module(dst_module_code)
    new_module = cst.parse_module(src_module_code)

    # Collect assignments from the new file
    new_collector = GlobalAssignmentCollector()
    new_module.visit(new_collector)

    # Transform the original file
    transformer = GlobalAssignmentTransformer(new_collector.assignments, new_collector.assignment_order)
    transformed_module = original_module.visit(transformer)

    return transformed_module.code


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
    cst.parse_module(src_module_code).visit(gatherer)
    try:
        for mod in gatherer.module_imports:
            AddImportsVisitor.add_needed_import(dst_context, mod)
            RemoveImportsVisitor.remove_unused_import(dst_context, mod)
        for mod, obj_seq in gatherer.object_mapping.items():
            for obj in obj_seq:
                if f"{mod}.{obj}" in helper_functions_fqn:
                    continue  # Skip adding imports for helper functions already in the context
                AddImportsVisitor.add_needed_import(dst_context, mod, obj)
                RemoveImportsVisitor.remove_unused_import(dst_context, mod, obj)
    except Exception as e:
        logger.exception(f"Error adding imports to destination module code: {e}")
        return dst_module_code
    for mod, asname in gatherer.module_aliases.items():
        AddImportsVisitor.add_needed_import(dst_context, mod, asname=asname)
        RemoveImportsVisitor.remove_unused_import(dst_context, mod, asname=asname)
    for mod, alias_pairs in gatherer.alias_mapping.items():
        for alias_pair in alias_pairs:
            if f"{mod}.{alias_pair[0]}" in helper_functions_fqn:
                continue
            AddImportsVisitor.add_needed_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])
            RemoveImportsVisitor.remove_unused_import(dst_context, mod, alias_pair[0], asname=alias_pair[1])

    try:
        parsed_module = cst.parse_module(dst_module_code)
    except cst.ParserSyntaxError as e:
        logger.exception(f"Syntax error in destination module code: {e}")
        return dst_module_code  # Return the original code if there's a syntax error
    try:
        transformed_module = AddImportsVisitor(dst_context).transform_module(parsed_module)
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
