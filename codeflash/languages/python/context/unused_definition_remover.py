from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.languages import current_language
from codeflash.languages.base import Language
from codeflash.languages.python.static_analysis.code_replacer import replace_function_definitions_in_module
from codeflash.models.models import CodeString, CodeStringsMarkdown

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.models import CodeOptimizationContext, FunctionSource


@dataclass
class UsageInfo:
    """Information about a name and its usage."""

    name: str
    used_by_qualified_function: bool = False
    dependencies: set[str] = field(default_factory=set)


def extract_names_from_targets(target: cst.CSTNode) -> list[str]:
    """Extract all variable names from a target node, including from tuple unpacking."""
    names = []

    # Handle a simple name
    if isinstance(target, cst.Name):
        names.append(target.value)

    # Handle any node with a value attribute (StarredElement, etc.)
    elif hasattr(target, "value"):
        names.extend(extract_names_from_targets(target.value))

    # Handle any node with elements attribute (tuples, lists, etc.)
    elif hasattr(target, "elements"):
        for element in target.elements:
            # Recursive call for each element
            names.extend(extract_names_from_targets(element))

    return names


def is_assignment_used(node: cst.CSTNode, definitions: dict[str, UsageInfo], name_prefix: str = "") -> bool:
    if isinstance(node, cst.Assign):
        targets = [target.target for target in node.targets]
    elif isinstance(node, (cst.AnnAssign, cst.AugAssign)):
        targets = [node.target]
    else:
        return False
    for target in targets:
        for name in extract_names_from_targets(target):
            lookup = f"{name_prefix}{name}" if name_prefix else name
            if lookup in definitions and definitions[lookup].used_by_qualified_function:
                return True
    return False


def recurse_sections(
    node: cst.CSTNode,
    section_names: list[str],
    prune_fn: Callable[[cst.CSTNode], tuple[cst.CSTNode | None, bool]],
    keep_non_target_children: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False
    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_fn(child)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target
            if keep_non_target_children:
                if section_found_target or new_children:
                    found_any_target |= section_found_target
                    updates[section] = new_children
            elif section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_fn(original_content)
            if keep_non_target_children:
                found_any_target |= found_target
                if filtered:
                    updates[section] = filtered
            elif found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered
    if keep_non_target_children:
        if updates:
            return node.with_changes(**updates), found_any_target
        return None, False
    if not found_any_target:
        return None, False
    return (node.with_changes(**updates) if updates else node), True


def collect_top_level_definitions(
    node: cst.CSTNode, definitions: Optional[dict[str, UsageInfo]] = None
) -> dict[str, UsageInfo]:
    """Recursively collect all top-level variable, function, and class definitions."""
    if definitions is None:
        definitions = {}

    if isinstance(node, cst.FunctionDef):
        name = node.name.value
        definitions[name] = UsageInfo(name=name)
        return definitions

    if isinstance(node, cst.ClassDef):
        name = node.name.value
        definitions[name] = UsageInfo(name=name)
        if isinstance(node.body, cst.IndentedBlock):
            prefix = name + "."
            for statement in node.body.body:
                if isinstance(statement, cst.FunctionDef):
                    method_name = prefix + statement.name.value
                    definitions[method_name] = UsageInfo(name=method_name)
        return definitions

    if isinstance(node, cst.Assign):
        for target in node.targets:
            for name in extract_names_from_targets(target.target):
                definitions[name] = UsageInfo(name=name)
        return definitions

    if isinstance(node, (cst.AnnAssign, cst.AugAssign)):
        for name in extract_names_from_targets(node.target):
            definitions[name] = UsageInfo(name=name)
        return definitions

    for section in get_section_names(node):
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            for child in original_content:
                collect_top_level_definitions(child, definitions)
        elif original_content is not None:
            collect_top_level_definitions(original_content, definitions)

    return definitions


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Return the section attribute names (e.g., body, orelse) for a given node if they exist."""
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


class DependencyCollector(cst.CSTVisitor):
    """Collects dependencies between definitions using the visitor pattern with depth tracking."""

    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider,)

    def __init__(self, definitions: dict[str, UsageInfo]) -> None:
        super().__init__()
        self.definitions = definitions
        # Track function and class depths
        self.function_depth = 0
        self.class_depth = 0
        # Track top-level qualified names
        self.current_top_level_name = ""
        self.current_class = ""
        # Track if we're processing a top-level variable
        self.processing_variable = False
        self.current_variable_names = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        function_name = node.name.value

        if self.function_depth == 0:
            # This is a top-level function
            if self.class_depth > 0:
                # If inside a class, we're now tracking dependencies at the class level
                self.current_top_level_name = f"{self.current_class}.{function_name}"
            else:
                # Regular top-level function
                self.current_top_level_name = function_name

        for param in node.params.params:
            if param.annotation:
                self._extract_names_from_annotation(param.annotation.annotation)

        self.function_depth += 1

    def _extract_names_from_annotation(self, node: cst.CSTNode) -> None:
        if isinstance(node, cst.Name):
            name = node.value
            if name in self.definitions and name != self.current_top_level_name and self.current_top_level_name:
                self.definitions[self.current_top_level_name].dependencies.add(name)
        elif isinstance(node, cst.Subscript):
            self._extract_names_from_annotation(node.value)
            for slice_item in node.slice:
                if hasattr(slice_item, "slice"):
                    self._extract_names_from_annotation(slice_item.slice)
        elif isinstance(node, cst.Attribute):
            self._extract_names_from_annotation(node.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self.function_depth -= 1

        if self.function_depth == 0 and self.class_depth == 0:
            # Exiting top-level function that's not in a class
            self.current_top_level_name = ""

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        class_name = node.name.value

        if self.class_depth == 0:
            # This is a top-level class
            self.current_class = class_name
            self.current_top_level_name = class_name

            # Track base classes as dependencies
            for base in node.bases:
                if isinstance(base.value, cst.Name):
                    base_name = base.value.value
                    if base_name in self.definitions and class_name in self.definitions:
                        self.definitions[class_name].dependencies.add(base_name)
                elif isinstance(base.value, cst.Attribute):
                    # Handle cases like module.ClassName
                    attr_name = base.value.attr.value
                    if attr_name in self.definitions and class_name in self.definitions:
                        self.definitions[class_name].dependencies.add(attr_name)

        self.class_depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self.class_depth -= 1

        if self.class_depth == 0:
            # Exiting top-level class
            self.current_class = ""
            self.current_top_level_name = ""

    def visit_Assign(self, node: cst.Assign) -> None:
        # Only handle top-level assignments
        if self.function_depth == 0 and self.class_depth == 0:
            for target in node.targets:
                # Extract all variable names from the target
                names = extract_names_from_targets(target.target)

                # Check if any of these names are top-level definitions we're tracking
                tracked_names = [name for name in names if name in self.definitions]
                if tracked_names:
                    self.processing_variable = True
                    self.current_variable_names.update(tracked_names)
                    # Use the first tracked name as the current top-level name (for dependency tracking)
                    self.current_top_level_name = tracked_names[0]

    def leave_Assign(self, original_node: cst.Assign) -> None:
        if self.processing_variable:
            self.processing_variable = False
            self.current_variable_names.clear()
            self.current_top_level_name = ""

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self.processing_variable = True
        if isinstance(node.target, cst.Name):
            self.current_variable_names.add(node.target.value)
        else:
            self.current_variable_names.update(extract_names_from_targets(node.target))

        self._extract_names_from_annotation(node.annotation.annotation)

        self.processing_variable = False
        self.current_variable_names.clear()

    def visit_Name(self, node: cst.Name) -> None:
        name = node.value

        # Skip if we're not inside a tracked definition
        if not self.current_top_level_name or self.current_top_level_name not in self.definitions:
            return

        # Skip if we're looking at the variable name itself in an assignment
        if self.processing_variable and name in self.current_variable_names:
            return

        if name in self.definitions and name != self.current_top_level_name:
            # Skip if this Name is the .attr part of an Attribute (e.g., 'x' in 'self.x')
            # We only want to track the base/value of attribute access, not the attribute name itself
            if self.class_depth > 0:
                parent = self.get_metadata(cst.metadata.ParentNodeProvider, node)
                if parent is not None and isinstance(parent, cst.Attribute):
                    # Check if this Name is the .attr (property name), not the .value (base)
                    # If it's the .attr, skip it - attribute names aren't references to definitions
                    if parent.attr is node:
                        return
                    # If it's the .value (base), only skip if it's self/cls
                    if name in ("self", "cls"):
                        return
            self.definitions[self.current_top_level_name].dependencies.add(name)


class QualifiedFunctionUsageMarker:
    """Marks definitions that are used by specific qualified functions."""

    def __init__(self, definitions: dict[str, UsageInfo], qualified_function_names: set[str]) -> None:
        self.definitions = definitions
        self.qualified_function_names = qualified_function_names
        self.expanded_qualified_functions = self._expand_qualified_functions()

    def _expand_qualified_functions(self) -> set[str]:
        """Expand the qualified function names to include related methods."""
        expanded = set(self.qualified_function_names)

        # Find class methods and add their containing classes and dunder methods
        for qualified_name in list(self.qualified_function_names):
            if "." in qualified_name:
                class_name, _method_name = qualified_name.split(".", 1)

                # Add the class itself
                expanded.add(class_name)

                # Add all dunder methods of the class
                for name in self.definitions:
                    if name.startswith(f"{class_name}.__") and name.endswith("__"):
                        expanded.add(name)

        return expanded

    def mark_used_definitions(self) -> None:
        """Find all qualified functions and mark them and their dependencies as used."""
        defs = self.definitions
        for func_name in self.expanded_qualified_functions & defs.keys():
            defs[func_name].used_by_qualified_function = True
            for dep in defs[func_name].dependencies:
                self.mark_as_used_recursively(dep)

    def mark_as_used_recursively(self, name: str) -> None:
        """Mark a name and all its dependencies as used recursively."""
        if name not in self.definitions:
            return

        if self.definitions[name].used_by_qualified_function:
            return  # Already marked

        self.definitions[name].used_by_qualified_function = True

        # Mark all dependencies as used
        for dep in self.definitions[name].dependencies:
            self.mark_as_used_recursively(dep)


def remove_unused_definitions_recursively(
    node: cst.CSTNode, definitions: dict[str, UsageInfo]
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node to remove unused definitions.

    Returns (filtered_node_or_None, used_by_function).
    """
    # Skip import statements
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return node, True

    # Never remove function definitions
    if isinstance(node, cst.FunctionDef):
        return node, True

    if isinstance(node, cst.ClassDef):
        class_name = node.name.value
        class_has_dependencies = class_name in definitions and definitions[class_name].used_by_qualified_function

        if isinstance(node.body, cst.IndentedBlock):
            new_statements = []
            for statement in node.body.body:
                if isinstance(statement, cst.FunctionDef):
                    new_statements.append(statement)
                elif isinstance(statement, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
                    if class_has_dependencies or is_assignment_used(
                        statement, definitions, name_prefix=f"{class_name}."
                    ):
                        new_statements.append(statement)
                else:
                    new_statements.append(statement)
            return node.with_changes(body=node.body.with_changes(body=new_statements)), True

        return node, class_has_dependencies

    # Handle assignments (Assign, AnnAssign, AugAssign)
    if isinstance(node, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
        if is_assignment_used(node, definitions):
            return node, True
        return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False
    return recurse_sections(
        node, section_names, lambda child: remove_unused_definitions_recursively(child, definitions)
    )


def collect_top_level_defs_with_usages(
    code: Union[str, cst.Module], qualified_function_names: set[str]
) -> dict[str, UsageInfo]:
    """Collect all top level definitions (classes, variables or functions) and their usages."""
    module = code if isinstance(code, cst.Module) else cst.parse_module(code)
    # Collect all definitions (top level classes, variables or function)
    definitions = collect_top_level_definitions(module)

    # Collect dependencies between definitions using the visitor pattern
    wrapper = cst.MetadataWrapper(module)
    dependency_collector = DependencyCollector(definitions)
    wrapper.visit(dependency_collector)

    # Mark definitions used by specified functions, and their dependencies recursively
    usage_marker = QualifiedFunctionUsageMarker(definitions, qualified_function_names)
    usage_marker.mark_used_definitions()
    return definitions


def remove_unused_definitions_by_function_names(
    code: Union[str, cst.Module], qualified_function_names: set[str]
) -> cst.Module:
    """Remove top-level definitions (classes, variables, functions) not used by the specified qualified function names."""
    try:
        module = code if isinstance(code, cst.Module) else cst.parse_module(code)
    except Exception as e:
        logger.debug(f"Failed to parse code with libcst: {type(e).__name__}: {e}")
        return code if isinstance(code, cst.Module) else cst.parse_module("")

    try:
        defs_with_usages = collect_top_level_defs_with_usages(module, qualified_function_names)

        # Apply the recursive removal transformation
        modified_module, _ = remove_unused_definitions_recursively(module, defs_with_usages)

        return modified_module if modified_module else cst.parse_module("")
    except Exception as e:
        # If any other error occurs during processing, return the original code
        logger.debug(f"Error processing code to remove unused definitions: {type(e).__name__}: {e}")
        return module


def revert_unused_helper_functions(
    project_root: Path, unused_helpers: list[FunctionSource], original_helper_code: dict[Path, str]
) -> None:
    """Revert unused helper functions back to their original definitions."""
    if not unused_helpers:
        return

    logger.debug(f"Reverting {len(unused_helpers)} unused helper function(s) to original definitions")

    # Resolve path keys for consistent comparison (Windows 8.3 short names may differ from Jedi-resolved paths)
    resolved_original_helper_code = {p.resolve(): code for p, code in original_helper_code.items()}

    unused_helpers_by_file = defaultdict(list)
    for helper in unused_helpers:
        unused_helpers_by_file[helper.file_path.resolve()].append(helper)

    for file_path, helpers_in_file in unused_helpers_by_file.items():
        if file_path in resolved_original_helper_code:
            try:
                original_code = resolved_original_helper_code[file_path]
                helper_names = [helper.qualified_name for helper in helpers_in_file]
                reverted_code = replace_function_definitions_in_module(
                    function_names=helper_names,
                    optimized_code=CodeStringsMarkdown(
                        code_strings=[
                            CodeString(code=original_code, file_path=Path(file_path).relative_to(project_root))
                        ]
                    ),
                    module_abspath=file_path,
                    preexisting_objects=set(),  # Empty set since we're reverting
                    project_root_path=project_root,
                    should_add_global_assignments=False,  # file already has global assignments from the optimization pass
                )

                if reverted_code:
                    logger.debug(f"Reverted unused helpers in {file_path}: {', '.join(helper_names)}")

            except Exception as e:
                logger.error(f"Error reverting unused helpers in {file_path}: {e}")


def _analyze_imports_in_optimized_code(
    optimized_ast: ast.AST, code_context: CodeOptimizationContext
) -> dict[str, set[str]]:
    """Map imported names to qualified helper names based on import statements in optimized code."""
    imported_names_map = defaultdict(set)

    # Precompute a two-level dict: module_name -> func_name -> [helpers]
    helpers_by_file_and_func = defaultdict(dict)
    helpers_by_file = defaultdict(list)  # preserved for "import module"
    for helper in code_context.helper_functions:
        jedi_type = helper.definition_type
        if jedi_type != "class":  # Include when definition_type is None (non-Python)
            func_name = helper.only_function_name
            module_name = helper.file_path.stem
            # Cache function lookup for this (module, func)
            helpers_by_file_and_func[module_name].setdefault(func_name, []).append(helper)
            helpers_by_file[module_name].append(helper)

    for node in ast.walk(optimized_ast):
        if isinstance(node, ast.ImportFrom):
            # Handle "from module import function" statements
            module_name = node.module
            if module_name:
                file_entry = helpers_by_file_and_func.get(module_name)
                if file_entry:
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        original_name = alias.name
                        helpers = file_entry.get(original_name)
                        if helpers:
                            imported_set = imported_names_map[imported_name]
                            for helper in helpers:
                                imported_set.add(helper.qualified_name)
                                imported_set.add(helper.fully_qualified_name)

        elif isinstance(node, ast.Import):
            # Handle "import module" statements
            for alias in node.names:
                imported_name = alias.asname if alias.asname else alias.name
                module_name = alias.name
                helpers = helpers_by_file.get(module_name)
                if helpers:
                    imported_set = imported_names_map[f"{imported_name}.{{func}}"]
                    for helper in helpers:
                        # For "import module" statements, functions would be called as module.function
                        full_call = f"{imported_name}.{helper.only_function_name}"
                        full_call_set = imported_names_map[full_call]
                        full_call_set.add(helper.qualified_name)
                        full_call_set.add(helper.fully_qualified_name)

    return dict(imported_names_map)


def find_target_node(
    root: ast.AST, function_to_optimize: FunctionToOptimize
) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    parents = function_to_optimize.parents
    node = root
    for parent in parents:
        # Fast loop: directly look for the matching ClassDef in node.body
        body = getattr(node, "body", None)
        if not body:
            return None
        for child in body:
            if isinstance(child, ast.ClassDef) and child.name == parent.name:
                node = child
                break
        else:
            return None

    # Now node is either the root or the target parent class; look for function
    body = getattr(node, "body", None)
    if not body:
        return None
    target_name = function_to_optimize.function_name
    for child in body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == target_name:
            return child
    return None


def _collect_attr_names(
    value_id: str, attr_name: str, class_name: str | None, names: set[str], imported_names_map: dict[str, set[str]]
) -> None:
    if value_id == "self":
        names.add(attr_name)
        if class_name:
            names.add(f"{class_name}.{attr_name}")
    else:
        names.add(attr_name)
        full_ref = f"{value_id}.{attr_name}"
        names.add(full_ref)
        mapped_names = imported_names_map.get(full_ref)
        if mapped_names:
            names.update(mapped_names)


def _collect_called_names(
    entrypoint_ast: ast.FunctionDef | ast.AsyncFunctionDef,
    function_to_optimize: FunctionToOptimize,
    imported_names_map: dict[str, set[str]],
) -> set[str]:
    called = {function_to_optimize.function_name}
    class_name = function_to_optimize.parents[0].name if function_to_optimize.parents else None

    for node in ast.walk(entrypoint_ast):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
                mapped_names = imported_names_map.get(node.func.id)
                if mapped_names:
                    called.update(mapped_names)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    _collect_attr_names(node.func.value.id, node.func.attr, class_name, called, imported_names_map)
                else:
                    called.add(node.func.attr)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            _collect_attr_names(node.value.id, node.attr, class_name, called, imported_names_map)

    return called


def detect_unused_helper_functions(
    function_to_optimize: FunctionToOptimize,
    code_context: CodeOptimizationContext,
    optimized_code: str | CodeStringsMarkdown,
) -> list[FunctionSource]:
    # Skip this analysis for non-Python languages since we use Python's ast module
    if current_language() != Language.PYTHON:
        logger.debug("Skipping unused helper function detection for non-Python languages")
        return []

    if isinstance(optimized_code, CodeStringsMarkdown) and len(optimized_code.code_strings) > 0:
        return list(
            chain.from_iterable(
                detect_unused_helper_functions(function_to_optimize, code_context, code.code)
                for code in optimized_code.code_strings
            )
        )

    try:
        optimized_ast = ast.parse(optimized_code)
        entrypoint_function_ast = find_target_node(optimized_ast, function_to_optimize)

        if not entrypoint_function_ast:
            logger.debug(f"Could not find entrypoint function {function_to_optimize.function_name} in optimized code")
            return []

        imported_names_map = _analyze_imports_in_optimized_code(optimized_ast, code_context)
        called_function_names = _collect_called_names(entrypoint_function_ast, function_to_optimize, imported_names_map)

        logger.debug(f"Functions called in optimized entrypoint: {called_function_names}")
        logger.debug(f"Imported names mapping: {imported_names_map}")

        unused_helpers = []
        entrypoint_file_path = function_to_optimize.file_path
        for helper_function in code_context.helper_functions:
            if helper_function.definition_type == "class":
                continue
            helper_qualified_name = helper_function.qualified_name
            helper_simple_name = helper_function.only_function_name
            helper_fully_qualified_name = helper_function.fully_qualified_name

            is_called = (
                helper_qualified_name in called_function_names
                or helper_simple_name in called_function_names
                or helper_fully_qualified_name in called_function_names
                or (
                    helper_function.file_path != entrypoint_file_path
                    and f"{helper_function.file_path.stem}.{helper_simple_name}" in called_function_names
                )
            )

            if not is_called:
                unused_helpers.append(helper_function)
                logger.debug(f"Helper function {helper_qualified_name} is not called in optimized code")
            else:
                logger.debug(f"Helper function {helper_qualified_name} is still called in optimized code")

    except Exception as e:
        logger.debug(f"Error detecting unused helper functions: {e}")
        return []
    else:
        return unused_helpers
