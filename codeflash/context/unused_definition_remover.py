from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst


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


def collect_top_level_definitions(node: cst.CSTNode, definitions: dict[str, UsageInfo] = None) -> dict[str, UsageInfo]:
    """Recursively collect all top-level variable, function, and class definitions."""
    if definitions is None:
        definitions = {}

    # Handle top-level function definitions
    if isinstance(node, cst.FunctionDef):
        name = node.name.value
        definitions[name] = UsageInfo(
            name=name,
            used_by_qualified_function=False,  # Will be marked later if in qualified functions
        )
        return definitions

    # Handle top-level class definitions
    if isinstance(node, cst.ClassDef):
        name = node.name.value
        definitions[name] = UsageInfo(name=name)

        # Also collect method definitions within the class
        if hasattr(node, "body") and isinstance(node.body, cst.IndentedBlock):
            for statement in node.body.body:
                if isinstance(statement, cst.FunctionDef):
                    method_name = f"{name}.{statement.name.value}"
                    definitions[method_name] = UsageInfo(name=method_name)

        return definitions

    # Handle top-level variable assignments
    if isinstance(node, cst.Assign):
        for target in node.targets:
            names = extract_names_from_targets(target.target)
            for name in names:
                definitions[name] = UsageInfo(name=name)
        return definitions

    if isinstance(node, (cst.AnnAssign, cst.AugAssign)):
        if isinstance(node.target, cst.Name):
            name = node.target.value
            definitions[name] = UsageInfo(name=name)
        else:
            names = extract_names_from_targets(node.target)
            for name in names:
                definitions[name] = UsageInfo(name=name)
        return definitions

    # Recursively process children. Takes care of top level assignments in if/else/while/for blocks
    section_names = get_section_names(node)

    if section_names:
        for section in section_names:
            original_content = getattr(node, section, None)
            # If section contains a list of nodes
            if isinstance(original_content, (list, tuple)):
                for child in original_content:
                    collect_top_level_definitions(child, definitions)
            # If section contains a single node
            elif original_content is not None:
                collect_top_level_definitions(original_content, definitions)

    return definitions


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Return the section attribute names (e.g., body, orelse) for a given node if they exist."""
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


class DependencyCollector(cst.CSTVisitor):
    """Collects dependencies between definitions using the visitor pattern with depth tracking."""

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

        # Check parameter type annotations for dependencies
        if hasattr(node, "params") and node.params:
            for param in node.params.params:
                if param.annotation:
                    # Visit the annotation to extract dependencies
                    self._collect_annotation_dependencies(param.annotation)

        self.function_depth += 1

    def _collect_annotation_dependencies(self, annotation: cst.Annotation) -> None:
        """Extract dependencies from type annotations"""
        if hasattr(annotation, "annotation"):
            # Extract names from annotation (could be Name, Attribute, Subscript, etc.)
            self._extract_names_from_annotation(annotation.annotation)

    def _extract_names_from_annotation(self, node: cst.CSTNode) -> None:
        """Extract names from a type annotation node"""
        # Simple name reference like 'int', 'str', or custom type
        if isinstance(node, cst.Name):
            name = node.value
            if name in self.definitions and name != self.current_top_level_name and self.current_top_level_name:
                self.definitions[self.current_top_level_name].dependencies.add(name)

        # Handle compound annotations like List[int], Dict[str, CustomType], etc.
        elif isinstance(node, cst.Subscript):
            if hasattr(node, "value"):
                self._extract_names_from_annotation(node.value)
            if hasattr(node, "slice"):
                for slice_item in node.slice:
                    if hasattr(slice_item, "slice"):
                        self._extract_names_from_annotation(slice_item.slice)

        # Handle attribute access like module.Type
        elif isinstance(node, cst.Attribute):
            if hasattr(node, "value"):
                self._extract_names_from_annotation(node.value)
            # No need to check the attribute name itself as it's likely not a top-level definition

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
        # Extract names from the variable annotations
        if hasattr(node, "annotation") and node.annotation:
            # First mark we're processing a variable to avoid recording it as a dependency of itself
            self.processing_variable = True
            if isinstance(node.target, cst.Name):
                self.current_variable_names.add(node.target.value)
            else:
                self.current_variable_names.update(extract_names_from_targets(node.target))

            # Process the annotation
            self._collect_annotation_dependencies(node.annotation)

            # Reset processing state
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

        # Check if name is a top-level definition we're tracking
        if name in self.definitions and name != self.current_top_level_name:
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
                class_name, method_name = qualified_name.split(".", 1)

                # Add the class itself
                expanded.add(class_name)

                # Add all dunder methods of the class
                for name in self.definitions:
                    if name.startswith(f"{class_name}.__") and name.endswith("__"):
                        expanded.add(name)

        return expanded

    def mark_used_definitions(self) -> None:
        """Find all qualified functions and mark them and their dependencies as used."""
        # First identify all specified functions (including expanded ones)
        functions_to_mark = [name for name in self.expanded_qualified_functions if name in self.definitions]

        # For each specified function, mark it and all its dependencies as used
        for func_name in functions_to_mark:
            self.definitions[func_name].used_by_qualified_function = True
            for dep in self.definitions[func_name].dependencies:
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

    Args:
        node: The CST node to process
        definitions: Dictionary of definition info

    Returns:
        (filtered_node, used_by_function):
          filtered_node: The modified CST node or None if it should be removed
          used_by_function: True if this node or any child is used by qualified functions

    """
    # Skip import statements
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return node, True

    # Never remove function definitions
    if isinstance(node, cst.FunctionDef):
        return node, True

    # Never remove class definitions
    if isinstance(node, cst.ClassDef):
        class_name = node.name.value

        # Check if any methods or variables in this class are used
        method_or_var_used = False
        class_has_dependencies = False

        # Check if class itself is marked as used
        if class_name in definitions and definitions[class_name].used_by_qualified_function:
            class_has_dependencies = True

        if hasattr(node, "body") and isinstance(node.body, cst.IndentedBlock):
            updates = {}
            new_statements = []

            for statement in node.body.body:
                # Keep all function definitions
                if isinstance(statement, cst.FunctionDef):
                    method_name = f"{class_name}.{statement.name.value}"
                    if method_name in definitions and definitions[method_name].used_by_qualified_function:
                        method_or_var_used = True
                    new_statements.append(statement)
                # Only process variable assignments
                elif isinstance(statement, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
                    var_used = False

                    # Check if any variable in this assignment is used
                    if isinstance(statement, cst.Assign):
                        for target in statement.targets:
                            names = extract_names_from_targets(target.target)
                            for name in names:
                                class_var_name = f"{class_name}.{name}"
                                if class_var_name in definitions and definitions[class_var_name].used_by_qualified_function:
                                    var_used = True
                                    method_or_var_used = True
                                    break
                    elif isinstance(statement, (cst.AnnAssign, cst.AugAssign)):
                        names = extract_names_from_targets(statement.target)
                        for name in names:
                            class_var_name = f"{class_name}.{name}"
                            if class_var_name in definitions and definitions[class_var_name].used_by_qualified_function:
                                var_used = True
                                method_or_var_used = True
                                break

                    if var_used or class_has_dependencies:
                        new_statements.append(statement)
                else:
                    # Keep all other statements in the class
                    new_statements.append(statement)

            # Update the class body
            new_body = node.body.with_changes(body=new_statements)
            updates["body"] = new_body

            return node.with_changes(**updates), True

        return node, method_or_var_used or class_has_dependencies

    # Handle assignments (Assign and AnnAssign)
    if isinstance(node, cst.Assign):
        for target in node.targets:
            names = extract_names_from_targets(target.target)
            for name in names:
                if name in definitions and definitions[name].used_by_qualified_function:
                    return node, True
        return None, False

    if isinstance(node, (cst.AnnAssign, cst.AugAssign)):
        names = extract_names_from_targets(node.target)
        for name in names:
            if name in definitions and definitions[name].used_by_qualified_function:
                return node, True
        return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates = {}
    found_used = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_used = False

            for child in original_content:
                filtered, used = remove_unused_definitions_recursively(child, definitions)
                if filtered:
                    new_children.append(filtered)
                section_found_used |= used

            if new_children or section_found_used:
                found_used |= section_found_used
                updates[section] = new_children
        elif original_content is not None:
            filtered, used = remove_unused_definitions_recursively(original_content, definitions)
            found_used |= used
            if filtered:
                updates[section] = filtered
    if not found_used:
        return None, False
    if updates:
        return node.with_changes(**updates), found_used

    return node, False


def remove_unused_definitions_by_function_names(code: str, qualified_function_names: set[str]) -> str:
    """Analyze a file and remove top level definitions not used by specified functions.

    Top level definitions, in this context, are only classes, variables or functions.
    If a class is referenced by a qualified function, we keep the entire class.

    Args:
        code: The code to process
        qualified_function_names: Set of function names to keep. For methods, use format 'classname.methodname'

    """
    module = cst.parse_module(code)
    # Collect all definitions (top level classes, variables or function)
    definitions = collect_top_level_definitions(module)

    # Collect dependencies between definitions using the visitor pattern
    dependency_collector = DependencyCollector(definitions)
    module.visit(dependency_collector)

    # Mark definitions used by specified functions, and their dependencies recursively
    usage_marker = QualifiedFunctionUsageMarker(definitions, qualified_function_names)
    usage_marker.mark_used_definitions()

    # Apply the recursive removal transformation
    modified_module, _ = remove_unused_definitions_recursively(module, definitions)

    return modified_module.code if modified_module else ""


def print_definitions(definitions: dict[str, UsageInfo]) -> None:
    """Print information about each definition without the complex node object, used for debugging."""
    print(f"Found {len(definitions)} definitions:")
    for name, info in sorted(definitions.items()):
        print(f"  - Name: {name}")
        print(f"    Used by qualified function: {info.used_by_qualified_function}")
        print(f"    Dependencies: {', '.join(sorted(info.dependencies)) if info.dependencies else 'None'}")
        print()
