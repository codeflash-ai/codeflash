"""LibCST visitor to find function definitions that call a specific qualified function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import libcst as cst


@dataclass
class FunctionCallLocation:
    """Represents a location where the target function is called."""

    calling_function: str  # Name of the function making the call
    line: int
    column: int
    call_node: cst.Call  # The actual call node for additional analysis if needed


@dataclass
class FunctionDefinitionInfo:
    """Contains information about a function definition."""

    name: str  # Qualified name of the function
    node: cst.FunctionDef  # The CST node of the function definition
    source_code: str  # The source code of the function
    start_line: int
    end_line: int
    is_method: bool  # Whether this is a class method
    class_name: Optional[str] = None  # Name of containing class if it's a method


class FunctionCallFinder(cst.CSTVisitor):
    """Visitor that finds all function definitions that call a specific qualified function.

    Args:
        target_function_name: The qualified name of the function to find (e.g., "module.function" or "function")
        target_filepath: The filepath where the target function is defined

    """

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, target_function_name: str, target_filepath: str) -> None:
        super().__init__()
        self.target_function_name = target_function_name
        self.target_filepath = target_filepath

        # Parse the target function name into parts
        self.target_parts = target_function_name.split(".")
        self.target_base_name = self.target_parts[-1]

        # Track current context
        self.current_function_stack: list[Tuple[str, cst.FunctionDef]] = []  # (name, node) pairs
        self.current_class_stack: list[str] = []

        # Track imports to resolve qualified names
        self.imports: dict = {}  # Maps imported names to their full paths

        # Results
        self.function_calls: list[FunctionCallLocation] = []
        self.calling_functions: set[str] = set()  # Unique function names that call the target
        self.function_definitions: Dict[str, FunctionDefinitionInfo] = {}  # Function name -> definition info

        # Track if we found calls in the current function
        self.found_call_in_current_function = False
        # Track functions with nested calls (parent functions that contain nested functions with calls)
        self.functions_with_nested_calls: set[str] = set()

    def visit_Import(self, node: cst.Import) -> None:
        """Track regular imports."""
        for name in node.names:
            if isinstance(name, cst.ImportAlias):
                if name.asname:
                    # import module as alias
                    module_name = name.name.value if isinstance(name.name, cst.Attribute) else str(name.name)
                    alias = name.asname.name.value
                    self.imports[alias] = module_name
                else:
                    # import module
                    module_name = self._get_dotted_name(name.name)
                    if module_name:
                        self.imports[module_name.split(".")[-1]] = module_name

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """Track from imports."""
        if not node.module:
            return

        module_path = self._get_dotted_name(node.module)
        if not module_path:
            return

        if isinstance(node.names, cst.ImportStar):
            # from module import *
            self.imports["*"] = module_path
        else:
            # from module import name1, name2
            for name in node.names:
                if isinstance(name, cst.ImportAlias):
                    import_name = name.name.value
                    if name.asname:
                        # from module import name as alias
                        alias = name.asname.name.value
                        self.imports[alias] = f"{module_path}.{import_name}"
                    else:
                        # from module import name
                        self.imports[import_name] = f"{module_path}.{import_name}"

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track when entering a class definition."""
        self.current_class_stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        """Track when leaving a class definition."""
        if self.current_class_stack:
            self.current_class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track when entering a function definition."""
        func_name = node.name.value

        # Build the full qualified name including class if applicable
        full_name = f"{'.'.join(self.current_class_stack)}.{func_name}" if self.current_class_stack else func_name

        self.current_function_stack.append((full_name, node))
        self.found_call_in_current_function = False

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track when leaving a function definition and store it if it contains target calls."""
        if self.current_function_stack:
            full_name, func_node = self.current_function_stack.pop()

            # If we found a call in this function, store its definition
            if self.found_call_in_current_function and full_name not in self.function_definitions:
                # Get position information
                position = self.get_metadata(cst.metadata.PositionProvider, func_node)

                # Extract function source code by converting node to module
                # For methods, we need to maintain proper indentation
                func_source = cst.Module(body=[func_node]).code

                # For methods, add proper indentation (4 spaces)
                if self.current_class_stack:
                    lines = func_source.split('\n')
                    func_source = '\n'.join('    ' + line if line else line for line in lines)

                self.function_definitions[full_name] = FunctionDefinitionInfo(
                    name=full_name,
                    node=func_node,
                    source_code=func_source.rstrip(),  # Remove trailing whitespace
                    start_line=position.start.line if position else -1,
                    end_line=position.end.line if position else -1,
                    is_method=bool(self.current_class_stack),
                    class_name=self.current_class_stack[-1] if self.current_class_stack else None
                )

            # Handle nested functions - mark parent as containing nested calls
            if self.found_call_in_current_function and self.current_function_stack:
                parent_name = self.current_function_stack[-1][0]
                self.functions_with_nested_calls.add(parent_name)
                # Also store the parent function if not already stored
                if parent_name not in self.function_definitions:
                    parent_func_node = self.current_function_stack[-1][1]
                    parent_position = self.get_metadata(cst.metadata.PositionProvider, parent_func_node)
                    parent_source = cst.Module(body=[parent_func_node]).code

                    # Get parent class context (go up one level in stack since we're inside the nested function)
                    parent_class_stack = self.current_class_stack[:-1] if len(self.current_function_stack) == 1 and self.current_class_stack else []

                    if parent_class_stack:
                        lines = parent_source.split('\n')
                        parent_source = '\n'.join('    ' + line if line else line for line in lines)

                    self.function_definitions[parent_name] = FunctionDefinitionInfo(
                        name=parent_name,
                        node=parent_func_node,
                        source_code=parent_source.rstrip(),
                        start_line=parent_position.start.line if parent_position else -1,
                        end_line=parent_position.end.line if parent_position else -1,
                        is_method=bool(parent_class_stack),
                        class_name=parent_class_stack[-1] if parent_class_stack else None
                    )

            # Reset the flag for parent function if we're in nested functions
            if self.current_function_stack:
                # Check if the parent function should also be marked as containing calls
                parent_name = self.current_function_stack[-1][0]
                self.found_call_in_current_function = parent_name in self.calling_functions

    def visit_Call(self, node: cst.Call) -> None:
        """Check if this call matches our target function."""
        if not self.current_function_stack:
            # Not inside a function, skip
            return

        if self._is_target_function_call(node):
            # Get position information
            position = self.get_metadata(cst.metadata.PositionProvider, node)

            current_func_name = self.current_function_stack[-1][0]

            call_location = FunctionCallLocation(
                calling_function=current_func_name,
                line=position.start.line if position else -1,
                column=position.start.column if position else -1,
                call_node=node,
            )

            self.function_calls.append(call_location)
            self.calling_functions.add(current_func_name)
            self.found_call_in_current_function = True

    def _is_target_function_call(self, node: cst.Call) -> bool:
        """Determine if this call node is calling our target function.

        Handles various call patterns:
        - Direct calls: function()
        - Qualified calls: module.function()
        - Method calls: obj.method()
        """
        func = node.func

        # Get the call name
        call_name = self._get_call_name(func)
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
                # Check if the imported path matches our target
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

    def _get_call_name(self, func: Union[cst.Name, cst.Attribute, cst.Call]) -> Optional[str]:
        """Extract the name being called from a function node."""
        if isinstance(func, cst.Name):
            return func.value
        if isinstance(func, cst.Attribute):
            return self._get_dotted_name(func)
        if isinstance(func, cst.Call):
            # Chained calls like foo()()
            return None
        return None

    def _get_dotted_name(self, node: Union[cst.Name, cst.Attribute]) -> Optional[str]:
        """Get the full dotted name from an Attribute or Name node."""
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parts = []
            current = node
            while isinstance(current, cst.Attribute):
                parts.append(current.attr.value)
                current = current.value
            if isinstance(current, cst.Name):
                parts.append(current.value)
                return ".".join(reversed(parts))
        return None

    def get_results(self) -> Dict[str, str]:
        """Get the results of the analysis.

        Returns:
            A dictionary mapping qualified function names to their source code definitions.
            Only includes functions that call the target function (directly or through nested functions).

        """
        return {
            info.name: info.source_code
            for info in self.function_definitions.values()
        }


def find_function_calls(source_code: str, target_function_name: str, target_filepath: str) -> Dict[str, str]:
    """Find all function definitions that call a specific target function.

    Args:
        source_code: The Python source code to analyze
        target_function_name: The qualified name of the function to find (e.g., "module.function")
        target_filepath: The filepath where the target function is defined

    Returns:
        A dictionary mapping qualified function names to their source code definitions.
        Example: {"function_a": "def function_a():\n    ...", "MyClass.method_one": "def method_one(self):\n    ..."}

    """
    # Parse the source code
    module = cst.parse_module(source_code)

    # Create and run the visitor
    visitor = FunctionCallFinder(target_function_name, target_filepath)
    wrapper = cst.metadata.MetadataWrapper(module)
    wrapper.visit(visitor)

    return visitor.get_results()


# Example usage
if __name__ == "__main__":
    # Example source code to analyze
    example_code = '''
import os
from pathlib import Path
from my_module import target_function as tf
import my_module

def function_a():
    """This function calls the target function directly."""
    result = tf(42)
    return result

def function_b():
    """This function calls the target function via module."""
    my_module.target_function("hello")

class MyClass:
    def method_one(self):
        """Method that calls the target."""
        tf(1, 2, 3)

    def method_two(self):
        """Method that doesn't call the target."""
        print("No call here")

def function_c():
    """This function doesn't call the target."""
    print("Just printing")

def nested_calls():
    """Function with nested function definitions."""
    def inner():
        tf("nested call")
    inner()
'''

    # Find calls to a specific function
    results = find_function_calls(
        example_code, target_function_name="my_module.target_function", target_filepath="/path/to/my_module.py"
    )

    print("Functions that call 'my_module.target_function':\n")

    # Simple usage - results is just a dict of {function_name: source_code}
    import json
    print("JSON representation of results:")
    print(json.dumps(list(results.keys()), indent=2))

    print("\nFormatted output:")
    for func_name, source_code in results.items():
        print(f"\n=== {func_name} ===")
        print(source_code)
        print()