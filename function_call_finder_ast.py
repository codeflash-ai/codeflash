"""AST-based visitor to find function definitions that call a specific qualified function."""

import ast
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


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

    def __init__(self, target_function_name: str, target_filepath: str, source_lines: List[str]):
        self.target_function_name = target_function_name
        self.target_filepath = target_filepath
        self.source_lines = source_lines  # Store original source lines for extraction

        # Parse the target function name into parts
        self.target_parts = target_function_name.split('.')
        self.target_base_name = self.target_parts[-1]

        # Track current context
        self.current_function_stack: List[Tuple[str, ast.FunctionDef]] = []
        self.current_class_stack: List[str] = []

        # Track imports to resolve qualified names
        self.imports: Dict[str, str] = {}  # Maps imported names to their full paths

        # Results
        self.function_calls: List[FunctionCallLocation] = []
        self.calling_functions: Set[str] = set()
        self.function_definitions: Dict[str, FunctionDefinitionInfo] = {}

        # Track if we found calls in the current function
        self.found_call_in_current_function = False
        self.functions_with_nested_calls: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Track regular imports."""
        for alias in node.names:
            if alias.asname:
                # import module as alias
                self.imports[alias.asname] = alias.name
            else:
                # import module
                self.imports[alias.name.split('.')[-1]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from imports."""
        if node.module:
            for alias in node.names:
                if alias.name == '*':
                    # from module import *
                    self.imports['*'] = node.module
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
        """Common logic for both regular and async function definitions."""
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
                end_line=node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                is_method=bool(self.current_class_stack),
                class_name=self.current_class_stack[-1] if self.current_class_stack else None
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
                    end_line=parent_node.end_lineno if hasattr(parent_node, 'end_lineno') else parent_node.lineno,
                    is_method=bool(parent_class_context),
                    class_name=parent_class_context[-1] if parent_class_context else None
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
                calling_function=current_func_name,
                line=node.lineno,
                column=node.col_offset
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
                if imported_path == self.target_function_name or imported_path.endswith(f".{self.target_function_name}"):
                    return True
            # Could also be a direct call if we're in the same file
            return True

        # Check for qualified calls with imports
        call_parts = call_name.split('.')
        if call_parts[0] in self.imports:
            # Resolve the full path using imports
            base_import = self.imports[call_parts[0]]
            full_path = f"{base_import}.{'.'.join(call_parts[1:])}" if len(call_parts) > 1 else base_import

            if full_path == self.target_function_name or full_path.endswith(f".{self.target_function_name}"):
                return True

        return False

    def _get_call_name(self, func_node) -> Optional[str]:
        """Extract the name being called from a function node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            parts = []
            current = func_node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return '.'.join(reversed(parts))
        return None

    def _extract_source_code(self, node: ast.FunctionDef) -> str:
        """Extract source code for a function node using original source lines."""
        if not self.source_lines or not hasattr(node, 'lineno'):
            # Fallback to ast.unparse if available (Python 3.9+)
            try:
                return ast.unparse(node)
            except AttributeError:
                return f"# Source code extraction not available for {node.name}"

        # Get the lines for this function
        start_line = node.lineno - 1  # Convert to 0-based index
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(self.source_lines)

        # Extract the function lines
        func_lines = self.source_lines[start_line:end_line]

        # Find the minimum indentation (excluding empty lines)
        min_indent = float('inf')
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

        return ''.join(result_lines).rstrip()

    def get_results(self) -> Dict[str, str]:
        """Get the results of the analysis.

        Returns:
            A dictionary mapping qualified function names to their source code definitions.
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
    tree = ast.parse(source_code)

    # Split source into lines for source extraction
    source_lines = source_code.splitlines(keepends=True)

    # Create and run the visitor
    visitor = FunctionCallFinder(target_function_name, target_filepath, source_lines)
    visitor.visit(tree)

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
        example_code,
        target_function_name="my_module.target_function",
        target_filepath="/path/to/my_module.py"
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