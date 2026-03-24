from __future__ import annotations

import ast
from importlib.util import find_spec

has_numba = find_spec("numba") is not None

NUMERICAL_MODULES = frozenset({"numpy", "torch", "numba", "jax", "tensorflow", "math", "scipy"})
# Modules that require numba to be installed for optimization
NUMBA_REQUIRED_MODULES = frozenset({"numpy", "math", "scipy"})


class NumericalUsageChecker(ast.NodeVisitor):
    """AST visitor that checks if a function uses numerical computing libraries."""

    def __init__(self, numerical_names: set[str]) -> None:
        self.numerical_names = numerical_names
        self.found_numerical = False

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for numerical library usage."""
        if self.found_numerical:
            return
        call_name = self.get_root_name(node.func)
        if call_name and call_name in self.numerical_names:
            self.found_numerical = True
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for numerical library usage."""
        if self.found_numerical:
            return
        root_name = self.get_root_name(node)
        if root_name and root_name in self.numerical_names:
            self.found_numerical = True
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check name references for numerical library usage."""
        if self.found_numerical:
            return
        if node.id in self.numerical_names:
            self.found_numerical = True

    def get_root_name(self, node: ast.expr) -> str | None:
        """Get the root name from an expression (e.g., 'np' from 'np.array')."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self.get_root_name(node.value)
        return None


def collect_numerical_imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Collect names that reference numerical computing libraries from imports.

    Returns:
        A tuple of (numerical_names, modules_used) where:
        - numerical_names: set of names/aliases that reference numerical libraries
        - modules_used: set of actual module names (e.g., "numpy", "math") being imported

    """
    numerical_names: set[str] = set()
    modules_used: set[str] = set()

    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                # import numpy or import numpy as np
                module_root = alias.name.split(".")[0]
                if module_root in NUMERICAL_MODULES:
                    # Use the alias if present, otherwise the module name
                    name = alias.asname if alias.asname else alias.name.split(".")[0]
                    numerical_names.add(name)
                    modules_used.add(module_root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_root = node.module.split(".")[0]
            if module_root in NUMERICAL_MODULES:
                # from numpy import array, zeros as z
                for alias in node.names:
                    if alias.name == "*":
                        # Can't track star imports, but mark the module as numerical
                        numerical_names.add(module_root)
                    else:
                        name = alias.asname if alias.asname else alias.name
                        numerical_names.add(name)
                modules_used.add(module_root)
        else:
            stack.extend(ast.iter_child_nodes(node))

    return numerical_names, modules_used


def find_function_node(tree: ast.Module, name_parts: list[str]) -> ast.FunctionDef | None:
    """Find a function node in the AST given its qualified name parts.

    Note: This function only finds regular (sync) functions, not async functions.

    Args:
        tree: The parsed AST module
        name_parts: List of name parts, e.g., ["ClassName", "method_name"] or ["function_name"]

    Returns:
        The function node if found, None otherwise

    """
    if not name_parts:
        return None

    if len(name_parts) == 1:
        # Top-level function
        func_name = name_parts[0]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None

    if len(name_parts) == 2:
        # Class method: ClassName.method_name
        class_name, method_name = name_parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef) and class_node.name == method_name:
                        return class_node
        return None

    return None


def is_numerical_code(code_string: str, function_name: str | None = None) -> bool:
    """Check if a function uses numerical computing libraries.

    Detects usage of numpy, torch, numba, jax, tensorflow, scipy, and math libraries
    within the specified function.

    Note: For math, numpy, and scipy usage, this function returns True only if numba
    is installed in the environment, as numba is required to optimize such code.

    Args:
        code_string: The entire file's content as a string
        function_name: The name of the function to check. Can be a simple name like "foo"
                      or a qualified name like "ClassName.method_name" for methods,
                      staticmethods, or classmethods.

    Returns:
        True if the function uses any numerical computing library functions, False otherwise.
        Returns False for math/numpy/scipy usage if numba is not installed.

    Examples:
        >>> code = '''
        ... import numpy as np
        ... def process_data(x):
        ...     return np.sum(x)
        ... '''
        >>> is_numerical_code(code, "process_data")  # Returns True only if numba is installed
        True

        >>> code = '''
        ... def simple_func(x):
        ...     return x + 1
        ... '''
        >>> is_numerical_code(code, "simple_func")
        False

    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return False

    # Collect names that reference numerical modules from imports
    numerical_names, modules_used = collect_numerical_imports(tree)

    if not function_name:
        # Return True if modules used and (numba available or modules don't all require numba)
        return bool(modules_used) and (has_numba or not modules_used.issubset(NUMBA_REQUIRED_MODULES))

    # Split the function name to handle class methods
    name_parts = function_name.split(".")

    # Find the target function node
    target_function = find_function_node(tree, name_parts)
    if target_function is None:
        return False

    # Check if the function body uses any numerical library
    checker = NumericalUsageChecker(numerical_names)
    checker.visit(target_function)

    if not checker.found_numerical:
        return False

    # If numba is not installed and all modules used require numba for optimization,
    # return False since we can't optimize this code
    return not (not has_numba and modules_used.issubset(NUMBA_REQUIRED_MODULES))
