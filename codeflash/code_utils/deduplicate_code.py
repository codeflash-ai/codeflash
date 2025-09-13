import ast
import hashlib
from typing import Dict, Set


class VariableNormalizer(ast.NodeTransformer):
    """Normalizes only local variable names in AST to canonical forms like var_0, var_1, etc.
    Preserves function names, class names, parameters, built-ins, and imported names.
    """

    def __init__(self):
        self.var_counter = 0
        self.var_mapping: Dict[str, str] = {}
        self.scope_stack = []
        self.builtins = set(dir(__builtins__))
        self.imports: Set[str] = set()
        self.global_vars: Set[str] = set()
        self.nonlocal_vars: Set[str] = set()
        self.parameters: Set[str] = set()  # Track function parameters

    def enter_scope(self):
        """Enter a new scope (function/class)"""
        self.scope_stack.append(
            {"var_mapping": dict(self.var_mapping), "var_counter": self.var_counter, "parameters": set(self.parameters)}
        )

    def exit_scope(self):
        """Exit current scope and restore parent scope"""
        if self.scope_stack:
            scope = self.scope_stack.pop()
            self.var_mapping = scope["var_mapping"]
            self.var_counter = scope["var_counter"]
            self.parameters = scope["parameters"]

    def get_normalized_name(self, name: str) -> str:
        """Get or create normalized name for a variable"""
        # Don't normalize if it's a builtin, import, global, nonlocal, or parameter
        if (
            name in self.builtins
            or name in self.imports
            or name in self.global_vars
            or name in self.nonlocal_vars
            or name in self.parameters
        ):
            return name

        # Only normalize local variables
        if name not in self.var_mapping:
            self.var_mapping[name] = f"var_{self.var_counter}"
            self.var_counter += 1
        return self.var_mapping[name]

    def visit_Import(self, node):
        """Track imported names"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name.split(".")[0])
        return node

    def visit_ImportFrom(self, node):
        """Track imported names from modules"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
        return node

    def visit_Global(self, node):
        """Track global variable declarations"""
        # Avoid repeated .add calls by using set.update with list
        self.global_vars.update(node.names)
        return node

    def visit_Nonlocal(self, node):
        """Track nonlocal variable declarations"""
        for name in node.names:
            self.nonlocal_vars.add(name)
        return node

    def visit_FunctionDef(self, node):
        """Process function but keep function name and parameters unchanged"""
        self.enter_scope()

        # Track all parameters (don't modify them)
        for arg in node.args.args:
            self.parameters.add(arg.arg)
        if node.args.vararg:
            self.parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.parameters.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            self.parameters.add(arg.arg)

        # Visit function body
        node = self.generic_visit(node)
        self.exit_scope()
        return node

    def visit_AsyncFunctionDef(self, node):
        """Handle async functions same as regular functions"""
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Process class but keep class name unchanged"""
        self.enter_scope()
        node = self.generic_visit(node)
        self.exit_scope()
        return node

    def visit_Name(self, node):
        """Normalize variable names in Name nodes"""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            # For assignments and deletions, check if we should normalize
            if (
                node.id not in self.builtins
                and node.id not in self.imports
                and node.id not in self.parameters
                and node.id not in self.global_vars
                and node.id not in self.nonlocal_vars
            ):
                node.id = self.get_normalized_name(node.id)
        elif isinstance(node.ctx, ast.Load):
            # For loading, use existing mapping if available
            if node.id in self.var_mapping:
                node.id = self.var_mapping[node.id]
        return node

    def visit_ExceptHandler(self, node):
        """Normalize exception variable names"""
        if node.name:
            node.name = self.get_normalized_name(node.name)
        return self.generic_visit(node)

    def visit_comprehension(self, node):
        """Normalize comprehension target variables"""
        # Create new scope for comprehension
        old_mapping = dict(self.var_mapping)
        old_counter = self.var_counter

        # Process the comprehension
        node = self.generic_visit(node)

        # Restore scope
        self.var_mapping = old_mapping
        self.var_counter = old_counter
        return node

    def visit_For(self, node):
        """Handle for loop target variables"""
        # The target in a for loop is a local variable that should be normalized
        return self.generic_visit(node)

    def visit_With(self, node):
        """Handle with statement as variables"""
        # micro-optimization: directly call NodeTransformer's generic_visit (fewer indirections than type-based lookup)
        return ast.NodeTransformer.generic_visit(self, node)


def normalize_code(code: str, remove_docstrings: bool = True) -> str:
    """Normalize Python code by parsing, cleaning, and normalizing only variable names.
    Function names, class names, and parameters are preserved.

    Args:
        code: Python source code as string
        remove_docstrings: Whether to remove docstrings

    Returns:
        Normalized code as string

    """
    try:
        tree = ast.parse(code)

        # Fast-path: skip docstring removal step if not requested
        if remove_docstrings:
            # Inline deduplication logic from remove_docstrings_from_ast for performance;
            # replaces ast.walk() with iterative traversal for fewer allocations
            nodes = [tree]
            while nodes:
                node = nodes.pop()
                # Only consider def, async def, class, module nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    body = node.body
                    if body:
                        expr0 = body[0]
                        if (
                            isinstance(expr0, ast.Expr)
                            and isinstance(expr0.value, ast.Constant)
                            and isinstance(expr0.value.value, str)
                        ):
                            node.body = body[1:]
                    # Extend with children efficiently
                    nodes.extend(
                        child
                        for child in getattr(node, "body", [])
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    )

            # No need to import remove_docstrings_from_ast

        # VariableNormalizer usage as before
        normalizer = VariableNormalizer()
        normalized_tree = normalizer.visit(tree)

        # Fix missing locations efficiently
        # ast.fix_missing_locations does a deep recursive update; cannot optimize without breaking API
        ast.fix_missing_locations(normalized_tree)

        # ast.unparse is required; cannot avoid
        return ast.unparse(normalized_tree)
    except SyntaxError as e:
        msg = f"Invalid Python syntax: {e}"
        raise ValueError(msg) from e


def remove_docstrings_from_ast(node):
    """Remove docstrings from AST nodes."""
    # Process all nodes in the tree, but avoid recursion
    for current_node in ast.walk(node):
        if isinstance(current_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                current_node.body
                and isinstance(current_node.body[0], ast.Expr)
                and isinstance(current_node.body[0].value, ast.Constant)
                and isinstance(current_node.body[0].value.value, str)
            ):
                current_node.body = current_node.body[1:]


def get_code_fingerprint(code: str) -> str:
    """Generate a fingerprint for normalized code.

    Args:
        code: Python source code

    Returns:
        SHA-256 hash of normalized code

    """
    normalized = normalize_code(code)
    return hashlib.sha256(normalized.encode()).hexdigest()


def are_codes_duplicate(code1: str, code2: str) -> bool:
    """Check if two code segments are duplicates after normalization.

    Args:
        code1: First code segment
        code2: Second code segment

    Returns:
        True if codes are structurally identical (ignoring local variable names)

    """
    try:
        normalized1 = normalize_code(code1)
        normalized2 = normalize_code(code2)
        return normalized1 == normalized2
    except Exception:
        return False
