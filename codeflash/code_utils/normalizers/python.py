"""Python code normalizer using AST transformation."""

from __future__ import annotations

import ast

from codeflash.code_utils.normalizers.base import CodeNormalizer


class VariableNormalizer(ast.NodeTransformer):
    """Normalizes only local variable names in AST to canonical forms like var_0, var_1, etc.

    Preserves function names, class names, parameters, built-ins, and imported names.
    """

    def __init__(self) -> None:
        self.var_counter = 0
        self.var_mapping: dict[str, str] = {}
        self.scope_stack: list[dict] = []
        self.builtins = set(dir(__builtins__))
        self.imports: set[str] = set()
        self.global_vars: set[str] = set()
        self.nonlocal_vars: set[str] = set()
        self.parameters: set[str] = set()

    def enter_scope(self) -> None:
        """Enter a new scope (function/class)."""
        self.scope_stack.append(
            {"var_mapping": dict(self.var_mapping), "var_counter": self.var_counter, "parameters": set(self.parameters)}
        )

    def exit_scope(self) -> None:
        """Exit current scope and restore parent scope."""
        if self.scope_stack:
            scope = self.scope_stack.pop()
            self.var_mapping = scope["var_mapping"]
            self.var_counter = scope["var_counter"]
            self.parameters = scope["parameters"]

    def get_normalized_name(self, name: str) -> str:
        """Get or create normalized name for a variable."""
        if (
            name in self.builtins
            or name in self.imports
            or name in self.global_vars
            or name in self.nonlocal_vars
            or name in self.parameters
        ):
            return name

        if name not in self.var_mapping:
            self.var_mapping[name] = f"var_{self.var_counter}"
            self.var_counter += 1
        return self.var_mapping[name]

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Track imported names."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name.split(".")[0])
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Track imported names from modules."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
        return node

    def visit_Global(self, node: ast.Global) -> ast.Global:
        """Track global variable declarations."""
        self.global_vars.update(node.names)
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        """Track nonlocal variable declarations."""
        self.nonlocal_vars.update(node.names)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Process function but keep function name and parameters unchanged."""
        self.enter_scope()

        for arg in node.args.args:
            self.parameters.add(arg.arg)
        if node.args.vararg:
            self.parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.parameters.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            self.parameters.add(arg.arg)

        node = self.generic_visit(node)
        self.exit_scope()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Handle async functions same as regular functions."""
        return self.visit_FunctionDef(node)  # type: ignore[return-value]

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Process class but keep class name unchanged."""
        self.enter_scope()
        node = self.generic_visit(node)
        self.exit_scope()
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Normalize variable names in Name nodes."""
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if (
                node.id not in self.builtins
                and node.id not in self.imports
                and node.id not in self.parameters
                and node.id not in self.global_vars
                and node.id not in self.nonlocal_vars
            ):
                node.id = self.get_normalized_name(node.id)
        elif isinstance(node.ctx, ast.Load):
            if node.id in self.var_mapping:
                node.id = self.var_mapping[node.id]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        """Normalize exception variable names."""
        if node.name:
            node.name = self.get_normalized_name(node.name)
        return self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> ast.comprehension:
        """Normalize comprehension target variables."""
        old_mapping = dict(self.var_mapping)
        old_counter = self.var_counter

        node = self.generic_visit(node)

        self.var_mapping = old_mapping
        self.var_counter = old_counter
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        """Handle for loop target variables."""
        return self.generic_visit(node)

    def visit_With(self, node: ast.With) -> ast.With:
        """Handle with statement as variables."""
        return self.generic_visit(node)


def _remove_docstrings_from_ast(node: ast.AST) -> None:
    """Remove docstrings from AST nodes."""
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
    stack = [node]
    while stack:
        current_node = stack.pop()
        if isinstance(current_node, node_types):
            body = current_node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                current_node.body = body[1:]
            stack.extend([child for child in body if isinstance(child, node_types)])


class PythonNormalizer(CodeNormalizer):
    """Python code normalizer using AST transformation.

    Normalizes Python code by:
    - Replacing local variable names with canonical forms (var_0, var_1, etc.)
    - Preserving function names, class names, parameters, and imports
    - Optionally removing docstrings
    """

    @property
    def language(self) -> str:
        """Return the language this normalizer handles."""
        return "python"

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Return file extensions this normalizer can handle."""
        return (".py", ".pyw", ".pyi")

    def normalize(self, code: str, remove_docstrings: bool = True) -> str:
        """Normalize Python code to a canonical form.

        Args:
            code: Python source code to normalize
            remove_docstrings: Whether to remove docstrings

        Returns:
            Normalized Python code as a string

        """
        tree = ast.parse(code)

        if remove_docstrings:
            _remove_docstrings_from_ast(tree)

        normalizer = VariableNormalizer()
        normalized_tree = normalizer.visit(tree)
        ast.fix_missing_locations(normalized_tree)

        return ast.unparse(normalized_tree)

    def normalize_for_hash(self, code: str) -> str:
        """Normalize Python code optimized for hashing.

        Returns AST dump which is faster than unparsing.

        Args:
            code: Python source code to normalize

        Returns:
            AST dump string suitable for hashing

        """
        tree = ast.parse(code)
        _remove_docstrings_from_ast(tree)

        normalizer = VariableNormalizer()
        normalized_tree = normalizer.visit(tree)

        return ast.dump(normalized_tree, annotate_fields=False, include_attributes=False)
