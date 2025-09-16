import ast
import hashlib


class VariableNormalizer(ast.NodeTransformer):
    """Normalizes only local variable names in AST to canonical forms like var_0, var_1, etc.

    Preserves function names, class names, parameters, built-ins, and imported names.
    """

    def __init__(self) -> None:
        self.var_counter = 0
        self.var_mapping: dict[str, str] = {}
        self.scope_stack = []
        self.builtins = set(dir(__builtins__))
        self.imports: set[str] = set()
        self.global_vars: set[str] = set()
        self.nonlocal_vars: set[str] = set()
        self.parameters: set[str] = set()  # Track function parameters

    def enter_scope(self):  # noqa : ANN201
        """Enter a new scope (function/class)."""
        self.scope_stack.append(
            {"var_mapping": dict(self.var_mapping), "var_counter": self.var_counter, "parameters": set(self.parameters)}
        )

    def exit_scope(self):  # noqa : ANN201
        """Exit current scope and restore parent scope."""
        if self.scope_stack:
            scope = self.scope_stack.pop()
            self.var_mapping = scope["var_mapping"]
            self.var_counter = scope["var_counter"]
            self.parameters = scope["parameters"]

    def get_normalized_name(self, name: str) -> str:
        """Get or create normalized name for a variable."""
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

    def visit_Import(self, node):  # noqa : ANN001, ANN201
        """Track imported names."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name.split(".")[0])
        return node

    def visit_ImportFrom(self, node):  # noqa : ANN001, ANN201
        """Track imported names from modules."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
        return node

    def visit_Global(self, node):  # noqa : ANN001, ANN201
        """Track global variable declarations."""
        # Avoid repeated .add calls by using set.update with list
        self.global_vars.update(node.names)
        return node

    def visit_Nonlocal(self, node):  # noqa : ANN001, ANN201
        """Track nonlocal variable declarations."""
        # Using set.update for batch insertion (faster than add-in-loop)
        self.nonlocal_vars.update(node.names)
        return node

    def visit_FunctionDef(self, node):  # noqa : ANN001, ANN201
        """Process function but keep function name and parameters unchanged."""
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

    def visit_AsyncFunctionDef(self, node):  # noqa : ANN001, ANN201
        """Handle async functions same as regular functions."""
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):  # noqa : ANN001, ANN201
        """Process class but keep class name unchanged."""
        self.enter_scope()
        node = self.generic_visit(node)
        self.exit_scope()
        return node

    def visit_Name(self, node):  # noqa : ANN001, ANN201
        """Normalize variable names in Name nodes."""
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
        elif isinstance(node.ctx, ast.Load):  # noqa : SIM102
            # For loading, use existing mapping if available
            if node.id in self.var_mapping:
                node.id = self.var_mapping[node.id]
        return node

    def visit_ExceptHandler(self, node):  # noqa : ANN001, ANN201
        """Normalize exception variable names."""
        if node.name:
            node.name = self.get_normalized_name(node.name)
        return self.generic_visit(node)

    def visit_comprehension(self, node):  # noqa : ANN001, ANN201
        """Normalize comprehension target variables."""
        # Create new scope for comprehension
        old_mapping = dict(self.var_mapping)
        old_counter = self.var_counter

        # Process the comprehension
        node = self.generic_visit(node)

        # Restore scope
        self.var_mapping = old_mapping
        self.var_counter = old_counter
        return node

    def visit_For(self, node):  # noqa : ANN001, ANN201
        """Handle for loop target variables."""
        # The target in a for loop is a local variable that should be normalized
        return self.generic_visit(node)

    def visit_With(self, node):  # noqa : ANN001, ANN201
        """Handle with statement as variables."""
        return self.generic_visit(node)


def normalize_code(code: str, remove_docstrings: bool = True, return_ast_dump: bool = False) -> str:  # noqa : FBT002, FBT001
    """Normalize Python code by parsing, cleaning, and normalizing only variable names.

    Function names, class names, and parameters are preserved.

    Args:
        code: Python source code as string
        remove_docstrings: Whether to remove docstrings
        return_ast_dump: return_ast_dump

    Returns:
        Normalized code as string

    """
    try:
        # Parse the code
        tree = ast.parse(code)

        # Remove docstrings if requested
        if remove_docstrings:
            remove_docstrings_from_ast(tree)

        # Normalize variable names
        normalizer = VariableNormalizer()
        normalized_tree = normalizer.visit(tree)
        if return_ast_dump:
            # This is faster than unparsing etc
            return ast.dump(normalized_tree, annotate_fields=False, include_attributes=False)

        # Fix missing locations in the AST
        ast.fix_missing_locations(normalized_tree)

        # Unparse back to code
        return ast.unparse(normalized_tree)
    except SyntaxError as e:
        msg = f"Invalid Python syntax: {e}"
        raise ValueError(msg) from e


def remove_docstrings_from_ast(node):  # noqa : ANN001, ANN201
    """Remove docstrings from AST nodes."""
    # Only FunctionDef, AsyncFunctionDef, ClassDef, and Module can contain docstrings in their body[0]
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
    # Use our own stack-based DFS instead of ast.walk for efficiency
    stack = [node]
    while stack:
        current_node = stack.pop()
        if isinstance(current_node, node_types):
            # Remove docstring if it's the first stmt in body
            body = current_node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                current_node.body = body[1:]
            # Only these nodes can nest more docstring-containing nodes
            # Add their body elements to stack, avoiding unnecessary traversal
            stack.extend([child for child in body if isinstance(child, node_types)])


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
        normalized1 = normalize_code(code1, return_ast_dump=True)
        normalized2 = normalize_code(code2, return_ast_dump=True)
    except Exception:
        return False
    else:
        return normalized1 == normalized2
