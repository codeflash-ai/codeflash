import ast


class PytestRaisesRemover(ast.NodeTransformer):
    """Replaces 'with pytest.raises()' blocks with the content inside them."""

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.AST]:
        # Directly visit children and check if they are nested with blocks
        for item in node.items:
            if (
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Attribute)
                and isinstance(item.context_expr.func.value, ast.Name)
                and item.context_expr.func.value.id == "pytest"
                and item.context_expr.func.attr == "raises"
            ):
                # Return the body contents instead of the with block
                return self._unwrap_body(node.body)

        # Generic visit for other types of 'with' blocks
        return self.generic_visit(node)

    def _unwrap_body(self, body: list[ast.stmt]) -> ast.AST | list[ast.AST]:
        # Unwrap the body either as a single statement or a list of statements
        if len(body) == 1:
            return body[0]
        return body


def remove_pytest_raises(tree: ast.AST) -> ast.AST:
    """Removes pytest.raises blocks and shifts their content out.

    Args:
        tree: The AST tree to transform

    Returns:
        The transformed AST with pytest.raises blocks removed

    """
    transformer = PytestRaisesRemover()
    return transformer.visit(tree)
