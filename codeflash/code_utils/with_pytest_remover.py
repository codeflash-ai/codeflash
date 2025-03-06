import ast
class PytestRaisesRemover(ast.NodeTransformer):
    """Replaces 'with pytest.raises()' blocks with the content inside them."""

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.AST]:
        # Process any nested with blocks first by recursively visiting children
        node = self.generic_visit(node)

        for item in node.items:
            # Check if this is a pytest.raises block
            if (isinstance(item.context_expr, ast.Call) and
                    isinstance(item.context_expr.func, ast.Attribute) and
                    isinstance(item.context_expr.func.value, ast.Name) and
                    item.context_expr.func.value.id == "pytest" and
                    item.context_expr.func.attr == "raises"):

                # Return the body contents instead of the with block
                # If there's multiple statements in the body, return them all
                if len(node.body) == 1:
                    return node.body[0]
                return node.body

        return node


def remove_pytest_raises(tree: ast.AST) -> ast.AST:
    """Removes pytest.raises blocks and shifts their content out.

    Args:
        tree: The AST tree to transform

    Returns:
        The transformed AST with pytest.raises blocks removed

    """
    transformer = PytestRaisesRemover()
    return transformer.visit(tree)
