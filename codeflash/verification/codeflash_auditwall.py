import ast


class AuditWallTransformer(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        last_import_index = -1
        for i, body_node in enumerate(node.body):
            if isinstance(body_node, (ast.Import, ast.ImportFrom)):
                last_import_index = i

        new_import = ast.ImportFrom(
            module="auditwall.core", names=[ast.alias(name="engage_auditwall")], level=0
        )
        function_call = ast.Expr(
            value=ast.Call(func=ast.Name(id="engage_auditwall", ctx=ast.Load()), args=[], keywords=[])
        )

        node.body.insert(last_import_index + 1, new_import)
        node.body.insert(last_import_index + 2, function_call)

        return node

def transform_code(source_code: str) -> str:
    tree = ast.parse(source_code)
    transformer = AuditWallTransformer()
    new_tree = transformer.visit(tree)
    return ast.unparse(new_tree)
