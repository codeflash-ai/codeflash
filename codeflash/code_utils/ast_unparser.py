import sys

if sys.version_info < (3, 9, 0):
    import astunparse
else:
    import ast


def ast_unparse(ast_obj) -> str:
    if sys.version_info < (3, 9, 0):
        return astunparse.ast_unparse(ast_obj)
    else:
        return ast.unparse(ast_obj)
