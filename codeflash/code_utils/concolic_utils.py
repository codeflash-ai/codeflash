from __future__ import annotations

import ast
import re
from typing import Optional


class AssertCleanup:
    def transform_asserts(self, code: str) -> str:
        lines = code.splitlines()
        transformed_lines = []

        for line in lines:
            transformed = self._transform_assert_line(line)
            if transformed is not None:
                transformed_lines.append(transformed)
            else:
                transformed_lines.append(line)

        return "\n".join(transformed_lines)

    def _transform_assert_line(self, line: str) -> Optional[str]:
        indent = line[: len(line) - len(line.lstrip())]

        assert_match = re.match(r"\s*assert\s+(.*?)(?:\s*==\s*.*)?$", line)
        if assert_match:
            expression = assert_match.group(1).strip()
            if expression.startswith("not "):
                return f"{indent}{expression}"

            expression = re.sub(r"[,;]\s*$", "", expression)
            return f"{indent}{expression}"

        unittest_match = re.match(r"(\s*)self\.assert([A-Za-z]+)\((.*)\)$", line)
        if unittest_match:
            indent, assert_method, args = unittest_match.groups()

            if args:
                arg_parts = self._split_top_level_args(args)
                if arg_parts and arg_parts[0]:
                    return f"{indent}{arg_parts[0]}"

        return None

    def _split_top_level_args(self, args_str: str) -> list[str]:
        result = []
        current = []
        depth = 0

        for char in args_str:
            if char in "([{":
                depth += 1
                current.append(char)
            elif char in ")]}":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            result.append("".join(current).strip())

        return result


def clean_concolic_tests(test_suite_code: str) -> str:
    try:
        tree = ast.parse(test_suite_code)
        can_parse = True
    except SyntaxError:
        can_parse = False

    if not can_parse:
        return AssertCleanup().transform_asserts(test_suite_code)

    class AssertTransform(ast.NodeTransformer):
        def visit_Assert(self, node):
            if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Call):
                return ast.Expr(value=node.test.left, lineno=node.lineno, col_offset=node.col_offset)
            return node

        def visit_FunctionDef(self, node):
            if node.name.startswith("test_"):
                node.body = [self.visit(stmt) for stmt in node.body]
            return node

    transformer = AssertTransform()
    transformer.visit(tree)
    return ast.unparse(tree).strip()
