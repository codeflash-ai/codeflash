from __future__ import annotations

import ast
import re
from typing import Optional


class AssertCleanup:
    def transform_asserts(self, code: str) -> str:
        lines = code.splitlines()
        result_lines = []

        for line in lines:
            transformed = self._transform_assert_line(line)
            if transformed is not None:
                result_lines.append(transformed)
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

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
        delimiters = {",": 0, "(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}

        for char in args_str:
            if char in delimiters:
                depth += delimiters[char]
                if char == "," and depth == 0:
                    result.append("".join(current).strip())
                    current = []
                else:
                    current.append(char)
            else:
                current.append(char)

        if current:
            result.append("".join(current).strip())

        return result


def clean_concolic_tests(test_suite_code: str) -> str:
    try:
        can_parse = True
        tree = ast.parse(test_suite_code)
    except SyntaxError:
        can_parse = False

    if not can_parse:
        return AssertCleanup().transform_asserts(test_suite_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assert):
                    if isinstance(stmt.test, ast.Compare) and isinstance(stmt.test.left, ast.Call):
                        new_body.append(ast.Expr(value=stmt.test.left))
                    else:
                        new_body.append(stmt)

                else:
                    new_body.append(stmt)
            node.body = new_body

    return ast.unparse(tree).strip()
