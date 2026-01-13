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
            result_lines.append(transformed if transformed is not None else line)

        return "\n".join(result_lines)

    def _transform_assert_line(self, line: str) -> Optional[str]:
        indent = line[: len(line) - len(line.lstrip())]

        assert_match = self.assert_re.match(line)
        if assert_match:
            expression = assert_match.group(1).strip()
            if expression.startswith("not "):
                return f"{indent}{expression}"

            expression = expression.rstrip(",;")
            return f"{indent}{expression}"

        unittest_match = self.unittest_re.match(line)
        if unittest_match:
            indent, assert_method, args = unittest_match.groups()

            if args:
                arg_parts = self._first_top_level_arg(args)
                if arg_parts:
                    return f"{indent}{arg_parts}"

        return None

    def __init__(self) -> None:
        # Pre-compiling regular expressions for faster execution
        self.assert_re = re.compile(r"\s*assert\s+(.*?)(?:\s*==\s*.*)?$")
        self.unittest_re = re.compile(r"(\s*)self\.assert([A-Za-z]+)\((.*)\)$")

    def _first_top_level_arg(self, args: str) -> str:
        depth = 0
        for i, ch in enumerate(args):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                return args[:i].strip()
        return args.strip()


def clean_concolic_tests(test_suite_code: str) -> str:
    try:
        tree = ast.parse(test_suite_code)
        can_parse = True
    except Exception:
        can_parse = False
        tree = None

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
