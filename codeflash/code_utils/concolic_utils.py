from __future__ import annotations

import ast
import re
import subprocess
import uuid
from typing import Optional

import sentry_sdk

from codeflash.code_utils.compat import get_codeflash_temp_dir, get_safe_sys_executable


def is_valid_concolic_test(test_code: str, project_root: Optional[str] = None) -> bool:
    try:
        ast.parse(test_code)
    except SyntaxError:
        sentry_sdk.capture_message(f"CrossHair generated test with syntax error:\n{test_code}")
        return False

    temp_path = (get_codeflash_temp_dir() / f"concolic_test_{uuid.uuid4().hex}.py").resolve()
    temp_path.write_text(test_code, encoding="utf-8")

    try:
        result = subprocess.run(
            [get_safe_sys_executable(), "-m", "pytest", "--collect-only", "-q", temp_path.as_posix()],
            check=False,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, Exception):
        return False
    else:
        return result.returncode == 0
    finally:
        temp_path.unlink(missing_ok=True)


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
