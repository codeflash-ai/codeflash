from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.time_utils import format_perf, format_time
from codeflash.models.models import (GeneratedTests, GeneratedTestsList,
                                     InvocationId)
from codeflash.result.critic import performance_gain
from codeflash.verification.verification_utils import TestConfig

if TYPE_CHECKING:
    from codeflash.models.models import InvocationId
    from codeflash.verification.verification_utils import TestConfig


def remove_functions_from_generated_tests(
    generated_tests: GeneratedTestsList, test_functions_to_remove: list[str]
) -> GeneratedTestsList:
    new_generated_tests = []
    for generated_test in generated_tests.generated_tests:
        for test_function in test_functions_to_remove:
            function_pattern = re.compile(
                rf"(@pytest\.mark\.parametrize\(.*?\)\s*)?def\s+{re.escape(test_function)}\(.*?\):.*?(?=\ndef\s|$)",
                re.DOTALL,
            )

            match = function_pattern.search(generated_test.generated_original_test_source)

            if match is None or "@pytest.mark.parametrize" in match.group(0):
                continue

            generated_test.generated_original_test_source = function_pattern.sub(
                "", generated_test.generated_original_test_source
            )

        new_generated_tests.append(generated_test)

    return GeneratedTestsList(generated_tests=new_generated_tests)


class CfoVisitor(ast.NodeVisitor):
    """AST visitor that finds all assignments to a variable named 'codeflash_output'.

    and reports their location relative to the function they're in.
    """

    def __init__(self, qualified_name: str, source_code: str) -> None:
        self.source_lines = source_code.splitlines()
        self.name = qualified_name.split(".")[-1]
        self.results: list[int] = []  # map actual line number to line number in ast

    def visit_Call(self, node):  # type: ignore[no-untyped-def] # noqa: ANN201, ANN001
        """Detect fn calls."""
        func_name = self._get_called_func_name(node.func)  # type: ignore[no-untyped-call]
        if func_name == self.name:
            self.results.append(node.lineno - 1)
        self.generic_visit(node)

    def _get_called_func_name(self, node):  # type: ignore[no-untyped-def] # noqa: ANN001, ANN202
        """Return name of called fn."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None


def find_codeflash_output_assignments(qualified_name: str, source_code: str) -> list[int]:
    tree = ast.parse(source_code)
    visitor = CfoVisitor(qualified_name, source_code)
    visitor.visit(tree)
    return visitor.results


def add_runtime_comments_to_generated_tests(
    qualified_name: str,
    test_cfg: TestConfig,
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    tests_root = test_cfg.tests_root
    module_root = test_cfg.project_root_path
    rel_tests_root = tests_root.relative_to(module_root)

    # ---- Preindex invocation results for O(1) matching -------
    # (rel_path, qualified_name, cfo_loc) -> list[runtimes]
    def _make_index(invocations):
        index = {}
        for invocation_id, runtimes in invocations.items():
            test_class = invocation_id.test_class_name
            test_func = invocation_id.test_function_name
            q_name = f"{test_class}.{test_func}" if test_class else test_func
            rel_path = Path(invocation_id.test_module_path.replace(".", os.sep)).with_suffix(".py")
            # Defensive: sometimes path processing can fail, fallback to string
            try:
                rel_path = rel_path.relative_to(rel_tests_root)
            except Exception:
                rel_path = str(rel_path)
            # Get CFO location integer
            try:
                cfo_loc = int(invocation_id.iteration_id.split("_")[0])
            except Exception:
                cfo_loc = None
            key = (str(rel_path), q_name, cfo_loc)
            if key not in index:
                index[key] = []
            index[key].extend(runtimes)
        return index

    orig_index = _make_index(original_runtimes)
    opt_index = _make_index(optimized_runtimes)

    # Optimized fast CST visitor base
    class RuntimeCommentTransformer(cst.CSTTransformer):
        def __init__(
            self, qualified_name: str, module: cst.Module, test: GeneratedTests, tests_root: Path, rel_tests_root: Path
        ) -> None:
            super().__init__()
            self.test = test
            self.context_stack: list[str] = []
            self.tests_root = tests_root
            self.rel_tests_root = rel_tests_root
            self.module = module
            self.cfo_locs: list[int] = []
            self.cfo_idx_loc_to_look_at: int = -1
            self.name = qualified_name.split(".")[-1]
            # Precompute test-local file relative paths for efficiency
            self.test_rel_behavior = str(test.behavior_file_path.relative_to(tests_root))
            self.test_rel_perf = str(test.perf_file_path.relative_to(tests_root))

        def visit_ClassDef(self, node: cst.ClassDef) -> None:
            self.context_stack.append(node.name.value)

        def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
            self.context_stack.pop()
            return updated_node

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            # This could be optimized further if you access CFO assignments via CST
            body_code = dedent(self.module.code_for_node(node.body))
            normalized_body_code = ast.unparse(ast.parse(body_code))
            self.cfo_locs = sorted(find_codeflash_output_assignments(qualified_name, normalized_body_code))
            self.cfo_idx_loc_to_look_at = -1
            self.context_stack.append(node.name.value)

        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            self.context_stack.pop()
            return updated_node

        def leave_SimpleStatementLine(
            self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
        ) -> cst.SimpleStatementLine:
            # Fast skip before deep call tree walk by screening for Name nodes
            if self._contains_myfunc_call(updated_node):
                self.cfo_idx_loc_to_look_at += 1
                if self.cfo_idx_loc_to_look_at >= len(self.cfo_locs):
                    return updated_node  # Defensive, should never happen

                cfo_loc = self.cfo_locs[self.cfo_idx_loc_to_look_at]

                qualified_name_chain = ".".join(self.context_stack)
                # Try both behavior and perf as possible locations; both are strings
                possible_paths = {self.test_rel_behavior, self.test_rel_perf}

                # Form index key(s)
                matching_original = []
                matching_optimized = []

                for rel_path_str in possible_paths:
                    key = (rel_path_str, qualified_name_chain, cfo_loc)
                    if key in orig_index:
                        matching_original.extend(orig_index[key])
                    if key in opt_index:
                        matching_optimized.extend(opt_index[key])
                if matching_original and matching_optimized:
                    original_time = min(matching_original)
                    optimized_time = min(matching_optimized)
                    if original_time != 0 and optimized_time != 0:
                        perf_gain_str = format_perf(
                            abs(
                                performance_gain(original_runtime_ns=original_time, optimized_runtime_ns=optimized_time)
                                * 100
                            )
                        )
                        status = "slower" if optimized_time > original_time else "faster"
                        comment_text = f"# {format_time(original_time)} -> {format_time(optimized_time)} ({perf_gain_str}% {status})"
                        return updated_node.with_changes(
                            trailing_whitespace=cst.TrailingWhitespace(
                                whitespace=cst.SimpleWhitespace(" "),
                                comment=cst.Comment(comment_text),
                                newline=updated_node.trailing_whitespace.newline,
                            )
                        )
            return updated_node

        def _contains_myfunc_call(self, node):
            """Recursively search for any Call node in the statement whose function is named self.name (including obj.myfunc)."""

            # IMPORTANT micro-optimization: early abort using an exception
            class Found(Exception):
                pass

            class Finder(cst.CSTVisitor):
                def __init__(self, name):
                    self.name = name

                def visit_Call(self, call_node):
                    func_expr = call_node.func
                    if (isinstance(func_expr, cst.Name) and func_expr.value == self.name) or (
                        isinstance(func_expr, cst.Attribute) and func_expr.attr.value == self.name
                    ):
                        raise Found

            try:
                node.visit(Finder(self.name))
            except Found:
                return True
            return False

    modified_tests = []
    for test in generated_tests.generated_tests:
        try:
            tree = cst.parse_module(test.generated_original_test_source)
            transformer = RuntimeCommentTransformer(qualified_name, tree, test, tests_root, rel_tests_root)
            modified_tree = tree.visit(transformer)
            modified_source = modified_tree.code
            modified_test = GeneratedTests(
                generated_original_test_source=modified_source,
                instrumented_behavior_test_source=test.instrumented_behavior_test_source,
                instrumented_perf_test_source=test.instrumented_perf_test_source,
                behavior_file_path=test.behavior_file_path,
                perf_file_path=test.perf_file_path,
            )
            modified_tests.append(modified_test)
        except Exception as e:
            logger.debug(f"Failed to add runtime comments to test: {e}")
            modified_tests.append(test)

    return GeneratedTestsList(generated_tests=modified_tests)
