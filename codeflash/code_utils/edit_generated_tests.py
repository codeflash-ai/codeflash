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
from codeflash.models.models import GeneratedTests, GeneratedTestsList
from codeflash.result.critic import performance_gain

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

    def __init__(self, function_name: str, source_code: str) -> None:
        self.source_lines = source_code.splitlines()
        self.name = function_name
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


def find_codeflash_output_assignments(function_name: str, source_code: str) -> list[int]:
    tree = ast.parse(source_code)
    visitor = CfoVisitor(function_name, source_code)
    visitor.visit(tree)
    return visitor.results


class Finder(cst.CSTVisitor):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.found = False
        self.name = name

    def visit_Call(self, call_node) -> None:  # type: ignore[no-untyped-def] # noqa : ANN001
        func_expr = call_node.func
        if isinstance(func_expr, cst.Name):
            if func_expr.value == self.name:
                self.found = True
        elif isinstance(func_expr, cst.Attribute):  # noqa : SIM102
            if func_expr.attr.value == self.name:
                self.found = True


# TODO: reduce for loops to one
class RuntimeCommentTransformer(cst.CSTTransformer):
    def __init__(
        self,
        qualified_name: str,
        module: cst.Module,
        test: GeneratedTests,
        tests_root: Path,
        original_runtimes: dict[InvocationId, list[int]],
        optimized_runtimes: dict[InvocationId, list[int]],
    ) -> None:
        super().__init__()
        self.test = test
        self.context_stack: list[str] = []
        self.tests_root = tests_root
        self.module = module
        self.cfo_locs: list[int] = []
        self.cfo_idx_loc_to_look_at: int = -1
        self.name = qualified_name.split(".")[-1]
        self.original_runtimes = original_runtimes
        self.optimized_runtimes = optimized_runtimes

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Track when we enter a class
        self.context_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:  # noqa: ARG002
        # Pop the context when we leave a class
        self.context_stack.pop()
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # convert function body to ast normalized string and find occurrences of codeflash_output
        body_code = dedent(self.module.code_for_node(node.body))
        normalized_body_code = ast.unparse(ast.parse(body_code))
        self.cfo_locs = sorted(
            find_codeflash_output_assignments(self.name, normalized_body_code)
        )  # sorted in order we will encounter them
        self.cfo_idx_loc_to_look_at = -1
        self.context_stack.append(node.name.value)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:  # noqa: ARG002
        # Pop the context when we leave a function
        self.context_stack.pop()
        return updated_node

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,  # noqa: ARG002
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine:
        # Check if this statement line contains a call to self.name
        if self._contains_myfunc_call(updated_node):  # type: ignore[no-untyped-call]
            # Find matching test cases by looking for this test function name in the test results
            self.cfo_idx_loc_to_look_at += 1
            matching_original_times = []
            matching_optimized_times = []
            # TODO : will not work if there are multiple test cases with the same name, match filename + test class + test function name + invocationid
            for invocation_id, runtimes in self.original_runtimes.items():
                # get position here and match in if condition
                qualified_name = (
                    invocation_id.test_class_name + "." + invocation_id.test_function_name  # type: ignore[operator]
                    if invocation_id.test_class_name
                    else invocation_id.test_function_name
                )
                abs_path = Path(invocation_id.test_module_path.replace(".", os.sep)).with_suffix(".py").resolve()
                if (
                    qualified_name == ".".join(self.context_stack)
                    and abs_path in [self.test.behavior_file_path, self.test.perf_file_path]
                    and int(invocation_id.iteration_id.split("_")[0]) == self.cfo_locs[self.cfo_idx_loc_to_look_at]  # type:ignore[union-attr]
                ):
                    matching_original_times.extend(runtimes)

            for invocation_id, runtimes in self.optimized_runtimes.items():
                # get position here and match in if condition
                qualified_name = (
                    invocation_id.test_class_name + "." + invocation_id.test_function_name  # type: ignore[operator]
                    if invocation_id.test_class_name
                    else invocation_id.test_function_name
                )
                abs_path = Path(invocation_id.test_module_path.replace(".", os.sep)).with_suffix(".py").resolve()
                if (
                    qualified_name == ".".join(self.context_stack)
                    and abs_path in [self.test.behavior_file_path, self.test.perf_file_path]
                    and int(invocation_id.iteration_id.split("_")[0]) == self.cfo_locs[self.cfo_idx_loc_to_look_at]  # type:ignore[union-attr]
                ):
                    matching_optimized_times.extend(runtimes)

            if matching_original_times and matching_optimized_times:
                original_time = min(matching_original_times)
                optimized_time = min(matching_optimized_times)
                if original_time != 0 and optimized_time != 0:
                    perf_gain = format_perf(
                        abs(
                            performance_gain(original_runtime_ns=original_time, optimized_runtime_ns=optimized_time)
                            * 100
                        )
                    )
                    status = "slower" if optimized_time > original_time else "faster"
                    # Create the runtime comment
                    comment_text = (
                        f"# {format_time(original_time)} -> {format_time(optimized_time)} ({perf_gain}% {status})"
                    )
                    return updated_node.with_changes(
                        trailing_whitespace=cst.TrailingWhitespace(
                            whitespace=cst.SimpleWhitespace(" "),
                            comment=cst.Comment(comment_text),
                            newline=updated_node.trailing_whitespace.newline,
                        )
                    )
        return updated_node

    def _contains_myfunc_call(self, node):  # type: ignore[no-untyped-def] # noqa : ANN202, ANN001
        """Recursively search for any Call node in the statement whose function is named self.name (including obj.myfunc)."""
        finder = Finder(self.name)
        node.visit(finder)
        return finder.found


def add_runtime_comments_to_generated_tests(
    qualified_name: str,
    test_cfg: TestConfig,
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    tests_root = test_cfg.tests_root

    # Process each generated test
    modified_tests = []
    for test in generated_tests.generated_tests:
        try:
            # Parse the test source code
            tree = cst.parse_module(test.generated_original_test_source)
            # Transform the tree to add runtime comments
            # qualified_name: str, module: cst.Module, test: GeneratedTests, tests_root: Path
            transformer = RuntimeCommentTransformer(
                qualified_name, tree, test, tests_root, original_runtimes, optimized_runtimes
            )
            modified_tree = tree.visit(transformer)

            # Convert back to source code
            modified_source = modified_tree.code

            # Create a new GeneratedTests object with the modified source
            modified_test = GeneratedTests(
                generated_original_test_source=modified_source,
                instrumented_behavior_test_source=test.instrumented_behavior_test_source,
                instrumented_perf_test_source=test.instrumented_perf_test_source,
                behavior_file_path=test.behavior_file_path,
                perf_file_path=test.perf_file_path,
            )
            modified_tests.append(modified_test)
        except Exception as e:
            # If parsing fails, keep the original test
            logger.debug(f"Failed to add runtime comments to test: {e}")
            modified_tests.append(test)

    return GeneratedTestsList(generated_tests=modified_tests)
