from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Union

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

    def __init__(self, source_code: str) -> None:
        self.source_lines = source_code.splitlines()
        self.results: list[int] = []  # map actual line number to line number in ast

    def _is_codeflash_output_target(self, target: Union[ast.expr, list]) -> bool:  # type: ignore[type-arg]
        """Check if the assignment target is the variable 'codeflash_output'."""
        if isinstance(target, ast.Name):
            return target.id == "codeflash_output"
        if isinstance(target, (ast.Tuple, ast.List)):
            # Handle tuple/list unpacking: a, codeflash_output, b = values
            return any(self._is_codeflash_output_target(elt) for elt in target.elts)
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            # Not a simple variable assignment
            return False
        return False

    def _record_assignment(self, node: ast.AST) -> None:
        """Record an assignment to codeflash_output."""
        relative_line = node.lineno - 1  # type: ignore[attr-defined]
        self.results.append(relative_line)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment statements: codeflash_output = value."""
        for target in node.targets:
            if self._is_codeflash_output_target(target):
                self._record_assignment(node)
                break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments: codeflash_output: int = value."""
        if self._is_codeflash_output_target(node.target):
            self._record_assignment(node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Visit augmented assignments: codeflash_output += value."""
        if self._is_codeflash_output_target(node.target):
            self._record_assignment(node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Visit walrus operator: (codeflash_output := value)."""
        if isinstance(node.target, ast.Name) and node.target.id == "codeflash_output":
            self._record_assignment(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loops: for codeflash_output in iterable."""
        if self._is_codeflash_output_target(node.target):
            self._record_assignment(node)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """Visit comprehensions: [x for codeflash_output in iterable]."""
        if self._is_codeflash_output_target(node.target):
            # Comprehensions don't have line numbers, so we skip recording
            pass
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Visit with statements: with expr as codeflash_output."""
        for item in node.items:
            if item.optional_vars and self._is_codeflash_output_target(item.optional_vars):
                self._record_assignment(node)
                break
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Visit except handlers: except Exception as codeflash_output."""
        if node.name == "codeflash_output":
            self._record_assignment(node)
        self.generic_visit(node)


def find_codeflash_output_assignments(source_code: str) -> list[int]:
    tree = ast.parse(source_code)
    visitor = CfoVisitor(source_code)
    visitor.visit(tree)
    return visitor.results


def add_runtime_comments_to_generated_tests(
    test_cfg: TestConfig,
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    tests_root = test_cfg.tests_root
    module_root = test_cfg.project_root_path
    rel_tests_root = tests_root.relative_to(module_root)

    # TODO: reduce for loops to one
    class RuntimeCommentTransformer(cst.CSTTransformer):
        def __init__(self, module: cst.Module, test: GeneratedTests, tests_root: Path, rel_tests_root: Path) -> None:
            super().__init__()
            self.test = test
            self.context_stack: list[str] = []
            self.tests_root = tests_root
            self.rel_tests_root = rel_tests_root
            self.module = module
            self.cfo_locs: list[int] = []
            self.cfo_idx_loc_to_look_at: int = -1

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
                find_codeflash_output_assignments(normalized_body_code)
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
            # Look for assignment statements that assign to codeflash_output
            # Handle both single statements and multiple statements on one line
            codeflash_assignment_found = False
            for stmt in updated_node.body:
                if isinstance(stmt, cst.Assign) and (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0].target, cst.Name)
                    and stmt.targets[0].target.value == "codeflash_output"
                ):
                    codeflash_assignment_found = True
                    break

            if codeflash_assignment_found:
                # Find matching test cases by looking for this test function name in the test results
                self.cfo_idx_loc_to_look_at += 1
                matching_original_times = []
                matching_optimized_times = []
                # TODO : will not work if there are multiple test cases with the same name, match filename + test class + test function name + invocationid
                for invocation_id, runtimes in original_runtimes.items():
                    # get position here and match in if condition
                    qualified_name = (
                        invocation_id.test_class_name + "." + invocation_id.test_function_name  # type: ignore[operator]
                        if invocation_id.test_class_name
                        else invocation_id.test_function_name
                    )
                    rel_path = (
                        Path(invocation_id.test_module_path.replace(".", os.sep))
                        .with_suffix(".py")
                        .relative_to(self.rel_tests_root)
                    )
                    if (
                        qualified_name == ".".join(self.context_stack)
                        and rel_path
                        in [
                            self.test.behavior_file_path.relative_to(self.tests_root),
                            self.test.perf_file_path.relative_to(self.tests_root),
                        ]
                        and int(invocation_id.iteration_id.split("_")[0]) == self.cfo_locs[self.cfo_idx_loc_to_look_at]  # type:ignore[union-attr]
                    ):
                        matching_original_times.extend(runtimes)

                for invocation_id, runtimes in optimized_runtimes.items():
                    # get position here and match in if condition
                    qualified_name = (
                        invocation_id.test_class_name + "." + invocation_id.test_function_name  # type: ignore[operator]
                        if invocation_id.test_class_name
                        else invocation_id.test_function_name
                    )
                    rel_path = (
                        Path(invocation_id.test_module_path.replace(".", os.sep))
                        .with_suffix(".py")
                        .relative_to(self.rel_tests_root)
                    )
                    if (
                        qualified_name == ".".join(self.context_stack)
                        and rel_path
                        in [
                            self.test.behavior_file_path.relative_to(self.tests_root),
                            self.test.perf_file_path.relative_to(self.tests_root),
                        ]
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

                        # Add comment to the trailing whitespace
                        new_trailing_whitespace = cst.TrailingWhitespace(
                            whitespace=cst.SimpleWhitespace(" "),
                            comment=cst.Comment(comment_text),
                            newline=updated_node.trailing_whitespace.newline,
                        )

                        return updated_node.with_changes(trailing_whitespace=new_trailing_whitespace)

            return updated_node

    # Process each generated test
    modified_tests = []
    for test in generated_tests.generated_tests:
        try:
            # Parse the test source code
            tree = cst.parse_module(test.generated_original_test_source)
            # Transform the tree to add runtime comments
            transformer = RuntimeCommentTransformer(tree, test, tests_root, rel_tests_root)
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
