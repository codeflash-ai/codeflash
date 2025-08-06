from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst
from libcst import MetadataWrapper
from libcst.metadata import PositionProvider

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.time_utils import format_perf, format_time
from codeflash.models.models import GeneratedTests, GeneratedTestsList
from codeflash.result.critic import performance_gain

if TYPE_CHECKING:
    from codeflash.models.models import InvocationId


class CommentMapper(ast.NodeVisitor):
    def __init__(
        self, test: GeneratedTests, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
    ) -> None:
        self.results: dict[int, str] = {}
        self.test: GeneratedTests = test
        self.original_runtimes = original_runtimes
        self.optimized_runtimes = optimized_runtimes
        self.abs_path = test.behavior_file_path.with_suffix("")
        self.context_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.context_stack.append(node.name)
        for inner_node in ast.walk(node):
            if isinstance(inner_node, ast.FunctionDef):
                self.visit_FunctionDef(inner_node)
        self.context_stack.pop()
        return node

    def get_comment(self, match_key: str) -> str:
        # calculate speedup and output comment
        original_time = self.original_runtimes[match_key]
        optimized_time = self.optimized_runtimes[match_key]
        perf_gain = format_perf(
            abs(performance_gain(original_runtime_ns=original_time, optimized_runtime_ns=optimized_time) * 100)
        )
        status = "slower" if optimized_time > original_time else "faster"
        # Create the runtime comment
        return f"# {format_time(original_time)} -> {format_time(optimized_time)} ({perf_gain}% {status})"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.context_stack.append(node.name)
        i = len(node.body) - 1
        test_qualified_name = ".".join(self.context_stack)
        key = test_qualified_name + "#" + str(self.abs_path)
        while i >= 0:
            line_node = node.body[i]
            if isinstance(line_node, (ast.With, ast.For, ast.While, ast.If)):
                j = len(line_node.body) - 1
                while j >= 0:
                    compound_line_node: ast.stmt = line_node.body[j]
                    internal_node: ast.AST
                    for internal_node in ast.walk(compound_line_node):
                        if isinstance(internal_node, (ast.stmt, ast.Assign)):
                            inv_id = str(i) + "_" + str(j)
                            match_key = key + "#" + inv_id
                            if match_key in self.original_runtimes and match_key in self.optimized_runtimes:
                                self.results[internal_node.lineno] = self.get_comment(match_key)
                    j -= 1
            else:
                inv_id = str(i)
                match_key = key + "#" + inv_id
                if match_key in self.original_runtimes and match_key in self.optimized_runtimes:
                    self.results[line_node.lineno] = self.get_comment(match_key)
            i -= 1
        self.context_stack.pop()
        return node


def get_fn_call_linenos(
    test: GeneratedTests, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]
) -> dict[int, str]:
    line_comment_ast_mapper = CommentMapper(test, original_runtimes, optimized_runtimes)
    source_code = test.generated_original_test_source
    tree = ast.parse(source_code)
    line_comment_ast_mapper.visit(tree)
    return line_comment_ast_mapper.results


class CommentAdder(cst.CSTTransformer):
    """Transformer that adds comments to specified lines."""

    # Declare metadata dependencies
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, line_to_comments: dict[int, str]) -> None:
        """Initialize the transformer with target line numbers.

        Args:
            line_to_comments: Mapping of line numbers (1-indexed) to comments

        """
        self.line_to_comments = line_to_comments
        super().__init__()

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        """Add comment to simple statement lines."""
        pos = self.get_metadata(PositionProvider, original_node)

        if pos and pos.start.line in self.line_to_comments:
            # Create a comment with trailing whitespace
            comment = cst.TrailingWhitespace(
                whitespace=cst.SimpleWhitespace(" "), comment=cst.Comment(self.line_to_comments[pos.start.line])
            )

            # Update the trailing whitespace of the line itself
            return updated_node.with_changes(trailing_whitespace=comment)

        return updated_node

    def leave_SimpleStatementSuite(
        self, original_node: cst.SimpleStatementSuite, updated_node: cst.SimpleStatementSuite
    ) -> cst.SimpleStatementSuite:
        """Add comment to simple statement suites (e.g., after if/for/while)."""
        pos = self.get_metadata(PositionProvider, original_node)

        if pos and pos.start.line in self.line_to_comments:
            # Create a comment with trailing whitespace
            comment = cst.TrailingWhitespace(
                whitespace=cst.SimpleWhitespace("  "), comment=cst.Comment(self.line_to_comments[pos.start.line])
            )

            # Update the trailing whitespace of the suite
            return updated_node.with_changes(trailing_whitespace=comment)

        return updated_node


def unique_inv_id(inv_id_runtimes: dict[InvocationId, list[int]]) -> dict[str, int]:
    unique_inv_ids: dict[str, int] = {}
    for inv_id, runtimes in inv_id_runtimes.items():
        test_qualified_name = (
            inv_id.test_class_name + "." + inv_id.test_function_name  # type: ignore[operator]
            if inv_id.test_class_name
            else inv_id.test_function_name
        )
        abs_path = str(Path(inv_id.test_module_path.replace(".", os.sep)).with_suffix(".py").resolve().with_suffix(""))
        if "__unit_test_" not in abs_path:
            continue
        key = test_qualified_name + "#" + abs_path  # type: ignore[operator]
        parts = inv_id.iteration_id.split("_").__len__()  # type: ignore[union-attr]
        cur_invid = inv_id.iteration_id.split("_")[0] if parts < 3 else "_".join(inv_id.iteration_id.split("_")[:-1])  # type: ignore[union-attr]
        match_key = key + "#" + cur_invid
        if match_key not in unique_inv_ids:
            unique_inv_ids[match_key] = 0
        unique_inv_ids[match_key] += min(runtimes)
    return unique_inv_ids


def add_runtime_comments_to_generated_tests(
    generated_tests: GeneratedTestsList,
    original_runtimes: dict[InvocationId, list[int]],
    optimized_runtimes: dict[InvocationId, list[int]],
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    original_runtimes_dict = unique_inv_id(original_runtimes)
    optimized_runtimes_dict = unique_inv_id(optimized_runtimes)
    # Process each generated test
    modified_tests = []
    for test in generated_tests.generated_tests:
        try:
            tree = cst.parse_module(test.generated_original_test_source)
            wrapper = MetadataWrapper(tree)
            line_to_comments = get_fn_call_linenos(test, original_runtimes_dict, optimized_runtimes_dict)
            comment_adder = CommentAdder(line_to_comments)
            modified_tree = wrapper.visit(comment_adder)
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
            # If parsing fails, keep the original test
            logger.debug(f"Failed to add runtime comments to test: {e}")
            modified_tests.append(test)

    return GeneratedTestsList(generated_tests=modified_tests)


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
