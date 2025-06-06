import re

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.time_utils import format_time
from codeflash.models.models import GeneratedTests, GeneratedTestsList, TestResults


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


def add_runtime_comments_to_generated_tests(
    generated_tests: GeneratedTestsList, original_test_results: TestResults, optimized_test_results: TestResults
) -> GeneratedTestsList:
    """Add runtime performance comments to function calls in generated tests."""
    # Create dictionaries for fast lookup of runtime data
    original_runtime_by_test = original_test_results.usable_runtime_data_by_test_case()
    optimized_runtime_by_test = optimized_test_results.usable_runtime_data_by_test_case()

    class RuntimeCommentTransformer(cst.CSTTransformer):
        def __init__(self) -> None:
            self.in_test_function = False
            self.current_test_name: str | None = None

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if node.name.value.startswith("test_"):
                self.in_test_function = True
                self.current_test_name = node.name.value
            else:
                self.in_test_function = False
                self.current_test_name = None

        def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
            if original_node.name.value.startswith("test_"):
                self.in_test_function = False
                self.current_test_name = None
            return updated_node

        def leave_SimpleStatementLine(
            self,
            original_node: cst.SimpleStatementLine,  # noqa: ARG002
            updated_node: cst.SimpleStatementLine,
        ) -> cst.SimpleStatementLine:
            if not self.in_test_function or not self.current_test_name:
                return updated_node

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
                matching_original_times = []
                matching_optimized_times = []

                for invocation_id, runtimes in original_runtime_by_test.items():
                    if invocation_id.test_function_name == self.current_test_name:
                        matching_original_times.extend(runtimes)

                for invocation_id, runtimes in optimized_runtime_by_test.items():
                    if invocation_id.test_function_name == self.current_test_name:
                        matching_optimized_times.extend(runtimes)

                if matching_original_times and matching_optimized_times:
                    original_time = min(matching_original_times)
                    optimized_time = min(matching_optimized_times)

                    # Create the runtime comment
                    comment_text = f"# {format_time(original_time)} -> {format_time(optimized_time)}"

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
            transformer = RuntimeCommentTransformer()
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
