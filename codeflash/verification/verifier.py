import ast
import logging

from codeflash.api.aiservice import generate_regression_tests
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.verification.verification_utils import ModifyInspiredTests
from codeflash.verification.verification_utils import delete_multiple_if_name_main
from injectperf.ast_unparser import ast_unparse


def generate_tests(
    source_code_being_tested: str,
    function: FunctionToOptimize,
    module_path: str,
    test_framework: str,
    use_cached_tests: bool,
) -> str | None:
    # TODO: Sometimes this recreates the original Class definition. This overrides and messes up the original
    #  class import. Remove the recreation of the class definition
    logging.info(f"Generating new tests for function {function.function_name} ...")
    if use_cached_tests:
        import importlib

        module = importlib.import_module(module_path)
        generated_test_source = module.CACHED_TESTS
        logging.info(f"Using cached tests from {module_path}.CACHED_TESTS")
    else:
        generated_test_source = generate_regression_tests(
            source_code_being_tested=source_code_being_tested,
            function_name=function.function_name,
            test_framework=test_framework,
        )
        if generated_test_source is None:
            logging.error("Test generation failed. Skipping test generation.")
            return None  # TODO: if we're generating inspired unit tests, don't return here yet

    inspired_unit_tests = ""

    unified_test_source = merge_unit_tests(
        generated_test_source, inspired_unit_tests, test_framework
    )

    return unified_test_source


def merge_unit_tests(unit_test_source: str, inspired_unit_tests: str, test_framework: str) -> str:
    inspired_unit_tests_ast = ast.parse(inspired_unit_tests)
    unit_test_source_ast = ast.parse(unit_test_source)
    import_list: list[ast.stmt] = list()
    modified_ast = ModifyInspiredTests(import_list, test_framework).visit(inspired_unit_tests_ast)
    if test_framework == "pytest":
        # Because we only want to modify the top level test functions
        for node in ast.iter_child_nodes(modified_ast):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    node.name = node.name + "__inspired"
    unit_test_source_ast.body.extend(modified_ast.body)
    unit_test_source_ast.body = import_list + unit_test_source_ast.body
    if test_framework == "unittest":
        unit_test_source_ast = delete_multiple_if_name_main(unit_test_source_ast)
    return ast_unparse(unit_test_source_ast)
