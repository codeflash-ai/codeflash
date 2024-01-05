import ast
import logging

from codeflash.api.aiservice import generate_regression_tests
from codeflash.code_utils.ast_unparser import ast_unparse
from codeflash.code_utils.code_utils import get_run_tmp_file
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.instrumentation.instrument_new_tests import InjectPerfAndLogging
from codeflash.instrumentation.instrument_new_tests import inject_logging_code
from codeflash.optimization.function_context import Source
from codeflash.verification.verification_utils import ModifyInspiredTests
from codeflash.verification.verification_utils import delete_multiple_if_name_main


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


def instrument_test_source(
    test_source: str,
    function: FunctionToOptimize,
    function_dependencies: list[Source],
    module_path: str,
    test_module_path: str,
    test_framework: str,
    test_timeout: int,
) -> str:
    module_node = ast.parse(test_source)
    auxiliary_function_names = [definition.full_name for definition in function_dependencies]
    new_module_node = InjectPerfAndLogging(
        function,
        auxiliary_function_names=auxiliary_function_names,
        test_module_path=test_module_path,
        test_framework=test_framework,
        test_timeout=test_timeout,
    ).visit(module_node)
    new_imports = [
        ast.Import(names=[ast.alias(name="time")]),
        ast.Import(names=[ast.alias(name="gc")]),
        # This adds the function to the namespace
        ast.ImportFrom(
            module=module_path,
            names=[ast.alias(name=function.top_level_parent_name)],
            col_offset=0,
            lineno=1,
            level=0,
        ),
    ]
    if test_framework == "unittest":
        new_imports += [ast.Import(names=[ast.alias(name="timeout_decorator")])]
    new_module_node.body = new_imports + new_module_node.body
    new_tests = ast_unparse(new_module_node)
    modified_new_tests = inject_logging_code(new_tests, tmp_dir=get_run_tmp_file(""))
    return modified_new_tests


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
