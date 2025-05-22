from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import get_run_tmp_file, module_name_from_file_path
from codeflash.verification.verification_utils import ModifyInspiredTests, delete_multiple_if_name_main

if TYPE_CHECKING:
    from codeflash.api.aiservice import AiServiceClient
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.verification.verification_utils import TestConfig


def generate_tests(
    aiservice_client: AiServiceClient,
    source_code_being_tested: str,
    function_to_optimize: FunctionToOptimize,
    helper_function_names: list[str],
    module_path: Path,
    test_cfg: TestConfig,
    test_timeout: int,
    function_trace_id: str,
    test_index: int,
    test_path: Path,
    test_perf_path: Path,
) -> tuple[str, str, Path] | None:
    # TODO: Sometimes this recreates the original Class definition. This overrides and messes up the original
    #  class import. Remove the recreation of the class definition
    start_time = time.perf_counter()
    test_module_path = Path(module_name_from_file_path(test_path, test_cfg.tests_project_rootdir))
    response = aiservice_client.generate_regression_tests(
        source_code_being_tested=source_code_being_tested,
        function_to_optimize=function_to_optimize,
        helper_function_names=helper_function_names,
        module_path=module_path,
        test_module_path=test_module_path,
        test_framework=test_cfg.test_framework,
        test_timeout=test_timeout,
        trace_id=function_trace_id,
        test_index=test_index,
    )
    if response and isinstance(response, tuple) and len(response) == 3:
        generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source = response
        temp_run_dir = get_run_tmp_file(Path()).as_posix()

        instrumented_behavior_test_source = instrumented_behavior_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
        instrumented_perf_test_source = instrumented_perf_test_source.replace(
            "{codeflash_run_tmp_dir_client_side}", temp_run_dir
        )
    else:
        logger.warning(f"Failed to generate and instrument tests for {function_to_optimize.function_name}")
        return None
    end_time = time.perf_counter()
    logger.debug(f"Generated tests in {end_time - start_time:.2f} seconds")
    return (
        generated_test_source,
        instrumented_behavior_test_source,
        instrumented_perf_test_source,
        test_path,
        test_perf_path,
    )


def merge_unit_tests(unit_test_source: str, inspired_unit_tests: str, test_framework: str) -> str:
    try:
        inspired_unit_tests_ast = ast.parse(inspired_unit_tests)
        unit_test_source_ast = ast.parse(unit_test_source)
    except SyntaxError as e:
        logger.exception(f"Syntax error in code: {e}")
        return unit_test_source
    import_list: list[ast.stmt] = []
    modified_ast = ModifyInspiredTests(import_list, test_framework).visit(inspired_unit_tests_ast)
    if test_framework == "pytest":
        # Because we only want to modify the top level test functions
        for node in ast.iter_child_nodes(modified_ast):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                node.name = node.name + "__inspired"
    unit_test_source_ast.body.extend(modified_ast.body)
    unit_test_source_ast.body = import_list + unit_test_source_ast.body
    if test_framework == "unittest":
        unit_test_source_ast = delete_multiple_if_name_main(unit_test_source_ast)
    return ast.unparse(unit_test_source_ast)
