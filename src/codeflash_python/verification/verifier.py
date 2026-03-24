from __future__ import annotations

import ast
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_python.code_utils.code_utils import module_name_from_file_path
from codeflash_python.verification.test_runner import process_generated_test_strings
from codeflash_python.verification.verification_utils import ModifyInspiredTests, delete_multiple_if_name_main

if TYPE_CHECKING:
    from codeflash_core.models import FunctionToOptimize
    from codeflash_python.api.aiservice import AiServiceClient


logger = logging.getLogger("codeflash_python")


def generate_tests(
    aiservice_client: AiServiceClient,
    source_code_being_tested: str,
    function_to_optimize: FunctionToOptimize,
    helper_function_names: list[str],
    module_path: Path,
    test_cfg_project_root: Path,
    test_timeout: int,
    function_trace_id: str,
    test_index: int,
    test_path: Path,
    test_perf_path: Path,
    is_numerical_code: bool | None = None,
) -> tuple[str, str, str, str | None, Path, Path] | None:
    """Generate regression tests for a single function.

    Wraps AiServiceClient.generate_regression_tests() and processes
    the returned test strings.
    """
    start_time = time.perf_counter()
    test_module_path = Path(module_name_from_file_path(test_path, test_cfg_project_root))

    response = aiservice_client.generate_regression_tests(
        source_code_being_tested=source_code_being_tested,
        function_to_optimize=function_to_optimize,
        helper_function_names=helper_function_names,
        module_path=module_path,
        test_module_path=test_module_path,
        test_framework="pytest",
        test_timeout=test_timeout,
        trace_id=function_trace_id,
        test_index=test_index,
        is_numerical_code=is_numerical_code,
    )

    if response and isinstance(response, tuple) and len(response) == 4:
        generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source, raw_generated_tests = (
            response
        )

        generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source = (
            process_generated_test_strings(
                generated_test_source=generated_test_source,
                instrumented_behavior_test_source=instrumented_behavior_test_source,
                instrumented_perf_test_source=instrumented_perf_test_source,
                function_to_optimize=function_to_optimize,
                test_path=test_path,
                test_cfg=None,
                project_module_system=None,
            )
        )
    else:
        logger.warning("Failed to generate tests for %s", function_to_optimize.function_name)
        return None

    end_time = time.perf_counter()
    logger.debug("Generated tests in %.2f seconds", end_time - start_time)
    return (
        generated_test_source,
        instrumented_behavior_test_source,
        instrumented_perf_test_source,
        raw_generated_tests,
        test_path,
        test_perf_path,
    )


def merge_unit_tests(unit_test_source: str, inspired_unit_tests: str, test_framework: str) -> str:
    try:
        inspired_unit_tests_ast = ast.parse(inspired_unit_tests)
        unit_test_source_ast = ast.parse(unit_test_source)
    except SyntaxError as e:
        logger.exception("Syntax error in code: %s", e)
        return unit_test_source
    import_list: list[ast.stmt] = []
    modified_ast = ModifyInspiredTests(import_list, test_framework).visit(inspired_unit_tests_ast)
    if test_framework == "pytest":
        for node in ast.iter_child_nodes(modified_ast):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                node.name = node.name + "__inspired"
    unit_test_source_ast.body.extend(modified_ast.body)
    unit_test_source_ast.body = import_list + unit_test_source_ast.body
    if test_framework == "unittest":
        unit_test_source_ast = delete_multiple_if_name_main(unit_test_source_ast)
    return ast.unparse(unit_test_source_ast)
