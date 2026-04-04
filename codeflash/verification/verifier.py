from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import module_name_from_file_path
from codeflash.languages import current_language_support
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
    is_numerical_code: bool | None = None,
    rerun_trace_id: str | None = None,
) -> tuple[str, str, str, str | None, Path, Path] | None:
    # TODO: Sometimes this recreates the original Class definition. This overrides and messes up the original
    #  class import. Remove the recreation of the class definition
    start_time = time.perf_counter()
    # Use traverse_up=True to handle co-located __tests__ directories that may be outside
    # the configured tests_root (e.g., src/gateway/__tests__/ when tests_root is test/)
    test_module_path = Path(module_name_from_file_path(test_path, test_cfg.tests_project_rootdir, traverse_up=True))

    # Detect module system via language support (non-None for JS/TS, None for Python)
    lang_support = current_language_support()
    source_file = Path(function_to_optimize.file_path)
    project_module_system = lang_support.detect_module_system(test_cfg.tests_project_rootdir, source_file)

    if project_module_system is not None:
        # For JavaScript/TypeScript, calculate the correct import path from the actual test location
        # (test_path) to the source file, not from tests_root
        import os

        source_file_abs = source_file.resolve().with_suffix("")
        test_dir_abs = test_path.resolve().parent
        # Compute relative path from test directory to source file
        rel_import_path = os.path.relpath(str(source_file_abs), str(test_dir_abs))
        # Ensure path starts with ./ or ../ for JavaScript/TypeScript imports
        if not rel_import_path.startswith("../"):
            rel_import_path = f"./{rel_import_path}"
        # ESM requires explicit file extensions in import specifiers.
        # TypeScript ESM also uses .js extensions (TS resolves .js → .ts).
        if project_module_system == "esm":
            rel_import_path += ".js"
        # Keep as string since Path() normalizes away the ./ prefix
        module_path = rel_import_path

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
        language=function_to_optimize.language,
        language_version=current_language_support().language_version,
        module_system=project_module_system,
        is_numerical_code=is_numerical_code,
        rerun_trace_id=rerun_trace_id,
    )
    if response and isinstance(response, tuple) and len(response) == 4:
        generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source, raw_generated_tests = (
            response
        )

        generated_test_source, instrumented_behavior_test_source, instrumented_perf_test_source = (
            lang_support.process_generated_test_strings(
                generated_test_source=generated_test_source,
                instrumented_behavior_test_source=instrumented_behavior_test_source,
                instrumented_perf_test_source=instrumented_perf_test_source,
                function_to_optimize=function_to_optimize,
                test_path=test_path,
                test_cfg=test_cfg,
                project_module_system=project_module_system,
            )
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
        raw_generated_tests,
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
