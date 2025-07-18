from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pygls import uris

from codeflash.either import is_successful
from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol

if TYPE_CHECKING:
    from lsprotocol import types

    from codeflash.models.models import GeneratedTestsList, OptimizationSet


@dataclass
class OptimizableFunctionsParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815


@dataclass
class FunctionOptimizationParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815
    functionName: str  # noqa: N815


server = CodeflashLanguageServer("codeflash-language-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol)


@server.feature("getOptimizableFunctions")
def get_optimizable_functions(
    server: CodeflashLanguageServer, params: OptimizableFunctionsParams
) -> dict[str, list[str]]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.show_message_log(f"Getting optimizable functions for: {file_path}", "Info")

    # Save original args to restore later
    original_file = getattr(server.optimizer.args, "file", None)
    original_function = getattr(server.optimizer.args, "function", None)
    original_checkpoint = getattr(server.optimizer.args, "previous_checkpoint_functions", None)

    server.show_message_log(f"Original args - file: {original_file}, function: {original_function}", "Info")

    try:
        # Set temporary args for this request only
        server.optimizer.args.file = file_path
        server.optimizer.args.function = None  # Always get ALL functions, not just one
        server.optimizer.args.previous_checkpoint_functions = False

        server.show_message_log("Calling get_optimizable_functions...", "Info")
        optimizable_funcs, _, _ = server.optimizer.get_optimizable_functions()

        path_to_qualified_names = {}
        for path, functions in optimizable_funcs.items():
            path_to_qualified_names[path.as_posix()] = [func.qualified_name for func in functions]

        server.show_message_log(
            f"Found {len(path_to_qualified_names)} files with functions: {path_to_qualified_names}", "Info"
        )
        return path_to_qualified_names
    finally:
        # Restore original args to prevent state corruption
        if original_file is not None:
            server.optimizer.args.file = original_file
        if original_function is not None:
            server.optimizer.args.function = original_function
        else:
            server.optimizer.args.function = None
        if original_checkpoint is not None:
            server.optimizer.args.previous_checkpoint_functions = original_checkpoint

        server.show_message_log(
            f"Restored args - file: {server.optimizer.args.file}, function: {server.optimizer.args.function}", "Info"
        )


@server.feature("initializeFunctionOptimization")
def initialize_function_optimization(
    server: CodeflashLanguageServer, params: FunctionOptimizationParams
) -> dict[str, str]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.show_message_log(f"Initializing optimization for function: {params.functionName} in {file_path}", "Info")

    # IMPORTANT: Store the specific function for optimization, but don't corrupt global state
    server.optimizer.args.function = params.functionName
    server.optimizer.args.file = file_path

    server.show_message_log(
        f"Args set - function: {server.optimizer.args.function}, file: {server.optimizer.args.file}", "Info"
    )

    optimizable_funcs, _, _ = server.optimizer.get_optimizable_functions()
    if not optimizable_funcs:
        server.show_message_log(f"No optimizable functions found for {params.functionName}", "Warning")
        return {"functionName": params.functionName, "status": "not found", "args": None}

    fto = optimizable_funcs.popitem()[1][0]
    server.optimizer.current_function_being_optimized = fto
    server.show_message_log(f"Successfully initialized optimization for {params.functionName}", "Info")
    return {"functionName": params.functionName, "status": "success"}


@server.feature("discoverFunctionTests")
def discover_function_tests(server: CodeflashLanguageServer, params: FunctionOptimizationParams) -> dict[str, str]:
    fto = server.optimizer.current_function_being_optimized
    optimizable_funcs = {fto.file_path: [fto]}

    devnull_writer = open(os.devnull, "w")  # noqa
    with contextlib.redirect_stdout(devnull_writer):
        function_to_tests, num_discovered_tests = server.optimizer.discover_tests(optimizable_funcs)

    server.optimizer.discovered_tests = function_to_tests

    return {"functionName": params.functionName, "status": "success", "discovered_tests": num_discovered_tests}


@server.feature("prepareOptimization")
def prepare_optimization(server: CodeflashLanguageServer, params: FunctionOptimizationParams) -> dict[str, str]:
    current_function = server.optimizer.current_function_being_optimized

    module_prep_result = server.optimizer.prepare_module_for_optimization(current_function.file_path)
    validated_original_code, original_module_ast = module_prep_result

    function_optimizer = server.optimizer.create_function_optimizer(
        current_function,
        function_to_optimize_source_code=validated_original_code[current_function.file_path].source_code,
        original_module_ast=original_module_ast,
        original_module_path=current_function.file_path,
    )

    server.optimizer.current_function_optimizer = function_optimizer
    if not function_optimizer:
        return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

    initialization_result = function_optimizer.can_be_optimized()
    if not is_successful(initialization_result):
        return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

    return {"functionName": params.functionName, "status": "success", "message": "Optimization preparation completed"}


@server.feature("generateTests")
def generate_tests(server: CodeflashLanguageServer, params: FunctionOptimizationParams) -> dict[str, str]:
    function_optimizer = server.optimizer.current_function_optimizer
    if not function_optimizer:
        return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

    initialization_result = function_optimizer.can_be_optimized()
    if not is_successful(initialization_result):
        return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

    should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()

    test_setup_result = function_optimizer.generate_and_instrument_tests(
        code_context, should_run_experiment=should_run_experiment
    )
    if not is_successful(test_setup_result):
        return {"functionName": params.functionName, "status": "error", "message": test_setup_result.failure()}
    generated_tests_list: GeneratedTestsList
    optimizations_set: OptimizationSet
    generated_tests_list, _, concolic__test_str, optimizations_set = test_setup_result.unwrap()

    generated_tests: list[str] = [
        generated_test.generated_original_test_source for generated_test in generated_tests_list.generated_tests
    ]
    optimizations_dict = {
        candidate.optimization_id: {"source_code": candidate.source_code, "explanation": candidate.explanation}
        for candidate in optimizations_set.control + optimizations_set.experiment
    }

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": {"generated_tests": generated_tests, "optimizations": optimizations_dict},
    }


@server.feature("performFunctionOptimization")
def perform_function_optimization(  # noqa: PLR0911
    server: CodeflashLanguageServer, params: FunctionOptimizationParams
) -> dict[str, str]:
    server.show_message_log(f"Starting optimization for function: {params.functionName}", "Info")
    current_function = server.optimizer.current_function_being_optimized

    if not current_function:
        server.show_message_log(f"No current function being optimized for {params.functionName}", "Error")
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": "No function currently being optimized",
        }

    module_prep_result = server.optimizer.prepare_module_for_optimization(current_function.file_path)

    validated_original_code, original_module_ast = module_prep_result

    function_optimizer = server.optimizer.create_function_optimizer(
        current_function,
        function_to_optimize_source_code=validated_original_code[current_function.file_path].source_code,
        original_module_ast=original_module_ast,
        original_module_path=current_function.file_path,
        function_to_tests=server.optimizer.discovered_tests or {},
    )

    server.optimizer.current_function_optimizer = function_optimizer
    if not function_optimizer:
        return {"functionName": params.functionName, "status": "error", "message": "No function optimizer found"}

    initialization_result = function_optimizer.can_be_optimized()
    if not is_successful(initialization_result):
        return {"functionName": params.functionName, "status": "error", "message": initialization_result.failure()}

    should_run_experiment, code_context, original_helper_code = initialization_result.unwrap()

    test_setup_result = function_optimizer.generate_and_instrument_tests(
        code_context, should_run_experiment=should_run_experiment
    )
    if not is_successful(test_setup_result):
        return {"functionName": params.functionName, "status": "error", "message": test_setup_result.failure()}
    (
        generated_tests,
        function_to_concolic_tests,
        concolic_test_str,
        optimizations_set,
        generated_test_paths,
        generated_perf_test_paths,
        instrumented_unittests_created_for_function,
        original_conftest_content,
    ) = test_setup_result.unwrap()

    baseline_setup_result = function_optimizer.setup_and_establish_baseline(
        code_context=code_context,
        original_helper_code=original_helper_code,
        function_to_concolic_tests=function_to_concolic_tests,
        generated_test_paths=generated_test_paths,
        generated_perf_test_paths=generated_perf_test_paths,
        instrumented_unittests_created_for_function=instrumented_unittests_created_for_function,
        original_conftest_content=original_conftest_content,
    )

    if not is_successful(baseline_setup_result):
        return {"functionName": params.functionName, "status": "error", "message": baseline_setup_result.failure()}

    (
        function_to_optimize_qualified_name,
        function_to_all_tests,
        original_code_baseline,
        test_functions_to_remove,
        file_path_to_helper_classes,
    ) = baseline_setup_result.unwrap()

    best_optimization = function_optimizer.find_and_process_best_optimization(
        optimizations_set=optimizations_set,
        code_context=code_context,
        original_code_baseline=original_code_baseline,
        original_helper_code=original_helper_code,
        file_path_to_helper_classes=file_path_to_helper_classes,
        function_to_optimize_qualified_name=function_to_optimize_qualified_name,
        function_to_all_tests=function_to_all_tests,
        generated_tests=generated_tests,
        test_functions_to_remove=test_functions_to_remove,
        concolic_test_str=concolic_test_str,
    )

    if not best_optimization:
        server.show_message_log(
            f"No best optimizations found for function {function_to_optimize_qualified_name}", "Warning"
        )
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": f"No best optimizations found for function {function_to_optimize_qualified_name}",
        }

    optimized_source = best_optimization.candidate.source_code
    speedup = original_code_baseline.runtime / best_optimization.runtime

    server.show_message_log(f"Optimization completed for {params.functionName} with {speedup:.2f}x speedup", "Info")

    # CRITICAL: Clear the function filter after optimization to prevent state corruption
    server.optimizer.args.function = None
    server.show_message_log("Cleared function filter to prevent state corruption", "Info")

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": "Optimization completed successfully",
        "extra": f"Speedup: {speedup:.2f}x faster",
        "optimization": optimized_source,
    }
