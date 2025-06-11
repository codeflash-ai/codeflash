from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pygls import uris

from codeflash.either import is_successful
from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol

if TYPE_CHECKING:
    from lsprotocol import types


@dataclass
class OptimizableFunctionsParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815


@dataclass
class OptimizeFunctionParams:
    textDocument: types.TextDocumentIdentifier  # noqa: N815
    functionName: str  # noqa: N815


server = CodeflashLanguageServer("codeflash-language-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol)


@server.feature("getOptimizableFunctions")
def get_optimizable_functions(
    server: CodeflashLanguageServer, params: OptimizableFunctionsParams
) -> dict[str, list[str]]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.optimizer.args.file = file_path
    server.optimizer.args.previous_checkpoint_functions = False
    optimizable_funcs, _ = server.optimizer.get_optimizable_functions()
    path_to_qualified_names = {}
    for path, functions in optimizable_funcs.items():
        path_to_qualified_names[path.as_posix()] = [func.qualified_name for func in functions]
    return path_to_qualified_names


@server.feature("optimizeFunction")
def optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, str]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.optimizer.args.function = params.functionName
    server.optimizer.args.file = file_path
    optimizable_funcs, _ = server.optimizer.get_optimizable_functions()
    if not optimizable_funcs:
        return {"functionName": params.functionName, "status": "not found", "args": None}
    fto = optimizable_funcs.popitem()[1][0]
    server.optimizer.current_function_being_optimized = fto
    return {"functionName": params.functionName, "status": "success", "info": fto.server_info}


@server.feature("second_step_in_optimize_function")
def second_step_in_optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, str]:
    current_function = server.optimizer.current_function_being_optimized

    optimizable_funcs = {current_function.file_path: [current_function]}

    function_to_tests, num_discovered_tests = server.optimizer.discover_tests(optimizable_funcs)
    # mocking in order to get things going
    return {"functionName": params.functionName, "status": "success", "generated_tests": str(num_discovered_tests)}


@server.feature("third_step_in_optimize_function")
def third_step_in_optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, str]:
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
        return {
            "functionName": params.functionName,
            "status": "error",
            "message": f"No best optimizations found for function {function_to_optimize_qualified_name}",
        }

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": "Optimization completed successfully",
        "extra": f"Speedup: {original_code_baseline.runtime / best_optimization.runtime:.2f}x faster",
    }


if __name__ == "__main__":
    from codeflash.cli_cmds.console import console

    console.quiet = True
    server.start_io()
