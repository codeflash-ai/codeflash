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

    return {
        "functionName": params.functionName,
        "status": "success",
        "message": "Function can be optimized",
        "extra": original_helper_code,
    }


if __name__ == "__main__":
    from codeflash.cli_cmds.console import console

    console.quiet = True
    server.start_io()
