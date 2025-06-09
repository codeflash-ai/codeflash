from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pygls import uris

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
def second_step_in_optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, str]:  # noqa: ARG001
    return {
        "functionName": params.functionName,
        "status": "success",
        "generated_tests": "5",
        "generated_optimizations": "3",
    }


if __name__ == "__main__":
    from codeflash.cli_cmds.console import console

    console.quiet = True
    server.start_io()
