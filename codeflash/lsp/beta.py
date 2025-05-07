from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pygls import uris

from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol

if TYPE_CHECKING:
    from lsprotocol import types


LSP_MODE = True


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
    server.optimizer.lsp_mode = True
    server.optimizer.args.replay_test = None
    server.optimizer.args.optimize_all = None
    server.optimizer.args.file = file_path
    server.optimizer.args.ignore_paths = []
    server.optimizer.args.optimize_all = None
    server.optimizer.args.replay_test = None
    server.optimizer.args.only_get_this_function = None
    server.optimizer.args.function = None
    server.optimizer.args.benchmark = None
    server.optimizer.server = server

    optimizable_funcs, _ = server.optimizer.discover_functions()
    path_to_qualified_names = {}
    for path, functions in optimizable_funcs.items():
        path_to_qualified_names[path.as_posix()] = [func.qualified_name for func in functions]
    return path_to_qualified_names


@server.feature("optimizeFunction")
def optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, str]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    server.optimizer.args.function = params.functionName
    server.optimizer.args.file = file_path
    optimizable_funcs, _ = server.optimizer.discover_functions()
    if not optimizable_funcs:
        return {"functionName": params.functionName, "status": "not found", "args": None}
    fto = optimizable_funcs.popitem()[1][0]
    return {"functionName": params.functionName, "status": "success", "info": fto.server_info}


if __name__ == "__main__":
    server.start_io()
