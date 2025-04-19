from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    optimizable_funcs, _ = get_functions_to_optimize(  # we can consider caching this later on, but for now it's fine
        file=file_path,
        test_cfg=server.optimizer.test_cfg,
        ignore_paths=[],
        project_root=server.optimizer.test_cfg.project_root_path,
        module_root=server.optimizer.args.module_root,
        optimize_all=None,
        replay_test=None,
        only_get_this_function=None,
    )
    path_to_qualified_names = {}
    for path, functions in optimizable_funcs.items():
        path_to_qualified_names[path.as_posix()] = [func.qualified_name for func in functions]
    return path_to_qualified_names


@server.feature("optimizeFunction")
def optimize_function(server: CodeflashLanguageServer, params: OptimizeFunctionParams) -> dict[str, Any]:
    file_path = Path(uris.to_fs_path(params.textDocument.uri))
    function_name = params.functionName

    return {"functionName": function_name}


if __name__ == "__main__":
    from codeflash.discovery.functions_to_optimize import get_functions_to_optimize

    server.start_io()
