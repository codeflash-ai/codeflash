from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    server: CodeflashLanguageServer,  # noqa: ARG001
    params: OptimizableFunctionsParams,
) -> dict[str, list[str]]:
    return {params.textDocument.uri: ["example"]}


if __name__ == "__main__":
    from codeflash.cli_cmds.console import console

    console.quiet = True
    server.start_io()
