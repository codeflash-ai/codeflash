from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lsprotocol.types import INITIALIZE
from pygls import uris
from pygls.protocol import LanguageServerProtocol, lsp_method
from pygls.server import LanguageServer

if TYPE_CHECKING:
    from lsprotocol.types import InitializeParams, InitializeResult


class CodeflashLanguageServerProtocol(LanguageServerProtocol):
    _server: CodeflashLanguageServer

    @lsp_method(INITIALIZE)
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        server = self._server
        initialize_result: InitializeResult = super().lsp_initialize(params)

        workspace_uri = params.root_uri
        if workspace_uri:
            workspace_path = uris.to_fs_path(workspace_uri)
            pyproject_toml_path = self._find_pyproject_toml(workspace_path)
            if pyproject_toml_path:
                server.initialize_optimizer(pyproject_toml_path)
                server.show_message(f"Found pyproject.toml at: {pyproject_toml_path}")
            else:
                server.show_message("No pyproject.toml found in workspace.")
        else:
            server.show_message("No workspace URI provided.")

        return initialize_result

    def _find_pyproject_toml(self, workspace_path: str) -> Path | None:
        workspace_path_obj = Path(workspace_path)
        for file_path in workspace_path_obj.rglob("pyproject.toml"):
            return file_path.resolve()
        return None


class CodeflashLanguageServer(LanguageServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self.optimizer = None

    def initialize_optimizer(self, config_file: Path) -> None:
        from codeflash.cli_cmds.cli import process_pyproject_only
        from codeflash.optimization.optimizer import Optimizer

        args = process_pyproject_only(config_file)
        self.optimizer = Optimizer(args)
