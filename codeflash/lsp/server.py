from __future__ import annotations

from typing import TYPE_CHECKING

from lsprotocol.types import LogMessageParams, MessageType
from pygls.lsp.server import LanguageServer
from pygls.protocol import LanguageServerProtocol

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.models.models import CodeOptimizationContext
    from codeflash.optimization.optimizer import Optimizer


class CodeflashLanguageServerProtocol(LanguageServerProtocol):
    _server: CodeflashLanguageServer


class CodeflashLanguageServer(LanguageServer):
    def __init__(self, name: str, version: str, protocol_cls: type[LanguageServerProtocol]) -> None:
        super().__init__(name, version, protocol_cls=protocol_cls)
        self.initialized: bool = False
        self.optimizer: Optimizer | None = None
        self.args = None
        self.current_optimization_init_result: tuple[bool, CodeOptimizationContext, dict[Path, str]] | None = None

    def prepare_optimizer_arguments(self, config_file: Path) -> None:
        from codeflash.cli_cmds.cli import parse_args

        args = parse_args()
        args.config_file = config_file
        args.no_pr = True  # LSP server should not create PRs
        args.worktree = True
        self.args = args
        # avoid initializing the optimizer during initialization, because it can cause an error if the api key is invalid

    def show_message_log(self, message: str, message_type: str) -> None:
        """Send a log message to the client's output channel.

        Args:
            message: The message to log
            message_type: String type - "Info", "Warning", "Error", or "Log"

        """
        # Convert string message type to LSP MessageType enum
        type_mapping = {
            "Info": MessageType.Info,
            "Warning": MessageType.Warning,
            "Error": MessageType.Error,
            "Log": MessageType.Log,
            "Debug": MessageType.Debug,
        }

        lsp_message_type = type_mapping.get(message_type, MessageType.Info)

        # Send log message to client (appears in output channel)
        log_params = LogMessageParams(type=lsp_message_type, message=message)
        self.protocol.notify("window/logMessage", log_params)

    def cleanup_the_optimizer(self) -> bool:
        self.current_optimization_init_result = None
        if not self.optimizer:
            return False
        try:
            self.optimizer.cleanup_temporary_paths()
            # restore args and test cfg
            if self.optimizer.original_args_and_test_cfg:
                self.optimizer.args, self.optimizer.test_cfg = self.optimizer.original_args_and_test_cfg
            self.optimizer.args.function = None
            self.optimizer.current_worktree = None
            self.optimizer.current_function_optimizer = None
        except Exception:
            self.show_message_log("Failed to cleanup optimizer", "Error")
            return False
        return True

    def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        self.cleanup_the_optimizer()
        super().shutdown()
