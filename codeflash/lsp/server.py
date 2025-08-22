from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP
from lsprotocol.types import INITIALIZE, LogMessageParams, MessageType
from pygls import uris
from pygls.protocol import LanguageServerProtocol, lsp_method
from pygls.server import LanguageServer

if TYPE_CHECKING:
    from lsprotocol.types import InitializeParams, InitializeResult

    from codeflash.optimization.optimizer import Optimizer


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
                server.prepare_optimizer_arguments(pyproject_toml_path)
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
        self.optimizer: Optimizer | None = None
        self.args = None
        self.mcp = FastMCP("codeflash-mcp-server")
        self._setup_mcp_tools()

    def prepare_optimizer_arguments(self, config_file: Path) -> None:
        from codeflash.cli_cmds.cli import parse_args, process_pyproject_config

        args = parse_args()
        args.config_file = config_file
        args.no_pr = True  # LSP server should not create PRs
        args.worktree = True
        args = process_pyproject_config(args)
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
        self.lsp.notify("window/logMessage", log_params)

    def _setup_mcp_tools(self) -> None:
        """Setup MCP tools and resources for code optimization."""

        @self.mcp.tool()
        def optimize_code(file: str, function: str) -> dict[str, Any]:
            """Optimize a specific function in a file using Codeflash AI optimization.

            Args:
                file: Path to the Python file containing the function
                function: Name of the function to optimize

            Returns:
                Dictionary containing optimization results including speedup and optimized code

            """
            try:
                if not self.optimizer:
                    return {"status": "error", "message": "Optimizer not initialized. Please provide API key first."}

                file_path = Path(file)
                if not file_path.exists():
                    return {"status": "error", "message": f"File {file} not found"}

                # Use the existing optimization logic
                from types import SimpleNamespace

                from codeflash.lsp.beta import FunctionOptimizationParams

                # Create mock text document
                mock_doc = SimpleNamespace(uri=f"file://{file_path.absolute()}")
                params = FunctionOptimizationParams(textDocument=mock_doc, functionName=function)

                # Initialize optimization
                init_result = self._mcp_initialize_function_optimization(params)
                if init_result.get("status") != "success":
                    return init_result

                # Discover tests
                test_result = self._mcp_discover_function_tests(params)
                if test_result.get("status") != "success":
                    return test_result

                # Perform optimization
                opt_result = self._mcp_perform_function_optimization(params)
                return opt_result

            except Exception as e:
                return {"status": "error", "message": f"Optimization failed: {e!s}"}

        @self.mcp.tool()
        def get_optimizable_functions(file: str) -> dict[str, Any]:
            """Get list of functions that can be optimized in a file.

            Args:
                file: Path to the Python file to analyze

            Returns:
                Dictionary containing list of optimizable function names

            """
            try:
                if not self.optimizer:
                    return {"status": "error", "message": "Optimizer not initialized. Please provide API key first."}

                file_path = Path(file)
                if not file_path.exists():
                    return {"status": "error", "message": f"File {file} not found"}

                from types import SimpleNamespace

                from codeflash.lsp.beta import OptimizableFunctionsParams

                mock_doc = SimpleNamespace(uri=f"file://{file_path.absolute()}")
                params = OptimizableFunctionsParams(textDocument=mock_doc)

                result = self._mcp_get_optimizable_functions(params)
                return result

            except Exception as e:
                return {"status": "error", "message": f"Failed to get optimizable functions: {e!s}"}

        @self.mcp.tool()
        def set_api_key(api_key: str) -> dict[str, str]:
            """Set Codeflash API key for optimization services.

            Args:
                api_key: Codeflash API key (should start with 'cf-')

            Returns:
                Dictionary with status and message

            """
            try:
                if not api_key.startswith("cf-"):
                    return {"status": "error", "message": "Invalid API key format. Should start with 'cf-'"}

                from codeflash.lsp.beta import ProvideApiKeyParams

                params = ProvideApiKeyParams(api_key=api_key)
                result = self._mcp_provide_api_key(params)
                return result

            except Exception as e:
                return {"status": "error", "message": f"Failed to set API key: {e!s}"}

        @self.mcp.resource("file://{path}")
        def get_file_content(path: str) -> str:
            """Get the content of a file.

            Args:
                path: Path to the file

            Returns:
                File content as string

            """
            try:
                file_path = Path(path)
                if file_path.exists():
                    return file_path.read_text()
                return f"File not found: {path}"
            except Exception as e:
                return f"Error reading file: {e!s}"

        @self.mcp.resource("codeflash://functions/{file_path}")
        def get_optimizable_functions_resource(file_path: str) -> str:
            """Get optimizable functions in a file as a resource.

            Args:
                file_path: Path to the Python file

            Returns:
                JSON string with optimizable functions

            """
            try:
                result = get_optimizable_functions(file_path)
                return json.dumps(result, indent=2)
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

    def _mcp_initialize_function_optimization(self, params) -> dict[str, str]:
        """MCP wrapper for function optimization initialization."""
        from codeflash.lsp.beta import initialize_function_optimization

        return initialize_function_optimization(self, params)

    def _mcp_discover_function_tests(self, params) -> dict[str, str]:
        """MCP wrapper for test discovery."""
        from codeflash.lsp.beta import discover_function_tests

        return discover_function_tests(self, params)

    def _mcp_perform_function_optimization(self, params) -> dict[str, str]:
        """MCP wrapper for function optimization."""
        from codeflash.lsp.beta import perform_function_optimization

        return perform_function_optimization(self, params)

    def _mcp_get_optimizable_functions(self, params) -> dict[str, Any]:
        """MCP wrapper for getting optimizable functions."""
        from codeflash.lsp.beta import get_optimizable_functions

        return get_optimizable_functions(self, params)

    def _mcp_provide_api_key(self, params) -> dict[str, str]:
        """MCP wrapper for providing API key."""
        from codeflash.lsp.beta import provide_api_key

        return provide_api_key(self, params)

    def get_mcp_server(self) -> FastMCP:
        """Get the MCP server instance."""
        return self.mcp
