"""Dual server runner that can start either LSP or MCP server based on arguments.

Usage:
    python dual_server.py --mode lsp     # Start LSP server
    python dual_server.py --mode mcp     # Start MCP server  
    python dual_server.py --help         # Show help
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[Codeflash-Server] %(asctime)s [%(levelname)s] %(message)s"))

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    return root_logger


def prepare_server_config(server: CodeflashLanguageServer) -> None:
    """Prepare server configuration by finding pyproject.toml."""
    try:
        cwd = Path.cwd()
        pyproject_path = None
        
        # Look for pyproject.toml in current directory and parents
        for parent in [cwd] + list(cwd.parents):
            potential_pyproject = parent / "pyproject.toml"
            if potential_pyproject.exists():
                pyproject_path = potential_pyproject
                break
        
        if pyproject_path:
            server.prepare_optimizer_arguments(pyproject_path)
            logging.info(f"Found pyproject.toml at: {pyproject_path}")
        else:
            logging.info("No pyproject.toml found, server will require manual configuration")
            
    except Exception as e:
        logging.warning(f"Could not prepare optimizer arguments: {e}")


def start_lsp_server() -> None:
    """Start the LSP server."""
    from codeflash.lsp.beta import server
    
    log = setup_logging()
    log.info("Starting Codeflash Language Server (LSP mode)...")
    
    prepare_server_config(server)
    
    log.info("LSP Server ready with custom features for VS Code extension")
    server.start_io()


async def start_mcp_server() -> None:
    """Start the MCP server."""
    log = setup_logging()
    log.info("Starting Codeflash MCP Server...")

    # Create server instance
    server = CodeflashLanguageServer(
        "codeflash-mcp-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol
    )
    
    prepare_server_config(server)

    # Get the MCP server instance and run it
    mcp_server = server.get_mcp_server()
    
    log.info("MCP Server ready with tools:")
    log.info("  - optimize_code(file, function): Optimize a function in a file")
    log.info("  - get_optimizable_functions(file): Get list of optimizable functions")
    log.info("  - set_api_key(api_key): Set Codeflash API key")
    log.info("MCP Server ready with resources:")
    log.info("  - file://{path}: Get file content")
    log.info("  - codeflash://functions/{file_path}: Get optimizable functions as JSON")
    
    # Run the MCP server
    await mcp_server.run()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Codeflash Dual Server - Run as either LSP or MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dual_server.py --mode lsp     # Start Language Server Protocol server
  python dual_server.py --mode mcp     # Start Model Context Protocol server
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["lsp", "mcp"], 
        required=True,
        help="Server mode: 'lsp' for Language Server Protocol, 'mcp' for Model Context Protocol"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "lsp":
            start_lsp_server()
        elif args.mode == "mcp":
            asyncio.run(start_mcp_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()