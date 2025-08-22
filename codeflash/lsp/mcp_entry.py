"""Dedicated entry point for the Codeflash MCP Server.

This script runs the Model Context Protocol server to provide
Codeflash optimization capabilities as MCP tools and resources.
"""

import asyncio
import logging
import sys
from pathlib import Path

from codeflash.lsp.server import CodeflashLanguageServer, CodeflashLanguageServerProtocol


async def main() -> None:
    """Run the Codeflash MCP server."""
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Starting Codeflash MCP Server...")

    # Create the language server instance (which now includes MCP functionality)
    server = CodeflashLanguageServer(
        "codeflash-mcp-server", "v1.0", protocol_cls=CodeflashLanguageServerProtocol
    )

    # Try to find and prepare pyproject.toml if we're in a project directory
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
            log.info(f"Found pyproject.toml at: {pyproject_path}")
        else:
            log.info("No pyproject.toml found, server will require manual configuration")
            
    except Exception as e:
        log.warning(f"Could not prepare optimizer arguments: {e}")

    # Get the MCP server instance and run it
    mcp_server = server.get_mcp_server()
    
    log.info("MCP Server ready with tools: optimize_code, get_optimizable_functions, set_api_key")
    log.info("MCP Server ready with resources: file://{path}, codeflash://functions/{file_path}")
    
    # Run the MCP server
    await mcp_server.run()


if __name__ == "__main__":
    asyncio.run(main())