"""Dedicated entry point for the Codeflash Language Server.

Initializes the server and redirects its logs to stderr so that the
VS Code client can display them in the output channel.

This script is run by the VS Code extension and is not intended to be
executed directly by users.
"""

from codeflash.lsp.beta import server
from codeflash.lsp.lsp_logger import setup_logging

if __name__ == "__main__":
    # Set up logging
    root_logger = setup_logging()
    root_logger.info("Starting Codeflash Language Server...")

    # Start the language server
    server.start_io()
