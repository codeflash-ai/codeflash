"""Dedicated entry point for the Codeflash Language Server.

Initializes the server and redirects its logs to stderr so that the
VS Code client can display them in the output channel.

This script is run by the VS Code extension and is not intended to be
executed directly by users.
"""

import logging
import sys

from codeflash.lsp.beta import server


# Configure logging to stderr for VS Code output channel
def setup_logging() -> logging.Logger:
    # Clear any existing handlers to prevent conflicts
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set up stderr handler for VS Code output channel with [LSP-Server] prefix
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[LSP-Server] %(asctime)s [%(levelname)s]: %(message)s"))

    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    # Also configure the pygls logger specifically
    pygls_logger = logging.getLogger("pygls")
    pygls_logger.setLevel(logging.INFO)

    return root_logger


if __name__ == "__main__":
    # Set up logging
    log = setup_logging()
    log.info("Starting Codeflash Language Server...")

    # Start the language server
    server.start_io()
