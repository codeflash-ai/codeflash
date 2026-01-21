"""Compatibility utilities for jedi in compiled binaries."""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE, is_compiled_or_bundled_binary

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def safe_jedi_executable() -> Generator[None, None, None]:
    """
    Context manager that temporarily replaces sys.executable with SAFE_SYS_EXECUTABLE.

    This allows jedi to spawn subprocesses for inference operations in compiled mode,
    where sys.executable points to a temporary extraction path that doesn't exist.

    Usage:
        with safe_jedi_executable():
            script = jedi.Script(...)
            name.type  # This would normally fail in compiled mode
    """
    if not is_compiled_or_bundled_binary():
        # In normal mode, no patching needed
        yield
        return

    # Save original sys.executable
    original_executable = sys.executable

    try:
        # Replace with real Python interpreter
        sys.executable = SAFE_SYS_EXECUTABLE
        yield
    finally:
        # Always restore original value
        sys.executable = original_executable
