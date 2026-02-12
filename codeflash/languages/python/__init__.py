"""Python language support for Codeflash.

This module provides the PythonSupport class which wraps the existing
Python-specific implementations (LibCST, Jedi, pytest, etc.) to conform
to the LanguageSupport protocol.
"""

from codeflash.languages.python.call_graph import CallGraph
from codeflash.languages.python.support import PythonSupport

__all__ = ["CallGraph", "PythonSupport"]
