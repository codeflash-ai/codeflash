from __future__ import annotations

import os
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_config_dir

if TYPE_CHECKING:
    from jedi.api.environment import InterpreterEnvironment


def is_compiled_or_bundled_binary() -> bool:
    """Check if running in a compiled/bundled binary."""
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        return True

    return "__compiled__" in globals()


def find_executable_in_venv(exe_names: list[str]) -> str | None:
    """Find an executable in venv directories.

    Searches for venv in current directory and parent directories.
    Returns the first matching executable found, or None if not found.

    Args:
        exe_names: List of possible executable names (e.g., ["python3", "python"])

    Returns:
        Path to executable if found, None otherwise

    """
    if not is_compiled_or_bundled_binary():
        return None

    current_dir = Path.cwd()
    venv_names = [".venv", "venv"]

    # Walk up directory tree looking for venv
    for parent in [current_dir, *current_dir.parents]:
        for venv_name in venv_names:
            venv_dir = parent / venv_name
            if venv_dir.is_dir():
                bin_dir = venv_dir / ("bin" if os.name != "nt" else "Scripts")
                for exe_name in exe_names:
                    exe_path = bin_dir / exe_name
                    if exe_path.is_file():
                        return str(exe_path)

    return None


def _find_python_executable() -> str:
    """Find the appropriate Python executable.

    For compiled binaries, searches for venv in cwd/parent dirs, then falls back to system Python.
    For normal execution, returns sys.executable.
    """
    if not is_compiled_or_bundled_binary():
        return sys.executable

    python_names = ["python3", "python"] if os.name != "nt" else ["python.exe"]

    # Try venv first
    venv_python = find_executable_in_venv(python_names)
    if venv_python:
        return venv_python

    # Fall back to system Python
    for python_name in python_names:
        system_python = shutil.which(python_name)
        if system_python:
            return system_python

    # Last resort: return sys.executable (even though it may not work)
    return sys.executable


LF: str = os.linesep
IS_POSIX: bool = os.name != "nt"


@lru_cache(maxsize=1)
def get_safe_sys_executable() -> str:
    """Get a safe Python executable path with forward slashes."""
    return Path(_find_python_executable()).as_posix()


@lru_cache(maxsize=1)
def get_codeflash_cache_dir() -> Path:
    """Get the codeflash cache directory, creating it if necessary."""
    return Path(user_config_dir(appname="codeflash", appauthor="codeflash-ai", ensure_exists=True))


@lru_cache(maxsize=1)
def get_codeflash_temp_dir() -> Path:
    """Get the codeflash temp directory, creating it if necessary."""
    temp_dir = Path(tempfile.gettempdir()) / "codeflash"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@lru_cache(maxsize=1)
def get_codeflash_cache_db() -> Path:
    """Get the path to the codeflash cache database."""
    return get_codeflash_cache_dir() / "codeflash_cache.db"


@lru_cache(maxsize=1)
def get_jedi_environment() -> InterpreterEnvironment | None:
    """Get the appropriate Jedi environment based on execution context.

    Returns InterpreterEnvironment for compiled/bundled binaries to avoid
    subprocess spawning issues. Returns None for normal Python execution.
    """
    if not is_compiled_or_bundled_binary():
        return None

    try:
        from jedi.api.environment import InterpreterEnvironment

        from codeflash.cli_cmds.console import logger

        logger.debug("Using Jedi InterpreterEnvironment for compiled/bundled binary")
        return InterpreterEnvironment()
    except Exception as e:
        from codeflash.cli_cmds.console import logger

        logger.warning(f"Could not create InterpreterEnvironment: {e}")
        return None
