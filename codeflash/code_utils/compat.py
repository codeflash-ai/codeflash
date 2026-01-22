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

    codeflash_temp_dir: Path
    codeflash_cache_dir: Path
    codeflash_cache_db: Path


def is_compiled_or_bundled_binary() -> bool:
    """Check if running in a compiled/bundled binary."""
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        return True

    return "__compiled__" in globals()


def _find_python_executable() -> str:
    """Find the appropriate Python executable.

    For compiled binaries, searches for venv in cwd/parent dirs, then falls back to system Python.
    For normal execution, returns sys.executable.
    """
    if not is_compiled_or_bundled_binary():
        return sys.executable

    # Search for venv in current directory and parent directories
    current_dir = Path.cwd()
    venv_names = [".venv", "venv"]
    python_names = ["python3", "python"] if os.name != "nt" else ["python.exe"]

    # Walk up directory tree looking for venv
    for parent in [current_dir, *current_dir.parents]:
        for venv_name in venv_names:
            venv_dir = parent / venv_name
            if venv_dir.is_dir():
                # Check for Python executable in venv
                bin_dir = venv_dir / ("bin" if os.name != "nt" else "Scripts")
                for python_name in python_names:
                    python_path = bin_dir / python_name
                    if python_path.is_file():
                        return str(python_path)

    # Fall back to system Python
    for python_name in python_names:
        system_python = shutil.which(python_name)
        if system_python:
            return system_python

    # Last resort: return sys.executable (even though it may not work)
    return sys.executable


class Compat:
    # os-independent newline
    LF: str = os.linesep

    IS_POSIX: bool = os.name != "nt"

    @property
    def SAFE_SYS_EXECUTABLE(self) -> str:
        return Path(_find_python_executable()).as_posix()

    @property
    def codeflash_cache_dir(self) -> Path:
        return Path(user_config_dir(appname="codeflash", appauthor="codeflash-ai", ensure_exists=True))

    @property
    def codeflash_temp_dir(self) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "codeflash"
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    @property
    def codeflash_cache_db(self) -> Path:
        return self.codeflash_cache_dir / "codeflash_cache.db"


_compat = Compat()


codeflash_temp_dir = _compat.codeflash_temp_dir
codeflash_cache_dir = _compat.codeflash_cache_dir
codeflash_cache_db = _compat.codeflash_cache_db
LF = _compat.LF
SAFE_SYS_EXECUTABLE = _compat.SAFE_SYS_EXECUTABLE
IS_POSIX = _compat.IS_POSIX


@lru_cache(maxsize=1)
def get_jedi_environment() -> "InterpreterEnvironment | None":
    """Get the appropriate Jedi environment based on execution context.

    Returns InterpreterEnvironment for compiled/bundled binaries to avoid
    subprocess spawning issues. Returns None for normal Python execution.
    """
    if not is_compiled_or_bundled_binary():
        return None

    try:
        from jedi.api.environment import InterpreterEnvironment

        from codeflash.cli_cmds.console import logger

        logger.warning("Creating Jedi InterpreterEnvironment for compiled/bundled binary")
        logger.warning(f"sys.executable: {sys.executable}")
        logger.warning(f"sys.prefix: {sys.prefix}")

        # Check if jedi typeshed exists
        try:
            import jedi
            jedi_path = Path(jedi.__file__).parent
            typeshed_path = jedi_path / "third_party" / "typeshed"
            logger.warning(f"Jedi package location: {jedi_path}")
            logger.warning(f"Typeshed path exists: {typeshed_path.exists()}")
            if typeshed_path.exists():
                stdlib_path = typeshed_path / "stdlib"
                logger.warning(f"Typeshed stdlib exists: {stdlib_path.exists()}")
                if stdlib_path.exists():
                    # List first few items to verify
                    items = list(stdlib_path.iterdir())[:5]
                    logger.warning(f"First few stdlib items: {[str(p.name) for p in items]}")
        except Exception as e:
            logger.warning(f"Error checking typeshed location: {e}")

        env = InterpreterEnvironment()
        logger.warning(f"InterpreterEnvironment created with sys_path: {env.get_sys_path()[:3]}...")
        return env
    except (ImportError, AttributeError) as e:
        from codeflash.cli_cmds.console import logger
        logger.warning(f"Could not import InterpreterEnvironment, falling back to default: {e}")
        return None
    except Exception as e:
        from codeflash.cli_cmds.console import logger
        logger.warning(f"Error creating InterpreterEnvironment: {e}", exc_info=True)
        return None
