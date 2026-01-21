import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_config_dir

if TYPE_CHECKING:
    codeflash_temp_dir: Path
    codeflash_cache_dir: Path
    codeflash_cache_db: Path


def is_compiled_or_bundled_binary() -> bool:
    """Check if running from a compiled binary (Nuitka or PyInstaller)."""
    return (
        hasattr(sys, "__compiled__")
        or "onefile_" in sys.executable
        or getattr(sys, "frozen", False)
        or hasattr(sys, "_MEIPASS")
    )


def get_python_executable() -> str:
    """Get the Python executable path, handling Nuitka and PyInstaller compiled binaries."""
    if is_compiled_or_bundled_binary():
        # When running as a compiled binary, sys.executable points to the binary,
        # not a Python interpreter. We need to find a Python with access to dependencies.

        # 1. Check VIRTUAL_ENV environment variable (most reliable for active venv)
        venv_path = os.environ.get("VIRTUAL_ENV")
        if venv_path:
            venv_python = Path(venv_path) / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python3")
            if venv_python.exists():
                return venv_python.as_posix()

        # 2. Look for .venv in current working directory or parent directories
        cwd = Path.cwd()
        for directory in [cwd, *cwd.parents]:
            venv_python = directory / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python3")
            if venv_python.exists():
                return venv_python.as_posix()

        # 3. Fall back to system Python (may not have dependencies - this is a limitation)
        python_exe = shutil.which("python3") or shutil.which("python")
        if python_exe:
            return Path(python_exe).as_posix()

    return Path(sys.executable).as_posix()


class Compat:
    # os-independent newline
    LF: str = os.linesep

    SAFE_SYS_EXECUTABLE: str = get_python_executable()

    IS_POSIX: bool = os.name != "nt"

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
