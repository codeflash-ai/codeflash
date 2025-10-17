import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import user_config_dir

if TYPE_CHECKING:
    codeflash_temp_dir: Path
    codeflash_cache_dir: Path
    codeflash_cache_db: Path


class Compat:
    # os-independent newline
    LF: str = os.linesep

    SAFE_SYS_EXECUTABLE: str = Path(sys.executable).as_posix()

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
