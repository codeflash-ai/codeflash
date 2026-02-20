import os
import sys
import tempfile
from pathlib import Path

from platformdirs import user_config_dir

LF: str = os.linesep
IS_POSIX: bool = os.name != "nt"
SAFE_SYS_EXECUTABLE: str = Path(sys.executable).as_posix()

codeflash_cache_dir: Path = Path(user_config_dir(appname="codeflash", appauthor="codeflash-ai", ensure_exists=True))

codeflash_temp_dir: Path = Path(tempfile.gettempdir()) / "codeflash"
codeflash_temp_dir.mkdir(parents=True, exist_ok=True)

codeflash_cache_db: Path = codeflash_cache_dir / "codeflash_cache.db"
