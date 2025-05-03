import os
import sys
from pathlib import Path

from platformdirs import user_config_dir

# os-independent newline
# important for any user-facing output or files we write
# make sure to use this in f-strings e.g. f"some string{LF}"
# you can use "[^f]\".*\{LF\}\" to find any lines in your code that use this without the f-string
LF: str = os.linesep


SAFE_SYS_EXECUTABLE: str = Path(sys.executable).as_posix()

IS_POSIX = os.name != "nt"


codeflash_cache_dir = Path(user_config_dir(appname="codeflash", appauthor="codeflash-ai", ensure_exists=True))
