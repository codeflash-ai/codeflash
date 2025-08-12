import os
from functools import lru_cache


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.environ.get("CODEFLASH_LSP", "false").lower() == "true"
