from functools import lru_cache

from codeflash.cli_cmds.console import console


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return console.quiet
