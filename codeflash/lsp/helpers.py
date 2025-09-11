import os
from functools import lru_cache
from typing import Any, Callable

skip_lsp_log_prefix = "!lsp:"


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.getenv("CODEFLASH_LSP", default="false").lower() == "true"


def enhanced_log(msg: str, actual_log_fn: Callable[[str, Any, Any], None], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    lsp_enabled = is_LSP_enabled()
    str_msg = isinstance(msg, str)
    skip_lsp_log = str_msg and msg.strip().startswith(skip_lsp_log_prefix)

    if skip_lsp_log:
        msg = msg[len(skip_lsp_log_prefix) :]

    # normal cli mode
    if not lsp_enabled:
        actual_log_fn(msg, *args, **kwargs)
        return

    #### LSP mode ####
    if skip_lsp_log or not str_msg:
        return

    actual_log_fn(f"::::{msg}", *args, **kwargs)
