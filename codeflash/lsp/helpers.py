import os
from functools import lru_cache
from typing import Any, Callable


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.getenv("CODEFLASH_LSP", default="false").lower() == "true"


def enhanced_log(msg: str, actual_log_fn: Callable[[str, Any, Any], None], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    lsp_enabled = is_LSP_enabled()

    # normal cli moded
    if not lsp_enabled:
        actual_log_fn(msg, *args, **kwargs)
        return

    #### LSP mode ####
    if type(msg) != str:  # noqa: E721
        return

    if msg.startswith("Nonzero return code"):
        # skip logging the failed tests msg to the client
        return

    actual_log_fn(f"::::{msg}", *args, **kwargs)
