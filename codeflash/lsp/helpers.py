import os
from functools import lru_cache
from typing import Any, Callable


@lru_cache(maxsize=1)
def is_LSP_enabled() -> bool:
    return os.getenv("CODEFLASH_LSP", default="false").lower() == "true"


def lsp_log(msg: str, actual_log_fn: Callable[[str, Any, Any], None], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    if is_LSP_enabled():
        actual_log_fn(f"::::{msg}", *args, **kwargs)
    else:
        actual_log_fn(msg, *args, **kwargs)
