from typing import Any, Callable

from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.lsp.lsp_message import LspTextMessage

skip_lsp_log_prefix = "!lsp:"


def enhanced_log(msg: str, actual_log_fn: Callable[[str, Any, Any], None], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    lsp_enabled = is_LSP_enabled()
    str_msg = isinstance(msg, str)
    # if the message starts with !lsp:, it won't be sent to the client
    skip_lsp_log = str_msg and msg.strip().startswith(skip_lsp_log_prefix)

    if skip_lsp_log:
        # get the message without the prefix
        msg = msg[len(skip_lsp_log_prefix) :]

    # normal cli mode
    if not lsp_enabled:
        actual_log_fn(msg, *args, **kwargs)
        return

    #### LSP mode ####
    if skip_lsp_log or not str_msg:
        return

    if not msg.startswith("{"):
        # it is not a json message, use a text message
        msg = LspTextMessage(text=msg).serialize()

    actual_log_fn(msg, *args, **kwargs)
