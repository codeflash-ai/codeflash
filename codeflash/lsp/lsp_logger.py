from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.lsp.lsp_message import LspTextMessage


@dataclass
class LspMessageTags:
    # always set default values for message tags
    not_lsp: bool = False
    loading: bool = False


def extract_tags(msg: str) -> tuple[LspMessageTags, str]:
    if not isinstance(msg, str):
        return LspMessageTags(), msg

    parts = msg.split("|tags|")
    if len(parts) == 2:
        message_tags = LspMessageTags()
        tags = [tag.strip() for tag in parts[0].split(",")]
        if "!lsp" in tags:
            message_tags.not_lsp = True
        if "loading" in tags:
            message_tags.loading = True
        return message_tags, parts[1]

    return LspMessageTags(), msg


def enhanced_log(msg: str, actual_log_fn: Callable[[str, Any, Any], None], *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    lsp_enabled = is_LSP_enabled()
    if not lsp_enabled or not isinstance(msg, str):
        actual_log_fn(msg, *args, **kwargs)
        return

    is_lsp_json_message = msg.startswith('{"type"')
    is_normal_text_message = not is_lsp_json_message

    tags = LspMessageTags()
    clean_msg = msg

    if is_normal_text_message:
        tags, clean_msg = extract_tags(msg)

    # normal cli mode
    if not lsp_enabled:
        actual_log_fn(clean_msg, *args, **kwargs)
        return

    #### LSP mode ####
    if tags.not_lsp:
        return

    if is_normal_text_message:
        clean_msg = LspTextMessage(text=clean_msg, takes_time=tags.loading).serialize()

    actual_log_fn(clean_msg, *args, **kwargs)
