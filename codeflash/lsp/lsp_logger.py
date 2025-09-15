from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.lsp.lsp_message import LspTextMessage


@dataclass
class LspMessageTags:
    # always set default values for message tags
    not_lsp: bool = False  # !lsp         (prevent the message from being sent to the LSP)
    force_lsp: bool = False  # lsp          (you can use this to force a message to be sent to the LSP even if the level is not supported)
    loading: bool = False  # loading      (you can use this to indicate that the message is a loading message)

    h1: bool = False  # h1
    h2: bool = False  # h2
    h3: bool = False  # h3
    h4: bool = False  # h4


def add_heading_tags(msg: str, tags: LspMessageTags) -> str:
    if tags.h1:
        return "# " + msg
    if tags.h2:
        return "## " + msg
    if tags.h3:
        return "### " + msg
    if tags.h4:
        return "#### " + msg
    return msg


def extract_tags(msg: str) -> tuple[LspMessageTags, str]:
    if not isinstance(msg, str):
        return LspMessageTags(), msg

    parts = msg.split("|tags|")
    if len(parts) == 2:
        message_tags = LspMessageTags()
        tags = [tag.strip() for tag in parts[0].split(",")]
        if "!lsp" in tags:
            message_tags.not_lsp = True
        if "lsp" in tags:
            message_tags.force_lsp = True
        if "loading" in tags:
            message_tags.loading = True
        if "h1" in tags:
            message_tags.h1 = True
        if "h2" in tags:
            message_tags.h2 = True
        if "h3" in tags:
            message_tags.h3 = True
        if "h4" in tags:
            message_tags.h4 = True
        return message_tags, parts[1]

    return LspMessageTags(), msg


supported_lsp_log_levels = ("info", "debug")


def enhanced_log(
    msg: str,
    actual_log_fn: Callable[[str, Any, Any], None],
    level: str,
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
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
    unsupported_level = level not in supported_lsp_log_levels
    if not tags.force_lsp and (tags.not_lsp or unsupported_level):
        return

    if is_normal_text_message:
        clean_msg = add_heading_tags(clean_msg, tags)
        clean_msg = LspTextMessage(text=clean_msg, takes_time=tags.loading).serialize()

    actual_log_fn(clean_msg, *args, **kwargs)
