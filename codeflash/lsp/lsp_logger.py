from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.lsp.lsp_message import LSPMessageId, LspTextMessage, message_delimiter

root_logger = None

message_id_prefix = "id:"


@dataclass
class LspMessageTags:
    # always set default values for message tags
    not_lsp: bool = False  # !lsp           (prevent the message from being sent to the LSP)
    lsp: bool = False  # lsp                (lsp only)
    force_lsp: bool = False  # force_lsp    (you can use this to force a message to be sent to the LSP even if the level is not supported)
    loading: bool = False  # loading        (you can use this to indicate that the message is a loading message)
    message_id: Optional[LSPMessageId] = None  # example: id:best_candidate
    highlight: bool = False  # highlight    (you can use this to highlight the message by wrapping it in ``)
    h1: bool = False  # h1
    h2: bool = False  # h2
    h3: bool = False  # h3
    h4: bool = False  # h4


def add_highlight_tags(msg: str, tags: LspMessageTags) -> str:
    if tags.highlight:
        return "`" + msg + "`"
    return msg


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
    delimiter = "|"
    first_delim_idx = msg.find(delimiter)
    if first_delim_idx != -1 and msg.count(delimiter) == 1:
        tags_str = msg[:first_delim_idx]
        content = msg[first_delim_idx + 1 :]
        tags = {tag.strip() for tag in tags_str.split(",")}
        message_tags = LspMessageTags()
        # manually check and set to avoid repeated membership tests
        for tag in tags:
            if tag.startswith(message_id_prefix):
                message_tags.message_id = LSPMessageId(tag[len(message_id_prefix) :]).value
            elif tag == "lsp":
                message_tags.lsp = True
            elif tag == "!lsp":
                message_tags.not_lsp = True
            elif tag == "force_lsp":
                message_tags.force_lsp = True
            elif tag == "loading":
                message_tags.loading = True
            elif tag == "highlight":
                message_tags.highlight = True
            elif tag == "h1":
                message_tags.h1 = True
            elif tag == "h2":
                message_tags.h2 = True
            elif tag == "h3":
                message_tags.h3 = True
            elif tag == "h4":
                message_tags.h4 = True
        return message_tags, content

    return LspMessageTags(), msg


supported_lsp_log_levels = ("info", "debug")


def enhanced_log(
    msg: str | Any,  # noqa: ANN401
    actual_log_fn: Callable[[str, Any, Any], None],
    level: str,
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    if not isinstance(msg, str):
        actual_log_fn(msg, *args, **kwargs)
        return

    is_lsp_json_message = msg.startswith(message_delimiter) and msg.endswith(message_delimiter)
    is_normal_text_message = not is_lsp_json_message

    # Extract tags only from text messages
    tags, clean_msg = extract_tags(msg) if is_normal_text_message else (LspMessageTags(), msg)

    lsp_enabled = is_LSP_enabled()
    unsupported_level = level not in supported_lsp_log_levels

    # ---- Normal logging path ----
    if not tags.lsp:
        if not lsp_enabled:  # LSP disabled
            actual_log_fn(clean_msg, *args, **kwargs)
            return
        if tags.not_lsp:  # explicitly marked as not for LSP
            actual_log_fn(clean_msg, *args, **kwargs)
            return
        if unsupported_level and not tags.force_lsp:  # unsupported level
            actual_log_fn(clean_msg, *args, **kwargs)
            return

    if not lsp_enabled:
        # it's for LSP and LSP is disabled
        return

    # ---- LSP logging path ----
    if is_normal_text_message:
        clean_msg = add_heading_tags(clean_msg, tags)
        clean_msg = add_highlight_tags(clean_msg, tags)
        clean_msg = LspTextMessage(text=clean_msg, takes_time=tags.loading, message_id=tags.message_id).serialize()

    actual_log_fn(clean_msg, *args, **kwargs)


# Configure logging to stderr for VS Code output channel
def setup_logging() -> logging.Logger:
    global root_logger  # noqa: PLW0603
    if root_logger:
        return root_logger
    # Clear any existing handlers to prevent conflicts
    logger = logging.getLogger()
    logger.handlers.clear()

    # Set up stderr handler for VS Code output channel
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)

    # Configure root logger
    logger.addHandler(handler)

    # Also configure the pygls logger specifically
    pygls_logger = logging.getLogger("pygls")
    pygls_logger.setLevel(logging.INFO)

    root_logger = logger
    return logger
