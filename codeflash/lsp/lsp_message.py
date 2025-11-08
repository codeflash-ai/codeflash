from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from codeflash.lsp.helpers import is_LSP_enabled, replace_quotes_with_backticks, simplify_worktree_paths

json_primitive_types = (str, float, int, bool)
max_code_lines_before_collapse = 45

# \u241F is the message delimiter becuase it can be more than one message sent over the same message, so we need something to separate each message
message_delimiter = "\u241f"


# allow the client to know which message it is receiving
class LSPMessageId(enum.Enum):
    BEST_CANDIDATE = "best_candidate"
    CANDIDATE = "candidate"


@dataclass
class LspMessage:
    # to show a loading indicator if the operation is taking time like generating candidates or tests
    takes_time: bool = False
    message_id: Optional[str] = None

    def _loop_through(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, list):
            return [self._loop_through(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self._loop_through(v) for k, v in obj.items()}
        if isinstance(obj, json_primitive_types) or obj is None:
            return obj
        if isinstance(obj, Path):
            return obj.as_posix()
        return str(obj)

    def type(self) -> str:
        raise NotImplementedError

    def serialize(self) -> str:
        if not is_LSP_enabled():
            return ""
        from codeflash.lsp.context import execution_context_vars

        execution_ctx = execution_context_vars.get()
        current_task_id = execution_ctx.get("task_id", None)
        data = self._loop_through(asdict(self))
        ordered = {"type": self.type(), "task_id": current_task_id, **data}
        return message_delimiter + json.dumps(ordered) + message_delimiter


@dataclass
class LspTextMessage(LspMessage):
    text: str = ""

    def type(self) -> str:
        return "text"

    def serialize(self) -> str:
        self.text = simplify_worktree_paths(self.text)
        self.text = replace_quotes_with_backticks(self.text)
        return super().serialize()


# TODO: use it instead of the lspcodemessage to display multiple files in the same message
class LspMultiCodeMessage(LspMessage):
    files: list[LspCodeMessage]

    def type(self) -> str:
        return "code"

    def serialize(self) -> str:
        return super().serialize()


@dataclass
class LspCodeMessage(LspMessage):
    code: str = ""
    file_name: Optional[Path] = None
    function_name: Optional[str] = None
    collapsed: bool = False
    lines_count: Optional[int] = None

    def type(self) -> str:
        return "code"

    def serialize(self) -> str:
        code_lines_length = len(self.code.split("\n"))
        self.lines_count = code_lines_length
        if code_lines_length > max_code_lines_before_collapse:
            self.collapsed = True
        self.file_name = simplify_worktree_paths(str(self.file_name), highlight=False)
        return super().serialize()


@dataclass
class LspMarkdownMessage(LspMessage):
    markdown: str = ""

    def type(self) -> str:
        return "markdown"

    def serialize(self) -> str:
        self.markdown = simplify_worktree_paths(self.markdown)
        self.markdown = replace_quotes_with_backticks(self.markdown)
        return super().serialize()
