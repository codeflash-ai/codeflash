from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from codeflash.lsp.helpers import replace_quotes_with_backticks, simplify_worktree_paths

json_primitive_types = (str, float, int, bool)
max_code_lines_before_collapse = 45


@dataclass
class LspMessage:
    # to show a loading indicator if the operation is taking time like generating candidates or tests
    takes_time: bool = field(default=False, kw_only=True)

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
        if isinstance(self, LspTextMessage):
            self.text = simplify_worktree_paths(self.text)
            self.text = replace_quotes_with_backticks(self.text)
        if isinstance(self, LspCodeMessage):
            self.file_name = simplify_worktree_paths(str(self.file_name), highlight=False)
        if isinstance(self, LspMarkdownMessage):
            self.markdown = simplify_worktree_paths(self.markdown)
            self.markdown = replace_quotes_with_backticks(self.markdown)

        data = self._loop_through(asdict(self))
        # Important: keep type as the first key, for making it easy and fast for the client to know if this is a lsp message before parsing it
        ordered = {"type": self.type(), **data}
        return json.dumps(ordered)


@dataclass
class LspTextMessage(LspMessage):
    text: str

    def type(self) -> str:
        return "text"


@dataclass
class LspCodeMessage(LspMessage):
    code: str
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
        return super().serialize()


@dataclass
class LspMarkdownMessage(LspMessage):
    markdown: str

    def type(self) -> str:
        return "markdown"


@dataclass
class LspStatsMessage(LspMessage):
    stats: dict[str, Any]

    def type(self) -> str:
        return "stats"
