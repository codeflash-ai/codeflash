from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

json_primitive_types = (str, float, int, bool)


@dataclass
class LspMessage:
    takes_time: bool = field(
        default=False, kw_only=True
    )  # to show a loading indicator if the operation is taking time like generating candidates or tests

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
        data = asdict(self)
        data["type"] = self.type()
        return json.dumps(data)


@dataclass
class LspTextMessage(LspMessage):
    text: str

    def type(self) -> str:
        return "text"


@dataclass
class LspCodeMessage(LspMessage):
    code: str
    path: Optional[Path] = None
    function_name: Optional[str] = None

    def type(self) -> str:
        return "code"


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
