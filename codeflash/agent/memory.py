import uuid
from pathlib import Path
from typing import Any

from codeflash.code_utils.code_utils import encoded_tokens_len

json_primitive_types = (str, float, int, bool)


class Memory:
    def __init__(self) -> None:
        self._context_vars: dict[str, str] = {}
        self._messages: list[dict[str, str]] = []
        self.api_calls_counter = 0
        self.trace_id = str(uuid.uuid4())
        self.max_tokens = 16000

    def _serialize(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, list):
            return [self._serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        if isinstance(obj, json_primitive_types) or obj is None:
            return obj
        if isinstance(obj, Path):
            return obj.as_posix()
        return str(obj)

    def add_to_context_vars(self, key: str, value: any) -> dict[str, str]:
        self._context_vars[key] = self._serialize(value)
        return self._context_vars

    def get_context_vars(self) -> dict[str, str]:
        return self._context_vars

    def set_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        self._messages = messages
        if self.get_total_tokens() > self.max_tokens:
            # TODO: summarize messages
            pass
        return self._messages

    def get_messages(self) -> list[dict[str, str]]:
        return self._messages

    def get_total_tokens(self) -> int:
        return sum(encoded_tokens_len(message["content"]) for message in self._messages)
