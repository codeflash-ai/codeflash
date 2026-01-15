from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Tool(str, Enum):
    REPLACE_IN_FILE = "replace_in_file"
    ADD_TO_CONTEXT_VARS = "add_to_context_vars"
    EXECUTE_CODE = "execute_code"  # currently only supports python
    SEARCH_FUNCTION_REFRENCES = "search_function_references"
    GET_NAME_DEFINITION = "get_name_definition"
    TERMINATE = "terminate"  # terminates either with success or failure


# TODO: use this as a type for the api response
@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    args: dict[str, Any]
    needs_context_vars: bool = False


supported_tools: list[str] = [
    Tool.REPLACE_IN_FILE,
    Tool.ADD_TO_CONTEXT_VARS,
    Tool.EXECUTE_CODE,
    # Tool.SEARCH_FUNCTION_REFRENCES,
    # Tool.GET_NAME_DEFINITION,
    # Tool.TERMINATE,
]
