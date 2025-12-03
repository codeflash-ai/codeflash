"""Verification module for codeflash.

This module provides test running and verification functionality.
"""


def __getattr__(name: str):  # noqa: ANN202
    """Lazy import for LLM tools to avoid circular imports."""
    if name in (
        "AVAILABLE_TOOLS",
        "RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA",
        "execute_tool",
        "get_all_tool_schemas",
        "get_tool_schema",
        "run_behavioral_tests_tool",
    ):
        from codeflash.verification import llm_tools

        return getattr(llm_tools, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AVAILABLE_TOOLS",
    "RUN_BEHAVIORAL_TESTS_TOOL_SCHEMA",
    "execute_tool",
    "get_all_tool_schemas",
    "get_tool_schema",
    "run_behavioral_tests_tool",
]
