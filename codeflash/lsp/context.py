from __future__ import annotations

import contextvars

# Shared execution context for tracking task IDs and other metadata
execution_context_vars: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "execution_context_vars",
    default={},  # noqa: B039
)
