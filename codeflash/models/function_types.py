"""Simple function-related types with no dependencies.

This module contains basic types used for function representation.
It is intentionally kept dependency-free to avoid circular imports.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.type}:{self.name}"
