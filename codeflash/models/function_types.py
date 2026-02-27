"""Simple function-related types with no dependencies.

This module contains basic types used for function representation.
It is intentionally kept dependency-free to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class FunctionParent:
    name: str
    type: str

    def __str__(self) -> str:
        return f"{self.type}:{self.name}"


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class FunctionToOptimize:
    """Represent a function that is a candidate for optimization.

    This is the canonical dataclass for representing functions across all languages
    (Python, JavaScript, TypeScript). It captures all information needed to identify,
    locate, and work with a function.

    Attributes
    ----------
        function_name: The name of the function.
        file_path: The absolute file path where the function is located.
        parents: A list of parent scopes, which could be classes or functions.
        starting_line: The starting line number of the function in the file (1-indexed).
        ending_line: The ending line number of the function in the file (1-indexed).
        starting_col: The starting column offset (0-indexed, for precise location).
        ending_col: The ending column offset (0-indexed, for precise location).
        is_async: Whether this function is defined as async.
        is_method: Whether this is a method (belongs to a class).
        language: The programming language of this function (default: "python").
        doc_start_line: Line where docstring/JSDoc starts (or None if no doc comment).

    The qualified_name property provides the full name of the function, including
    any parent class or function names. The qualified_name_with_modules_from_root
    method extends this with the module name from the project root.

    """

    function_name: str
    file_path: Path
    parents: list[FunctionParent] = Field(default_factory=list)
    starting_line: Optional[int] = None
    ending_line: Optional[int] = None
    starting_col: Optional[int] = None
    ending_col: Optional[int] = None
    is_async: bool = False
    is_method: bool = False
    language: str = "python"
    doc_start_line: Optional[int] = None
    metadata: Optional[dict[str, Any]] = Field(default=None)

    @property
    def top_level_parent_name(self) -> str:
        return self.function_name if not self.parents else self.parents[0].name

    @property
    def class_name(self) -> str | None:
        """Get the immediate parent class name, if any."""
        for parent in reversed(self.parents):
            if parent.type == "ClassDef":
                return parent.name
        return None

    def __str__(self) -> str:
        qualified = f"{'.'.join([p.name for p in self.parents])}{'.' if self.parents else ''}{self.function_name}"
        line_info = f":{self.starting_line}-{self.ending_line}" if self.starting_line and self.ending_line else ""
        return f"{self.file_path}:{qualified}{line_info}"

    @property
    def qualified_name(self) -> str:
        if not self.parents:
            return self.function_name
        parent_path = ".".join(parent.name for parent in self.parents)
        return f"{parent_path}.{self.function_name}"

    def qualified_name_with_modules_from_root(self, project_root_path: Path) -> str:
        # Import here to avoid circular imports
        from codeflash.code_utils.code_utils import module_name_from_file_path

        return f"{module_name_from_file_path(self.file_path, project_root_path)}.{self.qualified_name}"
