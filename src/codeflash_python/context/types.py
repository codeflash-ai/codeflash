from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from codeflash_python.models.models import FunctionSource

from codeflash_core.models import HelperFunction


class CodeContextType(str, Enum):
    READ_WRITABLE = "READ_WRITABLE"
    READ_ONLY = "READ_ONLY"
    TESTGEN = "TESTGEN"
    HASHING = "HASHING"


@dataclass(frozen=True)
class IndexResult:
    file_path: Path
    cached: bool
    num_edges: int
    edges: tuple[tuple[str, str, bool], ...]  # (caller_qn, callee_name, is_cross_file)
    cross_file_edges: int
    error: bool


@dataclass
class PythonCodeContext:
    """Code context extracted for optimization.

    Contains the target function code and all relevant dependencies
    needed for the AI to understand and optimize the function.

    Attributes:
        target_code: Source code of the function to optimize.
        target_file: Path to the file containing the target function.
        helper_functions: List of helper functions called by the target.
        read_only_context: Additional context code (read-only dependencies).
        imports: List of import statements needed.
        language: The programming language.

    """

    target_code: str
    target_file: Path
    helper_functions: list[HelperFunction] = field(default_factory=list)
    read_only_context: str = ""
    imported_type_skeletons: str = ""
    imports: list[str] = field(default_factory=list)
    language: str = "python"


@dataclass
class ReferenceInfo:
    """Information about a reference (call site) to a function.

    This class captures information about where a function is called
    from, including the file, line number, context, and caller function.

    Attributes:
        file_path: Path to the file containing the reference.
        line: Line number (1-indexed).
        column: Column number (0-indexed).
        end_line: End line number (1-indexed).
        end_column: End column number (0-indexed).
        context: The line of code containing the reference.
        reference_type: Type of reference ("call", "callback", "memoized", "import", "reexport").
        import_name: Name used to import the function (may differ from original).
        caller_function: Name of the function containing this reference (or None for module-level).

    """

    file_path: Path
    line: int
    column: int
    end_line: int
    end_column: int
    context: str
    reference_type: str
    import_name: str | None
    caller_function: str | None = None


def function_sources_to_helpers(sources: list[FunctionSource]) -> list[HelperFunction]:
    """Convert FunctionSource objects to HelperFunction objects."""
    return [
        HelperFunction(
            name=fs.only_function_name,
            qualified_name=fs.qualified_name,
            file_path=fs.file_path,
            source_code=fs.source_code,
            start_line=1,  # TODO: FunctionSource should carry real line numbers from jedi definitions
            end_line=fs.source_code.count("\n") + 1,
        )
        for fs in sources
    ]


@runtime_checkable
class DependencyResolver(Protocol):
    """Protocol for language-specific dependency resolution.

    Implementations analyze source files to discover call-graph edges
    between functions so the optimizer can extract richer context.
    """

    def build_index(self, file_paths: Iterable[Path], on_progress: Callable[[IndexResult], None] | None = None) -> None:
        """Pre-index a batch of files."""
        ...

    def get_callees(
        self, file_path_to_qualified_names: dict[Path, set[str]]
    ) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:
        """Return callees for the given functions."""
        ...

    def count_callees_per_function(
        self, file_path_to_qualified_names: dict[Path, set[str]]
    ) -> dict[tuple[Path, str], int]:
        """Return the number of callees for each (file_path, qualified_name) pair."""
        ...

    def close(self) -> None:
        """Release resources (e.g. database connections)."""
        ...
