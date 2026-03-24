"""Rich-based console UI for codeflash_core.

Provides spinners, progress bars, panels, code display, and logging.
No LSP or subagent concerns — this is the core CLI output layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import cycle
from typing import TYPE_CHECKING

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Generator

    from rich.progress import TaskID

# ---------------------------------------------------------------------------
# Console and logging
# ---------------------------------------------------------------------------

console = Console(highlighter=NullHighlighter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=False,
            highlighter=NullHighlighter(),
            console=console,
            show_path=False,
            show_time=False,
        )
    ],
    format="%(message)s",
)

logger = logging.getLogger("codeflash")

# ---------------------------------------------------------------------------
# Spinners
# ---------------------------------------------------------------------------

SPINNER_TYPES = [
    "dots",
    "dots2",
    "dots3",
    "dots4",
    "dots5",
    "line",
    "line2",
    "arc",
    "circle",
    "star",
    "star2",
    "moon",
    "bouncingBar",
    "bouncingBall",
    "flip",
    "growVertical",
    "growHorizontal",
    "balloon",
    "noise",
    "bounce",
    "point",
    "layer",
    "betaWave",
]

_spinners = cycle(SPINNER_TYPES)

# ---------------------------------------------------------------------------
# Dummy types for fallback
# ---------------------------------------------------------------------------


class _DummyTask:
    def __init__(self) -> None:
        self.id = 0


class _DummyProgress:
    def advance(self, task_id: TaskID, advance: int = 1) -> None:
        pass


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------

_progress_bar_active = False


@contextmanager
def progress_bar(message: str, *, transient: bool = False) -> Generator[TaskID, None, None]:
    """Spinner with elapsed time. Avoids nesting Rich Live displays."""
    global _progress_bar_active  # noqa: PLW0603

    if _progress_bar_active:
        yield _DummyTask().id
        return

    _progress_bar_active = True
    try:
        progress = Progress(
            SpinnerColumn(next(_spinners)),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=transient,
        )
        task = progress.add_task(message, total=None)
        with progress:
            yield task
    finally:
        _progress_bar_active = False


@contextmanager
def test_files_progress_bar(total: int, description: str) -> Generator[tuple[Progress, TaskID], None, None]:
    """Progress bar with M/N counter for test files."""
    with Progress(
        SpinnerColumn(next(_spinners)),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green", pulse_style="yellow"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def paneled_text(text: str, *, title: str = "", border_style: str = "cyan") -> None:
    """Print text inside a bordered panel."""
    console.print(Panel(Text(text), title=title or None, border_style=border_style))


def code_print(code_str: str, *, language: str = "python") -> None:
    """Print code with syntax highlighting."""
    from rich.syntax import Syntax

    console.rule()
    console.print(Syntax(code_str, language, line_numbers=True, theme="github-dark"))
    console.rule()
