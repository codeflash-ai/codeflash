from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import cycle
from typing import TYPE_CHECKING, Generator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from codeflash.cli_cmds.console_constants import SPINNER_TYPES
from codeflash.cli_cmds.logging_config import BARE_LOGGING_FORMAT

if TYPE_CHECKING:
    from rich.progress import TaskID

console = Console(record=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False)],
    format=BARE_LOGGING_FORMAT,
)

logger = logging.getLogger("rich")


def paneled_text(
    text: str, panel_args: dict[str, str | bool] | None = None, text_args: dict[str, str] | None = None
) -> None:
    """Print text in a panel."""
    from rich.panel import Panel
    from rich.text import Text

    panel_args = panel_args or {}
    text_args = text_args or {}

    rich_text_obj = Text(text, **text_args)
    panel = Panel(rich_text_obj, **panel_args)
    console.print(panel)


def code_print(code_str: str) -> None:
    """Print code with syntax highlighting."""
    from rich.syntax import Syntax

    console.rule()
    console.print(Syntax(code_str, "python", line_numbers=True, theme="github-dark"))
    console.rule()


spinners = cycle(SPINNER_TYPES)


@contextmanager
def progress_bar(message: str, *, transient: bool = False) -> Generator[TaskID, None, None]:
    """Display a progress bar with a spinner and elapsed time."""
    progress = Progress(
        SpinnerColumn(next(spinners)),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )
    task = progress.add_task(message, total=None)
    with progress:
        yield task
