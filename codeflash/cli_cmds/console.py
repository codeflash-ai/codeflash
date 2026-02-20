from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import cycle
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from codeflash.cli_cmds.console_constants import SPINNER_TYPES
from codeflash.cli_cmds.logging_config import BARE_LOGGING_FORMAT
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.lsp.lsp_logger import enhanced_log
from codeflash.lsp.lsp_message import LspCodeMessage, LspTextMessage

if TYPE_CHECKING:
    from collections.abc import Generator

    from rich.progress import TaskID

    from codeflash.lsp.lsp_message import LspMessage

DEBUG_MODE = logging.getLogger().getEffectiveLevel() == logging.DEBUG

console = Console(highlighter=NullHighlighter())

if is_LSP_enabled():
    console.quiet = True

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
    format=BARE_LOGGING_FORMAT,
)

logger = logging.getLogger("rich")
logging.getLogger("parso").setLevel(logging.WARNING)

# override the logger to reformat the messages for the lsp
for level in ("info", "debug", "warning", "error"):
    real_fn = getattr(logger, level)
    setattr(
        logger,
        level,
        lambda msg, *args, _real_fn=real_fn, _level=level, **kwargs: enhanced_log(
            msg, _real_fn, _level, *args, **kwargs
        ),
    )


class DummyTask:
    def __init__(self) -> None:
        self.id = 0


class DummyProgress:
    def __init__(self) -> None:
        pass

    def advance(self, task_id: TaskID, advance: int = 1) -> None:
        pass


def lsp_log(message: LspMessage) -> None:
    if not is_LSP_enabled():
        return
    json_msg = message.serialize()
    logger.info(json_msg)


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


def code_print(
    code_str: str,
    file_name: Optional[str] = None,
    function_name: Optional[str] = None,
    lsp_message_id: Optional[str] = None,
    language: str = "python",
) -> None:
    """Print code with syntax highlighting.

    Args:
        code_str: The code to print
        file_name: Optional file name for LSP
        function_name: Optional function name for LSP
        lsp_message_id: Optional LSP message ID
        language: Programming language for syntax highlighting ('python', 'javascript', 'typescript')

    """
    if is_LSP_enabled():
        lsp_log(
            LspCodeMessage(code=code_str, file_name=file_name, function_name=function_name, message_id=lsp_message_id)
        )
        return

    from rich.syntax import Syntax

    # Map codeflash language names to rich/pygments lexer names
    lexer_map = {"python": "python", "javascript": "javascript", "typescript": "typescript"}
    lexer = lexer_map.get(language, "python")

    console.rule()
    console.print(Syntax(code_str, lexer, line_numbers=True, theme="github-dark"))
    console.rule()


spinners = cycle(SPINNER_TYPES)

# Track whether a progress bar is already active to prevent nested Live displays
_progress_bar_active = False


@contextmanager
def progress_bar(
    message: str, *, transient: bool = False, revert_to_print: bool = False
) -> Generator[TaskID, None, None]:
    """Display a progress bar with a spinner and elapsed time.

    If revert_to_print is True, falls back to printing a single logger.info message
    instead of showing a progress bar.

    If a progress bar is already active, yields a dummy task ID to avoid Rich's
    LiveError from nested Live displays.
    """
    global _progress_bar_active

    if is_LSP_enabled():
        lsp_log(LspTextMessage(text=message, takes_time=True))
        yield
        return

    if revert_to_print or _progress_bar_active:
        if revert_to_print:
            logger.info(message)

        # Create a fake task ID since we still need to yield something
        yield DummyTask().id
    else:
        _progress_bar_active = True
        try:
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
        finally:
            _progress_bar_active = False


@contextmanager
def test_files_progress_bar(total: int, description: str) -> Generator[tuple[Progress, TaskID], None, None]:
    """Progress bar for test files."""
    if is_LSP_enabled():
        lsp_log(LspTextMessage(text=description, takes_time=True))
        dummy_progress = DummyProgress()
        dummy_task = DummyTask()
        yield dummy_progress, dummy_task.id
        return

    with Progress(
        SpinnerColumn(next(spinners)),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green", pulse_style="yellow"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield progress, task_id
