from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from codeflash.cli_cmds.logging_config import BARE_LOGGING_FORMAT

console = Console(record=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RichHandler(rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False),
    ],
    format=BARE_LOGGING_FORMAT,
)

logger = logging.getLogger("rich")


def paneled_text(
    text: str,
    panel_args: dict[str, str | bool] | None = None,
    text_args: dict[str, str] | None = None,
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    panel_args = panel_args or {}
    text_args = text_args or {}

    rich_text_obj = Text(text, **text_args)
    panel = Panel(rich_text_obj, **panel_args)
    console.print(panel)


def code_print(code_str: str) -> None:
    from rich.syntax import Syntax

    console.rule()
    console.print(Syntax(code_str, "python", line_numbers=True, theme="github-dark"))
    console.rule()
