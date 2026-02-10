from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import cycle
from typing import TYPE_CHECKING, Optional

from rich.console import Console
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
    from collections.abc import Callable, Generator
    from pathlib import Path

    from rich.progress import TaskID

    from codeflash.context.call_graph import CallGraph, IndexResult
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.lsp.lsp_message import LspMessage

DEBUG_MODE = logging.getLogger().getEffectiveLevel() == logging.DEBUG

console = Console()

if is_LSP_enabled():
    console.quiet = True

logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False)],
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


MAX_TREE_ENTRIES = 8


@contextmanager
def call_graph_live_display(total: int) -> Generator[Callable[[IndexResult], None], None, None]:
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree

    if is_LSP_enabled():
        lsp_log(LspTextMessage(text="Building call graph", takes_time=True))
        yield lambda _result: None
        return

    progress = Progress(
        SpinnerColumn(next(spinners)),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green", pulse_style="yellow"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False,
    )
    task_id = progress.add_task("Analyzing files", total=total)

    results: list[IndexResult] = []
    stats_indexed = 0
    stats_cached = 0
    stats_edges = 0
    stats_external = 0
    stats_errors = 0

    def make_display() -> Panel:
        tree = Tree("[bold]Dependencies[/bold]")
        for result in results[-MAX_TREE_ENTRIES:]:
            name = result.file_path.name
            if result.error:
                tree.add(f"[red]{name}  (error)[/red]")
            elif result.cached:
                tree.add(f"[dim]{name}  (cached)[/dim]")
            else:
                local = result.num_edges - result.cross_file_edges
                parts = []
                if local:
                    parts.append(f"{local} in same file")
                if result.cross_file_edges:
                    parts.append(f"{result.cross_file_edges} from other modules")
                label = ", ".join(parts) if parts else "no dependencies"
                tree.add(f"[cyan]{name}[/cyan]  [dim]{label}[/dim]")

        parts: list[str] = []
        if stats_indexed:
            parts.append(f"{stats_indexed} files analyzed")
        if stats_cached:
            parts.append(f"{stats_cached} cached")
        if stats_errors:
            parts.append(f"{stats_errors} errors")
        parts.append(f"{stats_edges} dependencies found")
        if stats_external:
            parts.append(f"{stats_external} from other modules")
        stats_text = Text(" · ".join(parts), style="dim")

        return Panel(
            Group(progress, Text(""), tree, Text(""), stats_text),
            title="Building Call Graph",
            border_style="cyan",
        )

    def update(result: IndexResult) -> None:
        nonlocal stats_indexed, stats_cached, stats_edges, stats_external, stats_errors
        results.append(result)
        if result.error:
            stats_errors += 1
        elif result.cached:
            stats_cached += 1
        else:
            stats_indexed += 1
            stats_edges += result.num_edges
            stats_external += result.cross_file_edges
        progress.advance(task_id)
        live.update(make_display())

    with Live(make_display(), console=console, transient=True, refresh_per_second=8) as live:
        yield update


def call_graph_summary(call_graph: CallGraph, file_to_funcs: dict[Path, list[FunctionToOptimize]]) -> None:
    from rich.panel import Panel

    total_functions = sum(len(funcs) for funcs in file_to_funcs.values())
    if total_functions == 0:
        return

    total_callees = 0
    with_context = 0
    leaf_functions = 0

    for file_path, funcs in file_to_funcs.items():
        for func in funcs:
            _, func_callees = call_graph.get_callees({file_path: {func.qualified_name}})
            count = len(func_callees)
            total_callees += count
            if count > 0:
                with_context += 1
            else:
                leaf_functions += 1

    avg_callees = total_callees / total_functions if total_functions > 0 else 0

    summary = (
        f"{total_functions} functions ready for optimization · "
        f"avg {avg_callees:.1f} dependencies/function\n"
        f"{with_context} call other functions · "
        f"{leaf_functions} are self-contained"
    )

    if is_LSP_enabled():
        lsp_log(LspTextMessage(text=summary))
        return

    console.print(Panel(summary, title="Dependency Summary", border_style="cyan"))
