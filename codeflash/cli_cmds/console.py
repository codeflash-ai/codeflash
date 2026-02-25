from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from itertools import cycle
from typing import TYPE_CHECKING, Optional

from rich.console import Console
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

from codeflash.cli_cmds.console_constants import SPINNER_TYPES
from codeflash.cli_cmds.logging_config import BARE_LOGGING_FORMAT
from codeflash.lsp.helpers import is_LSP_enabled, is_subagent_mode
from codeflash.lsp.lsp_logger import enhanced_log
from codeflash.lsp.lsp_message import LspCodeMessage, LspTextMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from rich.progress import TaskID

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import DependencyResolver, IndexResult
    from codeflash.lsp.lsp_message import LspMessage
    from codeflash.models.models import TestResults

DEBUG_MODE = logging.getLogger().getEffectiveLevel() == logging.DEBUG

console = Console()

if is_LSP_enabled() or is_subagent_mode():
    console.quiet = True

if is_subagent_mode():
    import re
    import sys

    _lsp_prefix_re = re.compile(r"^(?:!?lsp,?|h[2-4]|loading)\|")
    _subagent_drop_patterns = (
        "Test log -",
        "Test failed to load",
        "Examining file ",
        "Generated ",
        "Add custom marker",
        "Disabling all autouse",
        "Reverting code and helpers",
    )

    class _AgentLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.msg = _lsp_prefix_re.sub("", str(record.msg))
            msg = record.getMessage()
            return not any(msg.startswith(p) for p in _subagent_drop_patterns)

    _agent_handler = logging.StreamHandler(sys.stderr)
    _agent_handler.addFilter(_AgentLogFilter())
    logging.basicConfig(level=logging.INFO, handlers=[_agent_handler], format="%(levelname)s: %(message)s")
else:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False)],
        format=BARE_LOGGING_FORMAT,
    )

logger = logging.getLogger("rich")
logging.getLogger("parso").setLevel(logging.WARNING)

# override the logger to reformat the messages for the lsp
if not is_subagent_mode():
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
    if is_subagent_mode():
        return
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
    if is_subagent_mode():
        return
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

    if is_subagent_mode():
        yield DummyTask().id
        return

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
    if is_subagent_mode():
        yield DummyProgress(), DummyTask().id
        return

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
def call_graph_live_display(
    total: int, project_root: Path | None = None
) -> Generator[Callable[[IndexResult], None], None, None]:
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.tree import Tree

    if is_subagent_mode():
        yield lambda _: None
        return

    if is_LSP_enabled():
        lsp_log(LspTextMessage(text="Building call graph", takes_time=True))
        yield lambda _: None
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

    results: deque[IndexResult] = deque(maxlen=MAX_TREE_ENTRIES)
    stats = {"indexed": 0, "cached": 0, "edges": 0, "external": 0, "errors": 0}

    tree = Tree("[bold]Recent Files[/bold]")
    stats_text = Text("0 calls found", style="dim")
    panel = Panel(
        Group(progress, Text(""), tree, Text(""), stats_text), title="Building Call Graph", border_style="cyan"
    )

    def create_tree_node(result: IndexResult) -> Tree:
        if project_root:
            try:
                name = str(result.file_path.resolve().relative_to(project_root.resolve()))
            except ValueError:
                name = f"{result.file_path.parent.name}/{result.file_path.name}"
        else:
            name = f"{result.file_path.parent.name}/{result.file_path.name}"

        if result.error:
            return Tree(f"[red]{name}  (error)[/red]")

        if result.cached:
            return Tree(f"[dim]{name}  (cached)[/dim]")

        local_edges = result.num_edges - result.cross_file_edges
        edge_info = []

        if local_edges:
            edge_info.append(f"{local_edges} calls in same file")
        if result.cross_file_edges:
            edge_info.append(f"{result.cross_file_edges} calls from other modules")

        label = ", ".join(edge_info) if edge_info else "no calls"
        return Tree(f"[cyan]{name}[/cyan]  [dim]{label}[/dim]")

    def refresh_display() -> None:
        tree.children = [create_tree_node(r) for r in results]
        tree.children.extend([Tree(" ")] * (MAX_TREE_ENTRIES - len(results)))

        # Update stats
        stat_parts = []
        if stats["indexed"]:
            stat_parts.append(f"{stats['indexed']} files analyzed")
        if stats["cached"]:
            stat_parts.append(f"{stats['cached']} cached")
        if stats["errors"]:
            stat_parts.append(f"{stats['errors']} errors")
        stat_parts.append(f"{stats['edges']} calls found")
        if stats["external"]:
            stat_parts.append(f"{stats['external']} cross-file calls")

        stats_text.truncate(0)
        stats_text.append(" · ".join(stat_parts), style="dim")

    batch: list[IndexResult] = []

    def process_batch() -> None:
        for result in batch:
            results.append(result)

            if result.error:
                stats["errors"] += 1
            elif result.cached:
                stats["cached"] += 1
            else:
                stats["indexed"] += 1
                stats["edges"] += result.num_edges
                stats["external"] += result.cross_file_edges

            progress.advance(task_id)

        batch.clear()
        refresh_display()
        live.refresh()

    def update(result: IndexResult) -> None:
        batch.append(result)
        if len(batch) >= 8:
            process_batch()

    with Live(panel, console=console, transient=False, auto_refresh=False) as live:
        yield update
        if batch:
            process_batch()


def call_graph_summary(call_graph: DependencyResolver, file_to_funcs: dict[Path, list[FunctionToOptimize]]) -> None:
    total_functions = sum(map(len, file_to_funcs.values()))
    if not total_functions:
        return

    if is_subagent_mode():
        return

    # Build the mapping expected by the dependency resolver
    file_items = file_to_funcs.items()
    mapping = {file_path: {func.qualified_name for func in funcs} for file_path, funcs in file_items}

    callee_counts = call_graph.count_callees_per_function(mapping)

    # Use built-in sum for C-level loops to reduce Python overhead
    total_callees = sum(callee_counts.values())
    with_context = sum(1 for count in callee_counts.values() if count > 0)

    leaf_functions = total_functions - with_context
    avg_callees = total_callees / total_functions

    function_label = "function" if total_functions == 1 else "functions"

    summary = (
        f"{total_functions} {function_label} ready for optimization\n"
        f"Uses other functions: {with_context} · "
        f"Standalone: {leaf_functions}"
    )

    if is_LSP_enabled():
        lsp_log(LspTextMessage(text=summary))
        return

    console.print(Panel(summary, title="Call Graph Summary", border_style="cyan"))


def subagent_log_optimization_result(
    function_name: str,
    file_path: Path,
    perf_improvement_line: str,
    original_runtime_ns: int,
    best_runtime_ns: int,
    raw_explanation: str,
    original_code: dict[Path, str],
    new_code: dict[Path, str],
    review: str,
    test_results: TestResults,
) -> None:
    import sys
    from xml.sax.saxutils import escape

    from codeflash.code_utils.code_utils import unified_diff_strings
    from codeflash.code_utils.time_utils import humanize_runtime
    from codeflash.models.test_type import TestType

    diff_parts = []
    for path in original_code:
        old = original_code.get(path, "")
        new = new_code.get(path, "")
        if old != new:
            diff = unified_diff_strings(old, new, fromfile=str(path), tofile=str(path))
            if diff:
                diff_parts.append(diff)

    diff_str = "\n".join(diff_parts)

    original_runtime = humanize_runtime(original_runtime_ns)
    optimized_runtime = humanize_runtime(best_runtime_ns)

    report = test_results.get_test_pass_fail_report_by_type()
    verification_rows = []
    for test_type in TestType:
        if test_type is TestType.INIT_STATE_TEST:
            continue
        name = test_type.to_name()
        if not name:
            continue
        passed = report[test_type]["passed"]
        failed = report[test_type]["failed"]
        if passed == 0 and failed == 0:
            status = "None Found"
        elif failed > 0:
            status = f"{failed} Failed, {passed} Passed"
        else:
            status = f"{passed} Passed"
        verification_rows.append(f'    <test type="{escape(name)}" status="{escape(status)}"/>')

    xml = [
        "<codeflash-optimization>",
        f"  <function>{escape(function_name)}</function>",
        f"  <file>{escape(str(file_path))}</file>",
        f"  <performance>{escape(perf_improvement_line)}</performance>",
        f"  <original-runtime>{escape(original_runtime)}</original-runtime>",
        f"  <optimized-runtime>{escape(optimized_runtime)}</optimized-runtime>",
    ]
    if review:
        xml.append(f"  <reviewer-confidence>{escape(review)}</reviewer-confidence>")
    xml.append(f"  <explanation>{escape(raw_explanation.strip())}</explanation>")
    xml.append("  <verification>")
    xml.extend(verification_rows)
    xml.append("  </verification>")
    if diff_str:
        xml.append(f"  <diff>{escape(diff_str)}</diff>")
    for path in new_code:
        if new_code[path] != original_code.get(path, ""):
            xml.append(f'  <optimized-code file="{escape(str(path))}">{escape(new_code[path])}</optimized-code>')
    xml.append("  <action>")
    xml.append("    First, present a short summary: function name, performance improvement, and explanation.")
    xml.append("    Then, for each optimized-code element, use the Write tool to write the optimized code to the file.")
    xml.append("    This will surface the diff in the user's IDE (VS Code, Cursor, Windsurf, etc.) for review.")
    xml.append("    The user can accept or reject the changes through the IDE's native diff view.")
    xml.append("  </action>")
    xml.append("</codeflash-optimization>")

    sys.stdout.write("\n".join(xml) + "\n")
