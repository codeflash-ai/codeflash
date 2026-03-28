"""Cross-branch benchmark comparison.

Compares benchmark performance between two git refs by:
1. Auto-detecting changed functions (or using an explicit list)
2. Creating worktrees for each ref
3. Instrumenting functions with @codeflash_trace
4. Running benchmarks via trace_benchmarks_pytest
5. Rendering a side-by-side Rich comparison table
"""

from __future__ import annotations

import ast
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git
from rich.table import Table

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeflash.models.function_types import FunctionToOptimize
    from codeflash.models.models import BenchmarkKey


@dataclass
class CompareResult:
    base_ref: str
    head_ref: str
    base_total_ns: dict[BenchmarkKey, int] = field(default_factory=dict)
    head_total_ns: dict[BenchmarkKey, int] = field(default_factory=dict)
    base_function_ns: dict[str, dict[BenchmarkKey, int]] = field(default_factory=dict)
    head_function_ns: dict[str, dict[BenchmarkKey, int]] = field(default_factory=dict)

    def format_markdown(self) -> str:
        """Format comparison results as GitHub-flavored markdown (for programmatic use, e.g. PR comments)."""
        if not self.base_total_ns and not self.head_total_ns:
            return "_No benchmark results to compare._"

        base_short = self.base_ref[:12]
        head_short = self.head_ref[:12]
        all_keys = sorted(set(self.base_total_ns) | set(self.head_total_ns), key=str)
        sections: list[str] = [f"## Benchmark: `{base_short}` vs `{head_short}`"]

        for bm_key in all_keys:
            base_ns = self.base_total_ns.get(bm_key)
            head_ns = self.head_total_ns.get(bm_key)

            # Extract short benchmark name from the full key
            bm_name = str(bm_key).rsplit("::", 1)[-1] if "::" in str(bm_key) else str(bm_key)

            # --- End-to-End table ---
            lines = [
                f"### {bm_name}",
                "",
                "| Branch | Time (ms) | vs base | Speedup |",
                "|:---|---:|---:|---:|",
                f"| `{base_short}` (base) | {_fmt_ms(base_ns)} | - | - |",
                f"| `{head_short}` (head) | {_fmt_ms(head_ns)} "
                f"| {_md_delta(base_ns, head_ns)} | {_md_speedup(base_ns, head_ns)} |",
            ]

            # --- Per-function breakdown ---
            all_funcs: set[str] = set()
            for d in [self.base_function_ns, self.head_function_ns]:
                for func_name, bm_dict in d.items():
                    if bm_key in bm_dict:
                        all_funcs.add(func_name)

            if all_funcs:

                def sort_key(fn: str, _bm_key: BenchmarkKey = bm_key) -> int:
                    return self.base_function_ns.get(fn, {}).get(_bm_key, 0)

                sorted_funcs = sorted(all_funcs, key=sort_key, reverse=True)

                lines.append("")
                lines.append("| Function | base (ms) | head (ms) | Improvement | Speedup |")
                lines.append("|:---|---:|---:|:---|---:|")

                for func_name in sorted_funcs:
                    b = self.base_function_ns.get(func_name, {}).get(bm_key)
                    h = self.head_function_ns.get(func_name, {}).get(bm_key)
                    short_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
                    lines.append(
                        f"| `{short_name}` | {_fmt_ms(b)} | {_fmt_ms(h)} | {_md_bar(b, h)} | {_md_speedup(b, h)} |"
                    )

                lines.append(
                    f"| **TOTAL** | **{_fmt_ms(base_ns)}** | **{_fmt_ms(head_ns)}** "
                    f"| {_md_bar(base_ns, head_ns)} | {_md_speedup(base_ns, head_ns)} |"
                )

                # --- Share of Benchmark Time (%) ---
                if base_ns and head_ns:
                    lines.append("")
                    lines.append("<details><summary>Share of Benchmark Time</summary>")
                    lines.append("")
                    lines.append("| Function | base | head |")
                    lines.append("|:---|:---|:---|")

                    for func_name in sorted_funcs:
                        b = self.base_function_ns.get(func_name, {}).get(bm_key)
                        h = self.head_function_ns.get(func_name, {}).get(bm_key)
                        short_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
                        b_pct = b / base_ns * 100 if b else 0
                        h_pct = h / head_ns * 100 if h else 0
                        lines.append(f"| `{short_name}` | {_pct_bar(b_pct)} | {_pct_bar(h_pct)} |")

                    lines.append("")
                    lines.append("</details>")

            sections.append("\n".join(lines))

        sections.append("---\n*Generated by codeflash optimization agent*")
        return "\n\n".join(sections)


def compare_branches(
    base_ref: str,
    head_ref: str,
    project_root: Path,
    benchmarks_root: Path,
    tests_root: Path,
    functions: Optional[dict[Path, list[FunctionToOptimize]]] = None,
    timeout: int = 600,
) -> CompareResult:
    """Compare benchmark performance between two git refs.

    If functions is None, auto-detects changed functions from git diff.
    Returns a CompareResult with timing data from both refs.
    """
    from codeflash.benchmarking.instrument_codeflash_trace import instrument_codeflash_trace_decorator
    from codeflash.benchmarking.plugin.plugin import CodeFlashBenchmarkPlugin
    from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest

    repo = git.Repo(project_root, search_parent_directories=True)
    repo_root = Path(repo.working_dir)

    # Auto-detect functions if not provided
    if functions is None:
        functions = _discover_changed_functions(base_ref, head_ref, repo_root)
        if not functions:
            logger.warning("No changed Python functions found between %s and %s", base_ref, head_ref)
            return CompareResult(base_ref=base_ref, head_ref=head_ref)

    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    base_short = base_ref[:12]
    head_short = head_ref[:12]

    func_count = sum(len(fns) for fns in functions.values())
    file_count = len(functions)

    # Build function tree for the panel
    from os.path import commonpath

    from rich.tree import Tree

    rel_paths = []
    for fp in functions:
        rel_paths.append(fp.relative_to(repo_root) if fp.is_relative_to(repo_root) else fp)

    # Strip common prefix so paths are short but unambiguous
    if len(rel_paths) > 1:
        common = Path(commonpath(rel_paths))
        short_paths = [p.relative_to(common) if p != common else Path(p.name) for p in rel_paths]
    else:
        short_paths = [Path(p.name) for p in rel_paths]

    fn_tree = Tree(f"[bold]{func_count} functions[/bold] [dim]across {file_count} files[/dim]", guide_style="dim")
    for (_fp, fns), short in zip(functions.items(), short_paths):
        branch = fn_tree.add(f"[cyan]{short}[/cyan]")
        for fn in fns:
            branch.add(f"[bold]{fn.function_name}[/bold]")

    # Set up worktree paths and trace DB paths
    from codeflash.code_utils.git_worktree_utils import worktree_dirs

    worktree_dirs.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    base_worktree = worktree_dirs / f"compare-base-{timestamp}"
    head_worktree = worktree_dirs / f"compare-head-{timestamp}"
    base_trace_db = worktree_dirs / f"trace-base-{timestamp}.db"
    head_trace_db = worktree_dirs / f"trace-head-{timestamp}.db"

    result = CompareResult(base_ref=base_ref, head_ref=head_ref)

    from rich.console import Group

    step_labels = ["Creating worktrees", f"Benchmarking base ({base_short})", f"Benchmarking head ({head_short})"]

    def build_steps(current_step: int) -> Group:
        lines: list[Text] = []
        for i, label in enumerate(step_labels):
            if i < current_step:
                lines.append(Text.from_markup(f"[green]\u2714[/green] {label}"))
            elif i == current_step:
                lines.append(Text.from_markup(f"[cyan]\u25cb[/cyan] {label}..."))
            else:
                lines.append(Text.from_markup(f"[dim]\u2500 {label}[/dim]"))
        return Group(*lines)

    def build_panel(current_step: int) -> Panel:
        # Two-column grid: tree left, steps right (vertically padded to center)
        tree_height = 1 + sum(1 + len(fns) for fns in functions.values())  # root + files + functions
        step_count = len(step_labels)
        pad_top = max(0, (tree_height - step_count) // 2)

        grid = Table(box=None, show_header=False, expand=True, padding=0)
        grid.add_column(ratio=3)
        grid.add_column(ratio=2)
        grid.add_row(fn_tree, Group(*([Text("")] * pad_top), build_steps(current_step)))

        return Panel(
            Group(
                Text.from_markup(
                    f"[bold cyan]{base_short}[/bold cyan] (base) vs [bold cyan]{head_short}[/bold cyan] (head)"
                ),
                "",
                grid,
            ),
            title="[bold]Benchmark Compare[/bold]",
            border_style="cyan",
            expand=True,
            padding=(1, 2),
        )

    try:
        with Live(build_panel(0), console=console, refresh_per_second=1) as live:
            # Step 1: Create worktrees (resolve to SHAs to avoid "already checked out" errors)
            base_sha = repo.commit(base_ref).hexsha
            head_sha = repo.commit(head_ref).hexsha
            repo.git.worktree("add", str(base_worktree), base_sha)
            repo.git.worktree("add", str(head_worktree), head_sha)
            live.update(build_panel(1))

            # Step 2: Run benchmarks on base
            _run_benchmark_on_worktree(
                worktree_dir=base_worktree,
                repo_root=repo_root,
                functions=functions,
                benchmarks_root=benchmarks_root,
                tests_root=tests_root,
                trace_db=base_trace_db,
                timeout=timeout,
                instrument_fn=instrument_codeflash_trace_decorator,
                trace_fn=trace_benchmarks_pytest,
            )
            live.update(build_panel(2))

            # Step 3: Run benchmarks on head
            _run_benchmark_on_worktree(
                worktree_dir=head_worktree,
                repo_root=repo_root,
                functions=functions,
                benchmarks_root=benchmarks_root,
                tests_root=tests_root,
                trace_db=head_trace_db,
                timeout=timeout,
                instrument_fn=instrument_codeflash_trace_decorator,
                trace_fn=trace_benchmarks_pytest,
            )

        # Load results
        if base_trace_db.exists():
            result.base_total_ns = CodeFlashBenchmarkPlugin.get_benchmark_timings(base_trace_db)
            result.base_function_ns = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(base_trace_db)

        if head_trace_db.exists():
            result.head_total_ns = CodeFlashBenchmarkPlugin.get_benchmark_timings(head_trace_db)
            result.head_function_ns = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(head_trace_db)

        # Render comparison
        _render_comparison(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")

    finally:
        # Cleanup worktrees
        from codeflash.code_utils.git_worktree_utils import remove_worktree

        remove_worktree(base_worktree)
        remove_worktree(head_worktree)
        repo.git.worktree("prune")
        # Cleanup trace DBs
        for db in [base_trace_db, head_trace_db]:
            if db.exists():
                db.unlink()

    return result


def _discover_changed_functions(base_ref: str, head_ref: str, repo_root: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Find only functions whose bodies overlap with changed lines between refs."""
    from io import StringIO

    from unidiff import PatchSet

    repo = git.Repo(repo_root, search_parent_directories=True)

    # Get the diff with line-level detail
    try:
        uni_diff_text = repo.git.diff(f"{base_ref}...{head_ref}", ignore_blank_lines=True, ignore_space_at_eol=True)
    except git.GitCommandError:
        uni_diff_text = repo.git.diff(base_ref, head_ref, ignore_blank_lines=True, ignore_space_at_eol=True)

    if not uni_diff_text.strip():
        return {}

    patch_set = PatchSet(StringIO(uni_diff_text))

    # Build map: file_path -> set of changed line numbers (in the target/head version)
    changed_lines_by_file: dict[Path, set[int]] = {}
    for patched_file in patch_set:
        file_path = Path(patched_file.path)
        if file_path.suffix != ".py":
            continue
        abs_path = repo_root / file_path

        added_lines: set[int] = {
            line.target_line_no
            for hunk in patched_file
            for line in hunk
            if line.is_added and line.value.strip() and line.target_line_no is not None
        }
        deleted_lines: set[int] = {hunk.target_start for hunk in patched_file}
        # Use added lines if available, otherwise use hunk starts (deletion-only changes)
        line_nos: set[int] = added_lines if added_lines else deleted_lines
        if line_nos:
            changed_lines_by_file[abs_path] = line_nos

    # Discover top-level functions in changed files using ast (lightweight, no libcst overhead)
    result: dict[Path, list[FunctionToOptimize]] = {}
    for abs_path, changed_lines in changed_lines_by_file.items():
        if not abs_path.exists():
            logger.debug(f"Skipping {abs_path} (does not exist)")
            continue

        modified_fns = _find_changed_toplevel_functions(abs_path, changed_lines)
        if modified_fns:
            result[abs_path] = modified_fns

    return result


def _find_changed_toplevel_functions(file_path: Path, changed_lines: set[int]) -> list[FunctionToOptimize]:
    """Find top-level functions overlapping changed lines using stdlib ast.

    Only discovers module-level functions (not methods inside classes, not nested
    functions). This is intentional: class methods can be called thousands of times
    in benchmarks (e.g. CST visitor methods), and @codeflash_trace pickles self on
    every call -- catastrophic overhead when self holds a full CST tree.
    """
    from codeflash.models.function_types import FunctionToOptimize

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        logger.debug(f"Skipping {file_path} (parse error)")
        return []

    functions: list[FunctionToOptimize] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.end_lineno is None:
            continue
        fn_lines = range(node.lineno, node.end_lineno + 1)
        if not changed_lines.isdisjoint(fn_lines):
            functions.append(
                FunctionToOptimize(
                    function_name=node.name,
                    file_path=file_path,
                    parents=[],
                    starting_line=node.lineno,
                    ending_line=node.end_lineno,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    is_method=False,
                )
            )

    return functions


def _run_benchmark_on_worktree(
    worktree_dir: Path,
    repo_root: Path,
    functions: dict[Path, list[FunctionToOptimize]],
    benchmarks_root: Path,
    tests_root: Path,
    trace_db: Path,
    timeout: int,
    instrument_fn: Callable[[dict[Path, list[FunctionToOptimize]]], None],
    trace_fn: Callable[[Path, Path, Path, Path, int], None],
) -> None:
    """Instrument, benchmark, and restore source in a worktree."""
    from codeflash.models.function_types import FunctionToOptimize

    # Remap function paths from repo_root to worktree_dir
    worktree_functions: dict[Path, list[FunctionToOptimize]] = {}
    for file_path, fns in functions.items():
        rel = file_path.relative_to(repo_root) if file_path.is_relative_to(repo_root) else file_path
        wt_path = worktree_dir / rel
        if not wt_path.exists():
            logger.debug(f"Skipping {rel} (not present in this ref)")
            continue

        remapped_fns = []
        for fn in fns:
            remapped_fns.append(
                FunctionToOptimize(
                    function_name=fn.function_name,
                    file_path=wt_path,
                    parents=fn.parents,
                    is_method=fn.is_method,
                    is_async=fn.is_async,
                )
            )
        worktree_functions[wt_path] = remapped_fns

    if not worktree_functions:
        logger.warning("No instrumentable functions found in this worktree")
        return

    # Save original source
    original_sources: dict[Path, str] = {}
    for file_path in worktree_functions:
        original_sources[file_path] = file_path.read_text(encoding="utf-8")

    # Remap benchmark and test roots to worktree
    wt_benchmarks = worktree_dir / benchmarks_root.relative_to(repo_root)
    wt_tests = worktree_dir / tests_root.relative_to(repo_root)

    if trace_db.exists():
        trace_db.unlink()

    try:
        instrument_fn(worktree_functions)
        try:
            trace_fn(wt_benchmarks, wt_tests, worktree_dir, trace_db, timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Benchmark timed out after {timeout}s — partial results may be available")
    finally:
        # Restore original source
        for file_path, source in original_sources.items():
            file_path.write_text(source, encoding="utf-8")


def _render_comparison(result: CompareResult) -> None:
    """Render Rich comparison tables to console."""
    if not result.base_total_ns and not result.head_total_ns:
        logger.warning("No benchmark results to compare")
        return

    base_short = result.base_ref[:12]
    head_short = result.head_ref[:12]

    # Find all benchmark keys across both refs
    all_benchmark_keys = set(result.base_total_ns.keys()) | set(result.head_total_ns.keys())

    for bm_key in sorted(all_benchmark_keys, key=str):
        # Show only the test function name, not the full module path
        bm_name = str(bm_key).rsplit("::", 1)[-1] if "::" in str(bm_key) else str(bm_key)
        console.print()
        console.rule(f"[bold]{bm_name}[/bold]")
        console.print()

        base_ns = result.base_total_ns.get(bm_key)
        head_ns = result.head_total_ns.get(bm_key)

        # Table 1: Total benchmark time
        t1 = Table(title="End-to-End", border_style="blue", show_lines=True, expand=False)
        t1.add_column("Ref", style="bold cyan")
        t1.add_column("Time (ms)", justify="right")
        t1.add_column("Delta", justify="right")
        t1.add_column("Speedup", justify="right")

        t1.add_row(f"{base_short} (base)", _fmt_ms(base_ns), "-", "-")
        t1.add_row(
            f"{head_short} (head)", _fmt_ms(head_ns), _fmt_delta(base_ns, head_ns), _fmt_speedup(base_ns, head_ns)
        )
        console.print(t1, justify="center")

        # Table 2: Per-function breakdown
        all_funcs = set()
        for d in [result.base_function_ns, result.head_function_ns]:
            for func_name, bm_dict in d.items():
                if bm_key in bm_dict:
                    all_funcs.add(func_name)

        if all_funcs:
            console.print()

            t2 = Table(title="Per-Function Breakdown", border_style="blue", show_lines=True, expand=False)
            t2.add_column("Function", style="cyan")
            t2.add_column("base (ms)", justify="right", style="yellow")
            t2.add_column("head (ms)", justify="right", style="yellow")
            t2.add_column("Delta", justify="right")
            t2.add_column("Speedup", justify="right")

            def sort_key(fn: str, _bm_key: BenchmarkKey = bm_key) -> int:
                return result.base_function_ns.get(fn, {}).get(_bm_key, 0)

            for func_name in sorted(all_funcs, key=sort_key, reverse=True):
                b_ns = result.base_function_ns.get(func_name, {}).get(bm_key)
                h_ns = result.head_function_ns.get(func_name, {}).get(bm_key)

                # Shorten function name for display
                short_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name

                t2.add_row(short_name, _fmt_ms(b_ns), _fmt_ms(h_ns), _fmt_delta(b_ns, h_ns), _fmt_speedup(b_ns, h_ns))

            # Totals row
            t2.add_section()
            t2.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{_fmt_ms(base_ns)}[/bold]",
                f"[bold]{_fmt_ms(head_ns)}[/bold]",
                _fmt_delta(base_ns, head_ns),
                _fmt_speedup(base_ns, head_ns),
            )
            console.print(t2, justify="center")

    console.print()


def _fmt_ms(ns: Optional[int]) -> str:
    if ns is None:
        return "-"
    ms = ns / 1_000_000
    if ms >= 1000:
        return f"{ms:,.0f}"
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 1:
        return f"{ms:.1f}"
    return f"{ms:.2f}"


def _fmt_speedup(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None or after == 0:
        return "-"
    ratio = before / after
    if ratio >= 1:
        return f"[green]{ratio:.2f}x[/green]"
    return f"[red]{ratio:.2f}x[/red]"


def _fmt_delta(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None:
        return "-"
    delta_ms = (after - before) / 1_000_000
    pct = ((after - before) / before) * 100 if before != 0 else 0
    if delta_ms < 0:
        return f"[green]{delta_ms:+,.0f}ms ({pct:+.0f}%)[/green]"
    return f"[red]{delta_ms:+,.0f}ms ({pct:+.0f}%)[/red]"


def _md_speedup(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None or after == 0:
        return "-"
    ratio = before / after
    emoji = "\U0001f7e2" if ratio >= 1 else "\U0001f534"
    return f"{emoji} {ratio:.2f}x"


def _md_delta(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None:
        return "-"
    delta_ms = (after - before) / 1_000_000
    pct = ((after - before) / before) * 100 if before != 0 else 0
    if delta_ms < 0:
        return f"{delta_ms:+,.0f}ms ({pct:+.0f}%)"
    return f"+{delta_ms:,.0f}ms ({pct:+.0f}%)"


def _md_bar(before: Optional[int], after: Optional[int], width: int = 10) -> str:
    """Render a unicode progress bar showing the change from before to after.

    Improvement (after < before) shows green filled portion for the reduction.
    Regression (after > before) shows the bar in reverse.
    """
    if before is None or after is None or before == 0:
        return "-"
    pct = ((before - after) / before) * 100
    filled = round(abs(pct) / 100 * width)
    filled = max(0, min(filled, width))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"`{bar}` {pct:+.0f}%"


def _pct_bar(pct: float, width: int = 10) -> str:
    """Render a unicode bar representing a percentage share."""
    filled = round(pct / 100 * width)
    filled = max(0, min(filled, width))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"`{bar}` {pct:.1f}%"
