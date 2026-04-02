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

    from codeflash.benchmarking.plugin.plugin import BenchmarkStats, MemoryStats
    from codeflash.models.function_types import FunctionToOptimize
    from codeflash.models.models import BenchmarkKey

_GREEN_TPL = "[green]%+.0f%%[/green]"

_RED_TPL = "[red]%+.0f%%[/red]"


@dataclass
class CompareResult:
    base_ref: str
    head_ref: str
    base_stats: dict[BenchmarkKey, BenchmarkStats] = field(default_factory=dict)
    head_stats: dict[BenchmarkKey, BenchmarkStats] = field(default_factory=dict)
    base_function_ns: dict[str, dict[BenchmarkKey, float]] = field(default_factory=dict)
    head_function_ns: dict[str, dict[BenchmarkKey, float]] = field(default_factory=dict)
    base_memory: dict[BenchmarkKey, MemoryStats] = field(default_factory=dict)
    head_memory: dict[BenchmarkKey, MemoryStats] = field(default_factory=dict)

    def format_markdown(self) -> str:
        if not self.base_stats and not self.head_stats and not self.base_memory and not self.head_memory:
            return "_No benchmark results to compare._"

        base_short = self.base_ref[:12]
        head_short = self.head_ref[:12]
        all_keys = sorted(
            set(self.base_stats) | set(self.head_stats) | set(self.base_memory) | set(self.head_memory), key=str
        )
        sections: list[str] = [f"## Benchmark: `{base_short}` vs `{head_short}`"]

        for bm_key in all_keys:
            base_s = self.base_stats.get(bm_key)
            head_s = self.head_stats.get(bm_key)

            bm_name = str(bm_key).rsplit("::", 1)[-1] if "::" in str(bm_key) else str(bm_key)

            lines = [f"### {bm_name}"]

            # Timing table (skip for memory-only benchmark keys)
            if base_s or head_s:
                lines.extend(
                    [
                        "",
                        "| | Min | Median | Mean | OPS | Rounds |",
                        "|:---|---:|---:|---:|---:|---:|",
                        f"| `{base_short}` (base) | {fmt_us(base_s.min_ns) if base_s else '-'}"
                        f" | {fmt_us(base_s.median_ns) if base_s else '-'}"
                        f" | {fmt_us(base_s.mean_ns) if base_s else '-'}"
                        f" | {md_ops(base_s.mean_ns) if base_s else '-'}"
                        f" | {f'{base_s.rounds:,}' if base_s else '-'} |",
                        f"| `{head_short}` (head) | {fmt_us(head_s.min_ns) if head_s else '-'}"
                        f" | {fmt_us(head_s.median_ns) if head_s else '-'}"
                        f" | {fmt_us(head_s.mean_ns) if head_s else '-'}"
                        f" | {md_ops(head_s.mean_ns) if head_s else '-'}"
                        f" | {f'{head_s.rounds:,}' if head_s else '-'} |",
                        f"| **Speedup** | **{md_speedup_val(base_s.min_ns, head_s.min_ns) if base_s and head_s else '-'}**"
                        f" | **{md_speedup_val(base_s.median_ns, head_s.median_ns) if base_s and head_s else '-'}**"
                        f" | **{md_speedup_val(base_s.mean_ns, head_s.mean_ns) if base_s and head_s else '-'}**"
                        f" | **{md_speedup_val(base_s.mean_ns, head_s.mean_ns) if base_s and head_s else '-'}**"
                        f" | |",
                    ]
                )

                # Per-function breakdown
                all_funcs: set[str] = set()
                for d in [self.base_function_ns, self.head_function_ns]:
                    for func_name, bm_dict in d.items():
                        if bm_key in bm_dict:
                            all_funcs.add(func_name)

                if all_funcs:

                    def sort_key(fn: str, _bm_key: BenchmarkKey = bm_key) -> float:
                        return self.base_function_ns.get(fn, {}).get(_bm_key, 0)

                    sorted_funcs = sorted(all_funcs, key=sort_key, reverse=True)

                    lines.append("")
                    lines.append("| Function | base (μs) | head (μs) | Improvement | Speedup |")
                    lines.append("|:---|---:|---:|:---|---:|")

                    for func_name in sorted_funcs:
                        b = self.base_function_ns.get(func_name, {}).get(bm_key)
                        h = self.head_function_ns.get(func_name, {}).get(bm_key)
                        short_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
                        lines.append(
                            f"| `{short_name}` | {fmt_us(b)} | {fmt_us(h)} | {md_bar(b, h)} | {md_speedup(b, h)} |"
                        )

            # Memory section (always show for memory-only keys, otherwise skip when delta is negligible)
            base_mem = self.base_memory.get(bm_key)
            head_mem = self.head_memory.get(bm_key)
            memory_only_key = not base_s and not head_s
            if memory_only_key or has_meaningful_memory_change(base_mem, head_mem):
                lines.append("")
                lines.append("#### Memory")
                lines.append("")
                lines.append("| Ref | Peak Memory | Allocations | Delta |")
                lines.append("|:---|---:|---:|:---|")
                if base_mem:
                    lines.append(
                        f"| `{base_short}` (base) | {md_bytes(base_mem.peak_memory_bytes)}"
                        f" | {base_mem.total_allocations:,} | |"
                    )
                if head_mem:
                    delta = md_memory_delta(
                        base_mem.peak_memory_bytes if base_mem else None, head_mem.peak_memory_bytes
                    )
                    lines.append(
                        f"| `{head_short}` (head) | {md_bytes(head_mem.peak_memory_bytes)}"
                        f" | {head_mem.total_allocations:,} | {delta} |"
                    )

            sections.append("\n".join(lines))

        sections.append("---\n*Generated by codeflash optimization agent*")
        return "\n\n".join(sections)


@dataclass
class ScriptCompareResult:
    base_ref: str
    head_ref: str
    base_results: dict[str, float] = field(default_factory=dict)
    head_results: dict[str, float] = field(default_factory=dict)
    base_memory: Optional[MemoryStats] = None
    head_memory: Optional[MemoryStats] = None

    def format_markdown(self) -> str:
        if not self.base_results and not self.head_results and not self.base_memory and not self.head_memory:
            return "_No benchmark results to compare._"

        base_short = self.base_ref[:12]
        head_short = self.head_ref[:12]
        lines: list[str] = [f"## Benchmark: `{base_short}` vs `{head_short}`"]

        all_keys = sorted((set(self.base_results) | set(self.head_results)) - {"__total__"})
        has_total = "__total__" in self.base_results or "__total__" in self.head_results

        lines.extend(["", "| Key | Base | Head | Delta | Speedup |", "|:---|---:|---:|:---|---:|"])
        for key in all_keys:
            b = self.base_results.get(key)
            h = self.head_results.get(key)
            lines.append(
                f"| `{key}` | {_fmt_seconds(b)} | {_fmt_seconds(h)} | {_md_delta_s(b, h)} | {md_speedup(b, h)} |"
            )

        if has_total:
            b = self.base_results.get("__total__")
            h = self.head_results.get("__total__")
            lines.append(
                f"| **TOTAL** | **{_fmt_seconds(b)}** | **{_fmt_seconds(h)}** | {_md_delta_s(b, h)} | {md_speedup(b, h)} |"
            )

        if self.base_memory or self.head_memory:
            lines.extend(
                ["", "#### Memory", "", "| Ref | Peak Memory | Allocations | Delta |", "|:---|---:|---:|:---|"]
            )
            if self.base_memory:
                lines.append(
                    f"| `{base_short}` (base) | {md_bytes(self.base_memory.peak_memory_bytes)}"
                    f" | {self.base_memory.total_allocations:,} | |"
                )
            if self.head_memory:
                delta = md_memory_delta(
                    self.base_memory.peak_memory_bytes if self.base_memory else None, self.head_memory.peak_memory_bytes
                )
                lines.append(
                    f"| `{head_short}` (head) | {md_bytes(self.head_memory.peak_memory_bytes)}"
                    f" | {self.head_memory.total_allocations:,} | {delta} |"
                )

        lines.extend(["", "---", "*Generated by codeflash optimization agent*"])
        return "\n".join(lines)


def compare_branches(
    base_ref: str,
    head_ref: str,
    project_root: Path,
    benchmarks_root: Path,
    tests_root: Path,
    functions: Optional[dict[Path, list[FunctionToOptimize]]] = None,
    timeout: int = 600,
    memory: bool = False,
) -> CompareResult:
    """Compare benchmark performance between two git refs.

    If functions is None, auto-detects changed functions from git diff.
    Returns a CompareResult with timing data from both refs.
    """
    import sys

    from codeflash.benchmarking.instrument_codeflash_trace import instrument_codeflash_trace_decorator
    from codeflash.benchmarking.plugin.plugin import CodeFlashBenchmarkPlugin
    from codeflash.benchmarking.trace_benchmarks import trace_benchmarks_pytest

    if memory and sys.platform == "win32":
        logger.error("--memory requires memray which is not available on Windows")
        return CompareResult(base_ref=base_ref, head_ref=head_ref)

    repo = git.Repo(project_root, search_parent_directories=True)
    repo_root = Path(repo.working_dir)

    # Auto-detect functions if not provided
    if functions is None:
        functions = discover_changed_functions(base_ref, head_ref, repo_root)
        if not functions:
            if not memory:
                logger.warning("No changed Python functions found between %s and %s", base_ref, head_ref)
                return CompareResult(base_ref=base_ref, head_ref=head_ref)
            logger.info("No changed top-level functions — running memory-only comparison")

    memory_only = memory and not functions

    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    base_short = base_ref[:12]
    head_short = head_ref[:12]

    from rich.tree import Tree

    if memory_only:
        fn_tree = Tree("[bold]Memory-only[/bold] [dim](no changed top-level functions)[/dim]", guide_style="dim")
    else:
        func_count = sum(len(fns) for fns in functions.values())
        file_count = len(functions)

        # Build function tree for the panel
        from os.path import commonpath

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
    base_memray_dir = worktree_dirs / f"memray-base-{timestamp}"
    head_memray_dir = worktree_dirs / f"memray-head-{timestamp}"
    memray_prefix = "cf-mem"

    result = CompareResult(base_ref=base_ref, head_ref=head_ref)

    from rich.console import Group

    step_labels = ["Creating worktrees"]
    if not memory_only:
        step_labels.extend([f"Benchmarking base ({base_short})", f"Benchmarking head ({head_short})"])
    if memory:
        step_labels.extend([f"Memory profiling base ({base_short})", f"Memory profiling head ({head_short})"])

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
        tree_height = 1 + sum(1 + len(fns) for fns in functions.values())
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
        step = 0
        with Live(build_panel(step), console=console, refresh_per_second=1) as live:
            # Create worktrees (resolve to SHAs to avoid "already checked out" errors)
            base_sha = repo.commit(base_ref).hexsha
            head_sha = repo.commit(head_ref).hexsha
            repo.git.worktree("add", str(base_worktree), base_sha)
            repo.git.worktree("add", str(head_worktree), head_sha)
            step += 1
            live.update(build_panel(step))

            if not memory_only:
                # Run trace benchmarks on base
                run_benchmark_on_worktree(
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
                step += 1
                live.update(build_panel(step))

                # Run trace benchmarks on head
                run_benchmark_on_worktree(
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

            # Memory profiling (reuses existing worktrees)
            if memory:
                from codeflash.benchmarking.trace_benchmarks import memory_benchmarks_pytest

                wt_base_benchmarks = base_worktree / benchmarks_root.relative_to(repo_root)
                wt_head_benchmarks = head_worktree / benchmarks_root.relative_to(repo_root)

                # Copy benchmarks into worktrees if not present (e.g. base ref predates benchmark dir)
                if memory_only:
                    import shutil

                    for wt_bm in [wt_base_benchmarks, wt_head_benchmarks]:
                        if not wt_bm.exists() and benchmarks_root.is_dir():
                            shutil.copytree(benchmarks_root, wt_bm)

                if not memory_only:
                    step += 1
                    live.update(build_panel(step))
                memory_benchmarks_pytest(wt_base_benchmarks, base_worktree, base_memray_dir, memray_prefix, timeout)

                step += 1
                live.update(build_panel(step))
                memory_benchmarks_pytest(wt_head_benchmarks, head_worktree, head_memray_dir, memray_prefix, timeout)

        # Load results
        if not memory_only:
            if base_trace_db.exists():
                result.base_stats = CodeFlashBenchmarkPlugin.get_benchmark_timings(base_trace_db)
                result.base_function_ns = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(base_trace_db)

            if head_trace_db.exists():
                result.head_stats = CodeFlashBenchmarkPlugin.get_benchmark_timings(head_trace_db)
                result.head_function_ns = CodeFlashBenchmarkPlugin.get_function_benchmark_timings(head_trace_db)

        if memory:
            from codeflash.benchmarking.plugin.plugin import MemoryStats

            if base_memray_dir.exists():
                result.base_memory = MemoryStats.parse_memray_results(base_memray_dir, memray_prefix)
            if head_memray_dir.exists():
                result.head_memory = MemoryStats.parse_memray_results(head_memray_dir, memray_prefix)

        # Render comparison
        render_comparison(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")

    finally:
        # Cleanup worktrees
        from codeflash.code_utils.git_worktree_utils import remove_worktree

        remove_worktree(base_worktree)
        remove_worktree(head_worktree)
        repo.git.worktree("prune")
        # Cleanup trace DBs and memray dirs
        for db in [base_trace_db, head_trace_db]:
            if db.exists():
                db.unlink()
        if memory:
            import shutil

            for memray_dir in [base_memray_dir, head_memray_dir]:
                if memray_dir.exists():
                    shutil.rmtree(memray_dir)

    return result


def discover_changed_functions(base_ref: str, head_ref: str, repo_root: Path) -> dict[Path, list[FunctionToOptimize]]:
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

        modified_fns = find_changed_toplevel_functions(abs_path, changed_lines)
        if modified_fns:
            result[abs_path] = modified_fns

    return result


def find_changed_toplevel_functions(file_path: Path, changed_lines: set[int]) -> list[FunctionToOptimize]:
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


def run_benchmark_on_worktree(
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

    # If benchmarks dir doesn't exist in this worktree (e.g. base ref predates
    # the benchmark), copy it from the working directory so both refs can run.
    if not wt_benchmarks.exists() and benchmarks_root.is_dir():
        import shutil

        shutil.copytree(benchmarks_root, wt_benchmarks)

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


def render_comparison(result: CompareResult) -> None:
    """Render Rich comparison tables to console."""
    has_timing = result.base_stats or result.head_stats
    has_memory = result.base_memory or result.head_memory
    if not has_timing and not has_memory:
        logger.warning("No benchmark results to compare")
        return

    base_short = result.base_ref[:12]
    head_short = result.head_ref[:12]

    all_benchmark_keys = (
        set(result.base_stats.keys())
        | set(result.head_stats.keys())
        | set(result.base_memory.keys())
        | set(result.head_memory.keys())
    )

    for bm_key in sorted(all_benchmark_keys, key=str):
        bm_name = str(bm_key).rsplit("::", 1)[-1] if "::" in str(bm_key) else str(bm_key)
        console.print()
        console.rule(f"[bold]{bm_name}[/bold]")
        console.print()

        base_s = result.base_stats.get(bm_key)
        head_s = result.head_stats.get(bm_key)

        # Table 1: Statistical summary (skip for memory-only benchmark keys)
        if base_s or head_s:
            t1 = Table(title="End-to-End (per iteration)", border_style="blue", show_lines=True, expand=False)
            t1.add_column("Ref", style="bold cyan")
            t1.add_column("Min", justify="right")
            t1.add_column("Median", justify="right")
            t1.add_column("Mean", justify="right")
            t1.add_column("OPS", justify="right")
            t1.add_column("Rounds", justify="right")

            if base_s:
                t1.add_row(
                    f"{base_short} (base)",
                    fmt_time(base_s.min_ns),
                    fmt_time(base_s.median_ns),
                    fmt_time(base_s.mean_ns),
                    fmt_ops(base_s.mean_ns),
                    f"{base_s.rounds:,}",
                )
            if head_s:
                t1.add_row(
                    f"{head_short} (head)",
                    fmt_time(head_s.min_ns),
                    fmt_time(head_s.median_ns),
                    fmt_time(head_s.mean_ns),
                    fmt_ops(head_s.mean_ns),
                    f"{head_s.rounds:,}",
                )
            if base_s and head_s:
                t1.add_section()
                t1.add_row(
                    "[bold]Speedup[/bold]",
                    fmt_speedup(base_s.min_ns, head_s.min_ns),
                    fmt_speedup(base_s.median_ns, head_s.median_ns),
                    fmt_speedup(base_s.mean_ns, head_s.mean_ns),
                    fmt_speedup_ops(base_s.mean_ns, head_s.mean_ns),
                    "",
                )
            console.print(t1, justify="center")

            # Table 2: Per-function breakdown (average per-iteration)
            all_funcs: set[str] = set()
            for d in [result.base_function_ns, result.head_function_ns]:
                for func_name, bm_dict in d.items():
                    if bm_key in bm_dict:
                        all_funcs.add(func_name)

            if all_funcs:
                console.print()

                t2 = Table(
                    title="Per-Function Breakdown (avg per iteration)",
                    border_style="blue",
                    show_lines=True,
                    expand=False,
                )
                t2.add_column("Function", style="cyan")
                t2.add_column("base", justify="right", style="yellow")
                t2.add_column("head", justify="right", style="yellow")
                t2.add_column("Delta", justify="right")
                t2.add_column("Speedup", justify="right")

                def sort_key(fn: str, _bm_key: BenchmarkKey = bm_key) -> float:
                    return result.base_function_ns.get(fn, {}).get(_bm_key, 0)

                for func_name in sorted(all_funcs, key=sort_key, reverse=True):
                    b_ns = result.base_function_ns.get(func_name, {}).get(bm_key)
                    h_ns = result.head_function_ns.get(func_name, {}).get(bm_key)
                    short_name = func_name.rsplit(".", 1)[-1] if "." in func_name else func_name
                    t2.add_row(
                        short_name, fmt_time(b_ns), fmt_time(h_ns), fmt_delta(b_ns, h_ns), fmt_speedup(b_ns, h_ns)
                    )

                console.print(t2, justify="center")

        # Table 3: Memory (always show for memory-only keys, otherwise skip when delta is negligible)
        base_mem = result.base_memory.get(bm_key)
        head_mem = result.head_memory.get(bm_key)
        memory_only_key = not base_s and not head_s
        if memory_only_key or has_meaningful_memory_change(base_mem, head_mem):
            console.print()
            t3 = Table(title="Memory (peak per test)", border_style="magenta", show_lines=True, expand=False)
            t3.add_column("Ref", style="bold cyan")
            t3.add_column("Peak Memory", justify="right")
            t3.add_column("Allocations", justify="right")
            t3.add_column("Delta", justify="right")

            if base_mem:
                t3.add_row(
                    f"{base_short} (base)", fmt_bytes(base_mem.peak_memory_bytes), f"{base_mem.total_allocations:,}", ""
                )
            if head_mem:
                delta = fmt_memory_delta(base_mem.peak_memory_bytes if base_mem else None, head_mem.peak_memory_bytes)
                t3.add_row(
                    f"{head_short} (head)",
                    fmt_bytes(head_mem.peak_memory_bytes),
                    f"{head_mem.total_allocations:,}",
                    delta,
                )
            console.print(t3, justify="center")

    console.print()


# --- Formatting helpers ---


def fmt_time(ns: Optional[float]) -> str:
    if ns is None:
        return "-"
    us = ns / 1_000
    if us >= 1_000_000:
        return f"{us / 1_000_000:,.1f}s"
    if us >= 1_000:
        return f"{us / 1_000:,.1f}ms"
    if us >= 1:
        return f"{us:,.1f}μs"
    return f"{ns:,.1f}ns"


def fmt_us(ns: Optional[float]) -> str:
    if ns is None:
        return "-"
    return f"{ns / 1_000:,.2f}μs"


def fmt_ops(mean_ns: Optional[float]) -> str:
    if mean_ns is None or mean_ns == 0:
        return "-"
    ops = 1e9 / mean_ns
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:,.2f} Mops/s"
    if ops >= 1_000:
        return f"{ops / 1_000:,.2f} Kops/s"
    return f"{ops:,.2f} ops/s"


def md_ops(mean_ns: Optional[float]) -> str:
    if mean_ns is None or mean_ns == 0:
        return "-"
    ops = 1e9 / mean_ns
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:,.2f} Mops/s"
    if ops >= 1_000:
        return f"{ops / 1_000:,.2f} Kops/s"
    return f"{ops:,.2f} ops/s"


def fmt_speedup_ops(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None or before == 0:
        return "-"
    ratio = before / after
    if ratio >= 1:
        return f"[green]{ratio:.2f}x[/green]"
    return f"[red]{ratio:.2f}x[/red]"


def fmt_speedup(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None or after == 0:
        return "-"
    ratio = before / after
    if ratio >= 1:
        return f"[green]{ratio:.2f}x[/green]"
    return f"[red]{ratio:.2f}x[/red]"


def fmt_delta(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "-"
    pct = ((after - before) / before) * 100 if before != 0 else 0
    if pct < 0:
        return _GREEN_TPL % pct
    return _RED_TPL % pct


def md_speedup(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None or after == 0:
        return "-"
    ratio = before / after
    emoji = "\U0001f7e2" if ratio >= 1 else "\U0001f534"
    return f"{emoji} {ratio:.2f}x"


def md_speedup_val(before: float, after: float) -> str:
    if after == 0:
        return "-"
    ratio = before / after
    emoji = "\U0001f7e2" if ratio >= 1 else "\U0001f534"
    return f"{emoji} {ratio:.2f}x"


def md_bar(before: Optional[float], after: Optional[float], width: int = 10) -> str:
    if before is None or after is None or before == 0:
        return "-"
    pct = ((before - after) / before) * 100
    filled = round(abs(pct) / 100 * width)
    filled = min(filled, width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"`{bar}` {pct:+.0f}%"


def fmt_bytes(b: Optional[int]) -> str:
    if b is None:
        return "-"
    if b >= 1 << 30:
        return f"{b / (1 << 30):,.1f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):,.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):,.1f} KiB"
    return f"{b:,} B"


def fmt_memory_delta(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None or before == 0:
        return "-"
    pct = ((after - before) / before) * 100
    if pct < 0:
        return _GREEN_TPL % pct
    return _RED_TPL % pct


def md_bytes(b: Optional[int]) -> str:
    if b is None:
        return "-"
    if b >= 1 << 30:
        return f"{b / (1 << 30):,.1f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):,.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):,.1f} KiB"
    return f"{b:,} B"


def md_memory_delta(before: Optional[int], after: Optional[int]) -> str:
    if before is None or after is None or before == 0:
        return "-"
    pct = ((after - before) / before) * 100
    emoji = "\U0001f7e2" if pct <= 0 else "\U0001f534"
    return f"{emoji} {pct:+.0f}%"


def has_meaningful_memory_change(
    base_mem: Optional[MemoryStats], head_mem: Optional[MemoryStats], threshold_pct: float = 1.0
) -> bool:
    """Return True if peak memory or allocation count changed by more than threshold_pct."""
    if base_mem is None or head_mem is None:
        return base_mem is not None or head_mem is not None
    if base_mem.peak_memory_bytes == 0 and head_mem.peak_memory_bytes == 0:
        return False
    if base_mem.peak_memory_bytes > 0:
        mem_pct = abs((head_mem.peak_memory_bytes - base_mem.peak_memory_bytes) / base_mem.peak_memory_bytes) * 100
        if mem_pct > threshold_pct:
            return True
    if base_mem.total_allocations > 0:
        alloc_pct = abs((head_mem.total_allocations - base_mem.total_allocations) / base_mem.total_allocations) * 100
        if alloc_pct > threshold_pct:
            return True
    return False


# --- Script-mode comparison ---


def _fmt_seconds(s: Optional[float]) -> str:
    if s is None:
        return "-"
    if s >= 60:
        return f"{s / 60:,.1f}m"
    return f"{s:,.2f}s"


def _fmt_delta_s(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return "-"
    pct = ((after - before) / before) * 100 if before != 0 else 0
    if pct < 0:
        return _GREEN_TPL % pct
    return _RED_TPL % pct


def _md_delta_s(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None or before == 0:
        return "-"
    pct = ((after - before) / before) * 100
    emoji = "\U0001f7e2" if pct <= 0 else "\U0001f534"
    return f"{emoji} {pct:+.1f}%"


def _speedup_s(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None or after == 0:
        return "-"
    ratio = before / after
    if ratio >= 1:
        return f"[green]{ratio:.2f}x[/green]"
    return f"[red]{ratio:.2f}x[/red]"


def compare_with_script(
    base_ref: str,
    head_ref: str,
    project_root: Path,
    script_cmd: str,
    script_output: str,
    timeout: int = 600,
    memory: bool = False,
) -> ScriptCompareResult:
    """Compare benchmark performance between two git refs using a custom script.

    The script is run in each worktree with CWD set to the worktree root.
    It must produce a JSON file at script_output (relative to worktree root)
    mapping keys to seconds, e.g. {"test1": 1.23, "__total__": 4.56}.
    """
    import sys

    if memory and sys.platform == "win32":
        logger.error("--memory requires memray which is not available on Windows")
        return ScriptCompareResult(base_ref=base_ref, head_ref=head_ref)

    repo = git.Repo(project_root, search_parent_directories=True)

    from codeflash.code_utils.git_worktree_utils import worktree_dirs

    worktree_dirs.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    base_worktree = worktree_dirs / f"compare-base-{timestamp}"
    head_worktree = worktree_dirs / f"compare-head-{timestamp}"
    base_memray_bin = worktree_dirs / f"script-memray-base-{timestamp}.bin"
    head_memray_bin = worktree_dirs / f"script-memray-head-{timestamp}.bin"

    result = ScriptCompareResult(base_ref=base_ref, head_ref=head_ref)

    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    base_short = base_ref[:12]
    head_short = head_ref[:12]

    step_labels = [
        "Creating worktrees",
        f"Running benchmark on base ({base_short})",
        f"Running benchmark on head ({head_short})",
    ]

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
        return Panel(
            Group(
                Text.from_markup(
                    f"[bold cyan]{base_short}[/bold cyan] (base) vs [bold cyan]{head_short}[/bold cyan] (head)"
                ),
                "",
                Text.from_markup(f"[dim]Script:[/dim] {script_cmd}"),
                "",
                build_steps(current_step),
            ),
            title="[bold]Script Benchmark Compare[/bold]",
            border_style="cyan",
            expand=True,
            padding=(1, 2),
        )

    try:
        step = 0
        with Live(build_panel(step), console=console, refresh_per_second=1) as live:
            base_sha = repo.commit(base_ref).hexsha
            head_sha = repo.commit(head_ref).hexsha
            repo.git.worktree("add", str(base_worktree), base_sha)
            repo.git.worktree("add", str(head_worktree), head_sha)
            step += 1
            live.update(build_panel(step))

            # Run script on base
            result.base_results = _run_script_in_worktree(
                script_cmd, base_worktree, script_output, timeout, base_memray_bin if memory else None
            )
            step += 1
            live.update(build_panel(step))

            # Run script on head
            result.head_results = _run_script_in_worktree(
                script_cmd, head_worktree, script_output, timeout, head_memray_bin if memory else None
            )

        # Parse memory results
        if memory:
            result.base_memory = _parse_memray_bin(base_memray_bin)
            result.head_memory = _parse_memray_bin(head_memray_bin)

        render_script_comparison(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — cleaning up...[/yellow]")

    finally:
        from codeflash.code_utils.git_worktree_utils import remove_worktree

        remove_worktree(base_worktree)
        remove_worktree(head_worktree)
        repo.git.worktree("prune")
        for f in [base_memray_bin, head_memray_bin]:
            if f.exists():
                f.unlink()

    return result


def _run_script_in_worktree(
    script_cmd: str, worktree_dir: Path, script_output: str, timeout: int, memray_bin: Optional[Path]
) -> dict[str, float]:
    import json

    cmd = script_cmd
    if memray_bin:
        cmd = f"python -m memray run --trace-python-allocators -o {memray_bin} -- {cmd}"

    try:
        proc = subprocess.run(  # noqa: S602
            cmd, shell=True, cwd=worktree_dir, timeout=timeout, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            logger.warning(f"Script exited with code {proc.returncode}")
            if proc.stderr:
                logger.debug(f"Script stderr:\n{proc.stderr[:2000]}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Script timed out after {timeout}s")
        return {}

    output_path = worktree_dir / script_output
    if not output_path.exists():
        logger.warning(f"Script output not found at {output_path}")
        return {}

    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("Script output JSON is not a dict")
            return {}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse script output JSON: {e}")
        return {}


def _parse_memray_bin(bin_path: Path) -> Optional[MemoryStats]:
    if not bin_path.exists():
        return None
    try:
        from memray import FileReader

        from codeflash.benchmarking.plugin.plugin import MemoryStats

        reader = FileReader(str(bin_path))
        meta = reader.metadata
        stats = MemoryStats(peak_memory_bytes=meta.peak_memory, total_allocations=meta.total_allocations)
        reader.close()
        return stats
    except ImportError:
        logger.warning("memray not installed — skipping memory results")
        return None
    except OSError as e:
        logger.warning(f"Failed to read memray binary: {e}")
        return None


def render_script_comparison(result: ScriptCompareResult) -> None:
    has_timing = result.base_results or result.head_results
    has_memory = result.base_memory or result.head_memory
    if not has_timing and not has_memory:
        logger.warning("No benchmark results to compare")
        return

    base_short = result.base_ref[:12]
    head_short = result.head_ref[:12]

    console.print()
    console.rule(f"[bold]Script Benchmark: {base_short} vs {head_short}[/bold]")
    console.print()

    if has_timing:
        all_keys = sorted((set(result.base_results) | set(result.head_results)) - {"__total__"})
        has_total = "__total__" in result.base_results or "__total__" in result.head_results

        t = Table(title="Benchmark Results", border_style="blue", show_lines=True, expand=False)
        t.add_column("Key", style="cyan")
        t.add_column("Base", justify="right", style="yellow")
        t.add_column("Head", justify="right", style="yellow")
        t.add_column("Delta", justify="right")
        t.add_column("Speedup", justify="right")

        for key in all_keys:
            b = result.base_results.get(key)
            h = result.head_results.get(key)
            t.add_row(key, _fmt_seconds(b), _fmt_seconds(h), _fmt_delta_s(b, h), _speedup_s(b, h))

        if has_total:
            t.add_section()
            b = result.base_results.get("__total__")
            h = result.head_results.get("__total__")
            t.add_row("[bold]TOTAL[/bold]", _fmt_seconds(b), _fmt_seconds(h), _fmt_delta_s(b, h), _speedup_s(b, h))

        console.print(t, justify="center")

    if has_memory:
        console.print()
        t_mem = Table(title="Memory (aggregate)", border_style="magenta", show_lines=True, expand=False)
        t_mem.add_column("Ref", style="bold cyan")
        t_mem.add_column("Peak Memory", justify="right")
        t_mem.add_column("Allocations", justify="right")
        t_mem.add_column("Delta", justify="right")

        if result.base_memory:
            t_mem.add_row(
                f"{base_short} (base)",
                fmt_bytes(result.base_memory.peak_memory_bytes),
                f"{result.base_memory.total_allocations:,}",
                "",
            )
        if result.head_memory:
            delta = fmt_memory_delta(
                result.base_memory.peak_memory_bytes if result.base_memory else None,
                result.head_memory.peak_memory_bytes,
            )
            t_mem.add_row(
                f"{head_short} (head)",
                fmt_bytes(result.head_memory.peak_memory_bytes),
                f"{result.head_memory.total_allocations:,}",
                delta,
            )
        console.print(t_mem, justify="center")

    console.print()
