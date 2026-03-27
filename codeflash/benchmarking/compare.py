"""Cross-branch benchmark comparison.

Compares benchmark performance between two git refs by:
1. Auto-detecting changed functions (or using an explicit list)
2. Creating worktrees for each ref
3. Instrumenting functions with @codeflash_trace
4. Running benchmarks via trace_benchmarks_pytest
5. Rendering a side-by-side Rich comparison table
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import git
from rich.console import Console
from rich.table import Table

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.compat import codeflash_cache_dir

if TYPE_CHECKING:
    from codeflash.models.models import BenchmarkKey


@dataclass
class CompareResult:
    base_ref: str
    head_ref: str
    base_total_ns: dict[BenchmarkKey, int] = field(default_factory=dict)
    head_total_ns: dict[BenchmarkKey, int] = field(default_factory=dict)
    base_function_ns: dict[str, dict[BenchmarkKey, int]] = field(default_factory=dict)
    head_function_ns: dict[str, dict[BenchmarkKey, int]] = field(default_factory=dict)


def compare_branches(
    base_ref: str,
    head_ref: str,
    project_root: Path,
    benchmarks_root: Path,
    tests_root: Path,
    functions: Optional[dict[Path, list]] = None,
    timeout: int = 600,
    svg_output: Optional[Path] = None,
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

    func_count = sum(len(fns) for fns in functions.values())
    file_count = len(functions)
    logger.info(f"Instrumenting {func_count} functions across {file_count} files")
    for file_path, fns in functions.items():
        rel = file_path.relative_to(repo_root) if file_path.is_relative_to(repo_root) else file_path
        logger.info(f"  {rel}: {', '.join(f.function_name for f in fns)}")

    # Set up worktree paths and trace DB paths
    worktree_dir = codeflash_cache_dir / "worktrees"
    worktree_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    base_worktree = worktree_dir / f"compare-base-{timestamp}"
    head_worktree = worktree_dir / f"compare-head-{timestamp}"
    base_trace_db = worktree_dir / f"trace-base-{timestamp}.db"
    head_trace_db = worktree_dir / f"trace-head-{timestamp}.db"

    result = CompareResult(base_ref=base_ref, head_ref=head_ref)

    try:
        # Create worktrees
        logger.info(f"Creating worktree for base ref: {base_ref}")
        repo.git.worktree("add", str(base_worktree), base_ref)

        logger.info(f"Creating worktree for head ref: {head_ref}")
        repo.git.worktree("add", str(head_worktree), head_ref)

        # Run benchmarks on base
        logger.info(f"Running benchmarks on {base_ref}...")
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

        # Run benchmarks on head
        logger.info(f"Running benchmarks on {head_ref}...")
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
        _render_comparison(result, svg_output=svg_output)

    finally:
        # Cleanup worktrees
        _cleanup_worktree(repo, base_worktree)
        _cleanup_worktree(repo, head_worktree)
        # Cleanup trace DBs
        for db in [base_trace_db, head_trace_db]:
            if db.exists():
                db.unlink()

    return result


def _discover_changed_functions(base_ref: str, head_ref: str, repo_root: Path) -> dict[Path, list]:
    """Find only functions whose bodies overlap with changed lines between refs."""
    from io import StringIO

    from unidiff import PatchSet

    from codeflash.discovery.functions_to_optimize import find_all_functions_in_file

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

        added_lines = {
            line.target_line_no for hunk in patched_file for line in hunk if line.is_added and line.value.strip()
        }
        deleted_lines = {hunk.target_start for hunk in patched_file}
        # Use added lines if available, otherwise use hunk starts (deletion-only changes)
        line_nos = added_lines if added_lines else deleted_lines
        if line_nos:
            changed_lines_by_file[abs_path] = line_nos

    # Discover all functions in changed files, then filter to those overlapping changed lines
    result: dict[Path, list] = {}
    for abs_path, changed_lines in changed_lines_by_file.items():
        if not abs_path.exists():
            logger.debug(f"Skipping {abs_path} (does not exist)")
            continue

        discovered = find_all_functions_in_file(abs_path)
        for fp, all_fns in discovered.items():
            modified_fns = []
            for fn in all_fns:
                # Skip methods inside classes — they can be called thousands of times
                # (e.g., CST visitor methods) and tracing overhead is catastrophic.
                # We only instrument top-level functions and static/class-level functions.
                if fn.parents:
                    continue
                if fn.starting_line is None or fn.ending_line is None:
                    # Can't determine range — include it to be safe
                    modified_fns.append(fn)
                    continue
                fn_lines = set(range(fn.starting_line, fn.ending_line + 1))
                if fn_lines & changed_lines:
                    modified_fns.append(fn)

            if modified_fns:
                result[fp] = modified_fns

    return result


def _run_benchmark_on_worktree(
    worktree_dir: Path,
    repo_root: Path,
    functions: dict[Path, list],
    benchmarks_root: Path,
    tests_root: Path,
    trace_db: Path,
    timeout: int,
    instrument_fn,
    trace_fn,
) -> None:
    """Instrument, benchmark, and restore source in a worktree."""
    # Remap function paths from repo_root to worktree_dir
    worktree_functions: dict[Path, list] = {}
    for file_path, fns in functions.items():
        rel = file_path.relative_to(repo_root) if file_path.is_relative_to(repo_root) else file_path
        wt_path = worktree_dir / rel
        if not wt_path.exists():
            logger.debug(f"Skipping {rel} (not present in this ref)")
            continue

        # Rebuild FunctionToOptimize with worktree paths
        from codeflash.models.function_types import FunctionToOptimize

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


def _cleanup_worktree(repo: git.Repo, worktree_dir: Path) -> None:
    """Remove a worktree, ignoring errors."""
    if not worktree_dir.exists():
        return
    try:
        repo.git.worktree("remove", "--force", str(worktree_dir))
        logger.debug(f"Removed worktree: {worktree_dir}")
    except Exception:
        logger.debug(f"Failed to remove worktree {worktree_dir}, attempting manual cleanup")
        import shutil

        try:
            shutil.rmtree(worktree_dir)
        except Exception:
            logger.warning(f"Could not clean up worktree: {worktree_dir}")


def _render_comparison(result: CompareResult, svg_output: Optional[Path] = None) -> None:
    """Render Rich comparison tables to console (and optionally SVG)."""
    if not result.base_total_ns and not result.head_total_ns:
        logger.warning("No benchmark results to compare")
        return

    console = Console(width=140, record=svg_output is not None)

    # Find all benchmark keys across both refs
    all_benchmark_keys = set(result.base_total_ns.keys()) | set(result.head_total_ns.keys())

    for bm_key in sorted(all_benchmark_keys, key=str):
        console.print()
        console.rule(f"[bold]Benchmark: {bm_key}[/bold]")
        console.print()

        base_ns = result.base_total_ns.get(bm_key)
        head_ns = result.head_total_ns.get(bm_key)

        # Table 1: Total benchmark time
        t1 = Table(title="Total Benchmark Time", border_style="blue", show_lines=True, width=100)
        t1.add_column("Ref", style="bold cyan", width=30)
        t1.add_column("Time (ms)", justify="right", width=15)
        t1.add_column("Delta", justify="right", width=25)
        t1.add_column("Speedup", justify="right", width=15)

        t1.add_row(result.base_ref, _fmt_ms(base_ns), "-", "-")
        t1.add_row(result.head_ref, _fmt_ms(head_ns), _fmt_delta(base_ns, head_ns), _fmt_speedup(base_ns, head_ns))
        console.print(t1)

        # Table 2: Per-function breakdown
        all_funcs = set()
        for d in [result.base_function_ns, result.head_function_ns]:
            for func_name, bm_dict in d.items():
                if bm_key in bm_dict:
                    all_funcs.add(func_name)

        if all_funcs:
            console.print()
            console.rule("[bold]Per-Function Breakdown[/bold]")
            console.print()

            t2 = Table(border_style="blue", show_lines=True, width=140)
            t2.add_column("Function", style="cyan", width=40, overflow="fold")
            t2.add_column(f"{result.base_ref} (ms)", justify="right", style="yellow", width=15)
            t2.add_column(f"{result.head_ref} (ms)", justify="right", style="yellow", width=15)
            t2.add_column("Delta", justify="right", width=25)
            t2.add_column("Speedup", justify="right", width=15)

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
            console.print(t2)

    console.print()

    if svg_output:
        console.save_svg(str(svg_output), title=f"Benchmark: {result.base_ref} vs {result.head_ref}")
        logger.info(f"Saved SVG to {svg_output}")


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
    return f"[red]+{delta_ms:,.0f}ms ({pct:+.0f}%)[/red]"
