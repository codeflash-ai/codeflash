"""CLI handler for `codeflash compare`."""

from __future__ import annotations

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.config_parser import parse_config_file


def run_compare(args: Namespace) -> None:
    """Entry point for the compare subcommand."""
    # Load project config
    pyproject_config, pyproject_file_path = parse_config_file(args.config_file)

    module_root = Path(pyproject_config.get("module_root", ".")).resolve()
    tests_root = Path(pyproject_config.get("tests_root", "tests")).resolve()
    benchmarks_root_str = pyproject_config.get("benchmarks_root")

    if not benchmarks_root_str:
        logger.error("benchmarks-root must be configured in [tool.codeflash] to use compare")
        sys.exit(1)

    benchmarks_root = Path(benchmarks_root_str).resolve()
    if not benchmarks_root.is_dir():
        logger.error(f"benchmarks-root {benchmarks_root} is not a valid directory")
        sys.exit(1)

    # Determine project_root from module_root (same logic as cli.py)
    if pyproject_file_path.parent == module_root:
        project_root = module_root
    else:
        project_root = module_root.parent.resolve()

    # Resolve head_ref
    head_ref = args.head_ref
    if args.pr:
        head_ref = _resolve_pr_branch(args.pr)
    if not head_ref:
        logger.error("Must provide head_ref or --pr")
        sys.exit(1)

    # Parse explicit functions if provided
    functions = None
    if args.functions:
        functions = _parse_functions_arg(args.functions, project_root)

    svg_output = Path(args.svg) if args.svg else None

    from codeflash.benchmarking.compare import compare_branches

    result = compare_branches(
        base_ref=args.base_ref,
        head_ref=head_ref,
        project_root=project_root,
        benchmarks_root=benchmarks_root,
        tests_root=tests_root,
        functions=functions,
        timeout=args.timeout,
        svg_output=svg_output,
    )

    if not result.base_total_ns and not result.head_total_ns:
        logger.warning("No benchmark data collected. Check that benchmarks-root is configured and benchmarks exist.")
        sys.exit(1)


def _resolve_pr_branch(pr_number: int) -> str:
    """Resolve a PR number to its head branch name using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(pr_number), "--json", "headRefName", "-q", ".headRefName"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        if branch:
            logger.info(f"Resolved PR #{pr_number} to branch: {branch}")
            return branch
        logger.error(f"Could not resolve PR #{pr_number} to a branch")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("gh CLI not found. Install it from https://cli.github.com/ or provide a branch name directly.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to resolve PR #{pr_number}: {e.stderr}")
        sys.exit(1)


def _parse_functions_arg(functions_str: str, project_root: Path) -> dict[Path, list]:
    """Parse --functions arg format: 'file.py::func1,func2;other.py::func3'."""
    from codeflash.models.function_types import FunctionToOptimize

    result: dict[Path, list] = {}
    for entry in functions_str.split(";"):
        entry = entry.strip()
        if "::" not in entry:
            logger.warning(f"Skipping malformed functions entry (missing '::'): {entry}")
            continue
        file_part, funcs_part = entry.split("::", 1)
        file_path = (project_root / file_part.strip()).resolve()
        if not file_path.exists():
            logger.warning(f"Skipping {file_path} (does not exist)")
            continue
        func_names = [f.strip() for f in funcs_part.split(",") if f.strip()]
        result[file_path] = [
            FunctionToOptimize(function_name=name, file_path=file_path, parents=[])
            for name in func_names
        ]
    return result
