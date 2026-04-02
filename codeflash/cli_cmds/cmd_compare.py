"""CLI handler for `codeflash compare`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.models.function_types import FunctionToOptimize

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

    from codeflash.cli_cmds.cli import project_root_from_module_root

    project_root = project_root_from_module_root(module_root, pyproject_file_path)

    # Resolve head_ref: explicit arg > --pr > current branch
    head_ref = args.head_ref
    if args.pr:
        head_ref = resolve_pr_branch(args.pr)
    if not head_ref:
        head_ref = get_current_branch()
        if not head_ref:
            logger.error("Must provide head_ref, --pr, or be on a branch")
            sys.exit(1)
        logger.info(f"Auto-detected head ref: {head_ref}")

    # Resolve base_ref: explicit arg > PR base branch > repo default branch
    base_ref = args.base_ref
    if not base_ref:
        base_ref = detect_base_ref(head_ref)
        if not base_ref:
            logger.error("Could not auto-detect base ref. Provide it explicitly or ensure gh CLI is available.")
            sys.exit(1)
        logger.info(f"Auto-detected base ref: {base_ref}")

    # Parse explicit functions if provided
    functions = None
    if args.functions:
        functions = parse_functions_arg(args.functions, project_root)

    from codeflash.benchmarking.compare import compare_branches

    result = compare_branches(
        base_ref=base_ref,
        head_ref=head_ref,
        project_root=project_root,
        benchmarks_root=benchmarks_root,
        tests_root=tests_root,
        functions=functions,
        timeout=args.timeout,
        memory=getattr(args, "memory", False),
    )

    if not result.base_stats and not result.head_stats:
        logger.warning("No benchmark data collected. Check that benchmarks-root is configured and benchmarks exist.")
        sys.exit(1)

    if args.output:
        md = result.format_markdown()
        Path(args.output).write_text(md, encoding="utf-8")
        logger.info(f"Markdown report written to {args.output}")


def get_current_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        branch = result.stdout.strip()
        return branch if branch and branch != "HEAD" else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def detect_base_ref(head_ref: str) -> str | None:
    # Try to find an open PR for this branch and use its base
    try:
        result = subprocess.run(
            ["gh", "pr", "view", head_ref, "--json", "baseRefName", "-q", ".baseRefName"],
            capture_output=True,
            text=True,
            check=True,
        )
        base = result.stdout.strip()
        if base:
            return base
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fall back to repo default branch
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "defaultBranchRef", "-q", ".defaultBranchRef.name"],
            capture_output=True,
            text=True,
            check=True,
        )
        default = result.stdout.strip()
        if default:
            return default
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Last resort: check for common default branch names
    try:
        for candidate in ("main", "master"):
            result = subprocess.run(
                ["git", "rev-parse", "--verify", candidate], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                return candidate
    except FileNotFoundError:
        pass

    return None


def resolve_pr_branch(pr_number: int) -> str:
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


def parse_functions_arg(functions_str: str, project_root: Path) -> dict[Path, list[FunctionToOptimize]]:
    """Parse --functions arg format: 'file.py::func1,func2;other.py::func3'."""
    from codeflash.models.function_types import FunctionToOptimize

    result: dict[Path, list[FunctionToOptimize]] = {}
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
            FunctionToOptimize(function_name=name, file_path=file_path, parents=[]) for name in func_names
        ]
    return result
