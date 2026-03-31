from __future__ import annotations

import logging
import os
import sys
from argparse import SUPPRESS, ArgumentParser
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeflash.cli_cmds import logging_config
from codeflash.cli_cmds.console import apologize_and_exit, logger
from codeflash.code_utils import env_utils
from codeflash.code_utils.code_utils import exit_with_message, normalize_ignore_paths
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.languages import set_current_language
from codeflash.languages.language_enum import Language
from codeflash.languages.test_framework import set_current_test_framework
from codeflash.lsp.helpers import is_LSP_enabled
from codeflash.version import __version__ as version

if TYPE_CHECKING:
    from argparse import Namespace

    from codeflash.code_utils.config_parser import LanguageConfig


def parse_args() -> Namespace:
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args()
    sys.argv[:] = [sys.argv[0], *unknown_args]
    if args.subagent:
        args.yes = True
        args.no_pr = True
        args.worktree = True
        args.effort = "low"
    if args.command in ("auth", "compare"):
        return args
    return process_and_validate_cmd_args(args)


def process_and_validate_cmd_args(args: Namespace) -> Namespace:
    from codeflash.code_utils.git_utils import (
        check_running_in_git_repo,
        confirm_proceeding_with_no_git_repo,
        get_repo_owner_and_name,
    )
    from codeflash.code_utils.github_utils import require_github_app_or_exit

    if args.server:
        os.environ["CODEFLASH_AIS_SERVER"] = args.server

    is_init: bool = args.command.startswith("init") if args.command else False
    if args.verbose:
        logging_config.set_level(logging.DEBUG, echo_setting=not is_init)
    else:
        logging_config.set_level(logging.INFO, echo_setting=not is_init)

    if args.version:
        logger.info(f"Codeflash version {version}")
        sys.exit()

    # Handle --show-config
    if getattr(args, "show_config", False):
        _handle_show_config()
        sys.exit()

    # Handle --reset-config
    if getattr(args, "reset_config", False):
        _handle_reset_config(confirm=not getattr(args, "yes", False))
        sys.exit()

    if not check_running_in_git_repo(module_root=args.module_root):
        if not confirm_proceeding_with_no_git_repo():
            exit_with_message("No git repository detected and user aborted run. Exiting...", error_on_exit=True)
        args.no_pr = True
    if args.function and not args.file:
        exit_with_message("If you specify a --function, you must specify the --file it is in", error_on_exit=True)
    if args.file:
        if not Path(args.file).exists():
            exit_with_message(f"File {args.file} does not exist", error_on_exit=True)
        args.file = Path(args.file).resolve()
        if not args.no_pr:
            owner, repo = get_repo_owner_and_name()
            require_github_app_or_exit(owner, repo)
    if args.replay_test:
        for test_path in args.replay_test:
            if not Path(test_path).is_file():
                exit_with_message(f"Replay test file {test_path} does not exist", error_on_exit=True)
        args.replay_test = [Path(replay_test).resolve() for replay_test in args.replay_test]
        if env_utils.is_ci():
            args.no_pr = True

    return args


def process_pyproject_config(args: Namespace) -> Namespace:
    try:
        pyproject_config, pyproject_file_path = parse_config_file(args.config_file)
    except ValueError as e:
        exit_with_message(f"Error parsing config file: {e}", error_on_exit=True)

    language = None
    lang_str = pyproject_config.get("language")
    if lang_str:
        language = Language(lang_str)

    args = resolve_config_onto_args(args, pyproject_config, pyproject_file_path, language)

    if is_LSP_enabled():
        args.all = None
        return args
    return handle_optimize_all_arg_parsing(args)


def resolve_config_onto_args(
    args: Namespace, config: dict[str, Any], config_path: Path, language: Language | None
) -> Namespace:
    supported_keys = [
        "module_root",
        "tests_root",
        "benchmarks_root",
        "ignore_paths",
        "pytest_cmd",
        "formatter_cmds",
        "disable_telemetry",
        "disable_imports_sorting",
        "git_remote",
        "override_fixtures",
    ]
    for key in supported_keys:
        attr_key = key.replace("-", "_")
        if key in config and (
            (hasattr(args, attr_key) and getattr(args, attr_key) is None) or not hasattr(args, attr_key)
        ):
            setattr(args, attr_key, config[key])

    assert args.module_root is not None, "--module-root must be specified"
    assert Path(args.module_root).is_dir(), f"--module-root {args.module_root} must be a valid directory"

    is_js_ts = language in (Language.JAVASCRIPT, Language.TYPESCRIPT)
    is_java = language == Language.JAVA

    if language is not None:
        set_current_language(language)

    if is_js_ts and config.get("test_framework"):
        set_current_test_framework(config["test_framework"])

    if args.tests_root is None:
        if is_java:
            for test_dir in ["src/test/java", "test", "tests"]:
                test_path = Path(args.module_root).parent / test_dir if "/" in test_dir else Path(test_dir)
                if not test_path.is_absolute():
                    test_path = Path.cwd() / test_path
                if test_path.is_dir():
                    args.tests_root = str(test_path)
                    break
            if args.tests_root is None:
                args.tests_root = str(Path.cwd() / "src" / "test" / "java")
        elif is_js_ts:
            for test_dir in ["test", "tests", "__tests__"]:
                if Path(test_dir).is_dir():
                    args.tests_root = test_dir
                    break
            if args.tests_root is None and args.module_root:
                module_root_path = Path(args.module_root)
                for test_dir in ["test", "tests", "__tests__"]:
                    test_path = module_root_path / test_dir
                    if test_path.is_dir():
                        args.tests_root = str(test_path)
                        break
            if args.tests_root is None:
                args.tests_root = args.module_root
        else:
            raise AssertionError("--tests-root must be specified")

    assert Path(args.tests_root).is_dir(), f"--tests-root {args.tests_root} must be a valid directory"

    if getattr(args, "benchmark", False):
        assert args.benchmarks_root is not None, "--benchmarks-root must be specified when running with --benchmark"
        assert Path(args.benchmarks_root).is_dir(), (
            f"--benchmarks-root {args.benchmarks_root} must be a valid directory"
        )
        if env_utils.get_pr_number() is not None:
            import git

            from codeflash.code_utils.git_utils import get_repo_owner_and_name
            from codeflash.code_utils.github_utils import get_github_secrets_page_url, require_github_app_or_exit

            assert env_utils.ensure_codeflash_api_key(), (
                "Codeflash API key not found. When running in a Github Actions Context, provide the "
                "'CODEFLASH_API_KEY' environment variable as a secret.\n"
                "You can add a secret by going to your repository's settings page, then clicking 'Secrets' in the left sidebar.\n"
                "Then, click 'New repository secret' and add your api key with the variable name CODEFLASH_API_KEY.\n"
                f"Here's a direct link: {get_github_secrets_page_url()}\n"
                "Exiting..."
            )

            repo = git.Repo(search_parent_directories=True)

            owner, repo_name = get_repo_owner_and_name(repo)

            require_github_app_or_exit(owner, repo_name)

    args.module_root = Path(args.module_root).resolve()
    if hasattr(args, "ignore_paths") and args.ignore_paths is not None:
        args.ignore_paths = normalize_ignore_paths(args.ignore_paths, base_path=args.module_root)
    args.project_root = project_root_from_module_root(Path(args.module_root), config_path)
    args.tests_root = Path(args.tests_root).resolve()
    if args.benchmarks_root:
        args.benchmarks_root = Path(args.benchmarks_root).resolve()
    args.test_project_root = project_root_from_module_root(args.tests_root, config_path)

    if is_java and config_path.is_dir():
        args.project_root = config_path.resolve()
        args.test_project_root = config_path.resolve()

    return args


def project_root_from_module_root(module_root: Path, pyproject_file_path: Path) -> Path:
    if pyproject_file_path.parent == module_root:
        return module_root

    # For Java projects, find the directory containing pom.xml or build.gradle
    # This handles the case where module_root is src/main/java
    current = module_root
    while current != current.parent:
        if (current / "pom.xml").exists():
            return current.resolve()
        if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
            return current.resolve()
        current = current.parent

    return module_root.parent.resolve()


def apply_language_config(args: Namespace, lang_config: LanguageConfig) -> Namespace:
    return resolve_config_onto_args(args, lang_config.config, lang_config.config_path, lang_config.language)


def handle_optimize_all_arg_parsing(args: Namespace) -> Namespace:
    if hasattr(args, "all") or (hasattr(args, "file") and args.file):
        no_pr = getattr(args, "no_pr", False)

        if not no_pr:
            import git

            from codeflash.code_utils.git_utils import check_and_push_branch, get_repo_owner_and_name
            from codeflash.code_utils.github_utils import require_github_app_or_exit

            # Ensure that the user can actually open PRs on the repo.
            try:
                git_repo = git.Repo(search_parent_directories=True)
            except (git.exc.InvalidGitRepositoryError, git.exc.NoSuchPathError):
                mode = "--all" if hasattr(args, "all") else "--file"
                logger.exception(
                    f"I couldn't find a git repository in the current directory. "
                    f"I need a git repository to run {mode} and open PRs for optimizations. Exiting..."
                )
                apologize_and_exit()
            git_remote = getattr(args, "git_remote", None) or "origin"
            if not check_and_push_branch(git_repo, git_remote=git_remote):
                exit_with_message("Branch is not pushed...", error_on_exit=True)
            owner, repo = get_repo_owner_and_name(git_repo)
            require_github_app_or_exit(owner, repo)
    if not hasattr(args, "all"):
        args.all = None
    elif args.all == "":
        # The default behavior of --all is to optimize everything in args.module_root
        args.all = args.module_root
    else:
        args.all = Path(args.all).resolve()
    return args


def _handle_show_config() -> None:
    """Show current or auto-detected Codeflash configuration."""
    from rich.table import Table

    from codeflash.cli_cmds.console import console
    from codeflash.setup.detector import detect_project, has_existing_config

    project_root = Path.cwd()
    config_exists, _ = has_existing_config(project_root)

    if config_exists:
        from codeflash.code_utils.config_parser import parse_config_file

        config, config_file_path = parse_config_file()
        status = "Saved config"

        console.print()
        console.print(f"[bold]Codeflash Configuration[/bold] ({status})")
        console.print(f"[dim]Config file: {config_file_path}[/dim]")
        console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="dim")
        table.add_column("Value")

        table.add_row("Project root", str(project_root))
        table.add_row("Module root", config.get("module_root", "(not set)"))
        table.add_row("Tests root", config.get("tests_root", "(not set)"))
        table.add_row("Test runner", config.get("test_framework", config.get("pytest_cmd", "(not set)")))
        table.add_row("Formatter", ", ".join(config["formatter_cmds"]) if config.get("formatter_cmds") else "(not set)")
        ignore_paths = config.get("ignore_paths", [])
        table.add_row("Ignore paths", ", ".join(str(p) for p in ignore_paths) if ignore_paths else "(none)")
    else:
        detected = detect_project(project_root)
        status = "Auto-detected (not saved)"

        console.print()
        console.print(f"[bold]Codeflash Configuration[/bold] ({status})")
        console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="dim")
        table.add_column("Value")

        table.add_row("Language", detected.language)
        table.add_row("Project root", str(detected.project_root))
        table.add_row("Module root", str(detected.module_root))
        table.add_row("Tests root", str(detected.tests_root) if detected.tests_root else "(not detected)")
        table.add_row("Test runner", detected.test_runner or "(not detected)")
        table.add_row("Formatter", ", ".join(detected.formatter_cmds) if detected.formatter_cmds else "(not detected)")
        table.add_row(
            "Ignore paths", ", ".join(str(p) for p in detected.ignore_paths) if detected.ignore_paths else "(none)"
        )
        table.add_row("Confidence", f"{detected.confidence:.0%}")

    console.print(table)
    console.print()

    if not config_exists:
        console.print("[dim]Run [bold]codeflash --file <file>[/bold] to auto-save this config.[/dim]")


def _handle_reset_config(confirm: bool = True) -> None:
    """Remove Codeflash configuration from project config file.

    Args:
        confirm: If True, prompt for confirmation before removing.

    """
    from codeflash.cli_cmds.console import console
    from codeflash.setup.config_writer import remove_config
    from codeflash.setup.detector import detect_project, has_existing_config

    project_root = Path.cwd()

    config_exists, _ = has_existing_config(project_root)
    if not config_exists:
        console.print("[yellow]No Codeflash configuration found to remove.[/yellow]")
        return

    detected = detect_project(project_root)

    if confirm:
        console.print("[bold]This will remove Codeflash configuration from your project.[/bold]")
        console.print()

        config_file = "pyproject.toml" if detected.language == "python" else "package.json"
        console.print(f"  Config file: {project_root / config_file}")
        console.print()

        try:
            response = console.input("[bold]Are you sure you want to remove the config? [y/N][/bold] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return

        if response.lower() not in ("y", "yes"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    success, message = remove_config(project_root, detected.language)

    # Escape brackets in message to prevent Rich markup interpretation
    escaped_message = message.replace("[", "\\[")

    if success:
        console.print(f"[green]✓[/green] {escaped_message}")
    else:
        console.print(f"[red]✗[/red] {escaped_message}")


@lru_cache(maxsize=1)
def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    subparsers.add_parser("init", help="Initialize Codeflash for your project.")
    subparsers.add_parser("vscode-install", help="Install the Codeflash VSCode extension")
    subparsers.add_parser("init-actions", help="Initialize GitHub Actions workflow")

    auth_parser = subparsers.add_parser("auth", help="Authentication commands")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", help="Auth sub-commands")
    auth_subparsers.add_parser("login", help="Log in to Codeflash via OAuth")
    auth_subparsers.add_parser("status", help="Check authentication status")

    compare_parser = subparsers.add_parser("compare", help="Compare benchmark performance between two git refs.")
    compare_parser.add_argument("base_ref", help="Base git ref (branch, tag, or commit)")
    compare_parser.add_argument("head_ref", nargs="?", default=None, help="Head git ref (default: current branch)")
    compare_parser.add_argument("--pr", type=int, help="Resolve head ref from a PR number (requires gh CLI)")
    compare_parser.add_argument(
        "--functions", type=str, help="Explicit functions to instrument: 'file.py::func1,func2;other.py::func3'"
    )
    compare_parser.add_argument("--timeout", type=int, default=600, help="Benchmark timeout in seconds (default: 600)")
    compare_parser.add_argument("--config-file", type=str, dest="config_file", help="Path to pyproject.toml")

    trace_optimize = subparsers.add_parser("optimize", help="Trace and optimize your project.", add_help=False)

    trace_optimize.add_argument(
        "--max-function-count",
        type=int,
        default=100,
        help="The maximum number of times to trace a single function. More calls to a function will not be traced. Default is 100.",
    )
    trace_optimize.add_argument(
        "--timeout",
        type=int,
        help="The maximum time in seconds to trace the entire workflow. Default is indefinite. This is useful while tracing really long workflows, to not wait indefinitely.",
    )
    trace_optimize.add_argument(
        "--output",
        type=str,
        default="codeflash.trace",
        help="The file to save the trace to. Default is codeflash.trace.",
    )
    trace_optimize.add_argument(
        "--config-file-path",
        type=str,
        help="The path to the pyproject.toml file which stores the Codeflash config. This is auto-discovered by default.",
    )

    parser.add_argument("--file", help="Try to optimize only this file")
    parser.add_argument("--function", help="Try to optimize only this function within the given file path")
    parser.add_argument(
        "--all",
        help="Try to optimize all functions. Can take a really long time. Can pass an optional starting directory to"
        " optimize code from. If no args specified (just --all), will optimize all code in the project.",
        nargs="?",
        const="",
        default=SUPPRESS,
    )
    parser.add_argument(
        "--module-root",
        type=str,
        help="Path to the project's module that you want to optimize."
        " This is the top-level root directory where all the source code is located.",
    )
    parser.add_argument(
        "--tests-root", type=str, help="Path to the test directory of the project, where all the tests are located."
    )
    parser.add_argument("--config-file", type=str, help="Path to the pyproject.toml with codeflash configs.")
    parser.add_argument("--replay-test", type=str, nargs="+", help="Paths to replay test to optimize functions from")
    parser.add_argument(
        "--rerun",
        type=str,
        help="Rerun a previous optimization by trace ID, using stored LLM results",
        metavar="TRACE_ID",
    )
    parser.add_argument(
        "--no-pr", action="store_true", help="Do not create a PR for the optimization, only update the code locally."
    )
    parser.add_argument(
        "--no-gen-tests", action="store_true", help="Do not generate tests, use only existing tests for optimization."
    )
    parser.add_argument(
        "--no-jit-opts", action="store_true", help="Do not generate JIT-compiled optimizations for numerical code."
    )
    parser.add_argument("--staging-review", action="store_true", help="Upload optimizations to staging for review")
    parser.add_argument(
        "--verify-setup",
        action="store_true",
        help="Verify that codeflash is set up correctly by optimizing bubble sort as a test.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose debug logs")
    parser.add_argument("--version", action="store_true", help="Print the version of codeflash")
    parser.add_argument(
        "--benchmark", action="store_true", help="Trace benchmark tests and calculate optimization impact on benchmarks"
    )
    parser.add_argument(
        "--benchmarks-root",
        type=str,
        help="Path to the directory of the project, where all the pytest-benchmark tests are located.",
    )
    parser.add_argument("--no-draft", default=False, action="store_true", help="Skip optimization for draft PRs")
    parser.add_argument("--worktree", default=False, action="store_true", help="Use worktree for optimization")
    parser.add_argument(
        "--testgen-review", default=False, action="store_true", help="Enable AI review and repair of generated tests"
    )
    parser.add_argument(
        "--testgen-review-turns", type=int, default=None, help="Number of review/repair cycles (default: 2)"
    )
    parser.add_argument(
        "--async",
        default=False,
        action="store_true",
        help="(Deprecated) Async function optimization is now enabled by default. This flag is ignored.",
    )
    parser.add_argument(
        "--server",
        type=str,
        choices=["local", "prod"],
        help="AI service server to use: 'local' for localhost:8000, 'prod' for app.codeflash.ai",
    )
    parser.add_argument(
        "--effort", type=str, help="Effort level for optimization", choices=["low", "medium", "high"], default="medium"
    )

    # Config management flags
    parser.add_argument(
        "--show-config", action="store_true", help="Show current or auto-detected configuration and exit."
    )
    parser.add_argument(
        "--reset-config", action="store_true", help="Remove codeflash configuration from project config file."
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts (useful for CI/scripts).")
    parser.add_argument(
        "--subagent",
        action="store_true",
        help="Subagent mode: skip all interactive prompts with sensible defaults. Designed for AI agent integrations.",
    )

    return parser
