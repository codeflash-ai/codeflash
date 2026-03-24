"""CLI entry point for codeflash."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash_core.strategy_utils import OptimizationStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfnext", description="Optimize Python code with AI.")
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument(
        "--show-config", action="store_true", help="Show current or auto-detected configuration and exit."
    )
    sub = parser.add_subparsers(dest="command")

    opt = sub.add_parser("optimize", help="Optimize functions in the given files.")
    opt.add_argument("files", nargs="*", help="Python files to optimize.")
    opt.add_argument("--file", dest="target_file", default=None, help="Single file to optimize.")
    opt.add_argument("--function", dest="target_function", default=None, help="Specific function name to optimize.")
    opt.add_argument("--all", action="store_true", dest="optimize_all", help="Optimize all files in module-root.")
    opt.add_argument("--project-root", type=Path, default=None, help="Override project root directory.")
    opt.add_argument("--module-root", default=None, help="Override module root (relative to project root).")
    opt.add_argument("--tests-root", default=None, help="Override tests root (relative to project root).")
    opt.add_argument("--benchmarks-root", default=None, help="Override benchmarks root (relative to project root).")
    opt.add_argument("--api-key", default=None, help="Codeflash API key (default: $CODEFLASH_API_KEY).")
    opt.add_argument("--effort", choices=["low", "medium", "high"], default=None, help="Effort level for optimization.")
    opt.add_argument(
        "--server",
        choices=["local", "prod"],
        default=None,
        help="AI service server: 'local' for localhost:8000, 'prod' for app.codeflash.ai.",
    )
    opt.add_argument(
        "--benchmark", action="store_true", help="Trace benchmark tests and calculate optimization impact."
    )
    opt.add_argument("--no-pr", action="store_true", help="Do not create a PR, only update code locally.")
    opt.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts.")
    opt.add_argument(
        "--formatter-cmds", nargs="*", default=None, help="Override formatter commands (each applied to $file)."
    )
    opt.add_argument("--pytest-cmd", default=None, help="Override the pytest command to use.")
    opt.add_argument("--disable-telemetry", action="store_true", help="Disable telemetry.")
    opt.add_argument(
        "--strategy",
        choices=["default", "k8bot"],
        default="default",
        help="Optimization strategy to use (default: default).",
    )
    opt.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    return parser


def collect_files(config_project_root: Path, config_module_root: str) -> list[Path]:
    """Collect all .py files under the module root."""
    module_dir = config_project_root / config_module_root if config_module_root else config_project_root
    if not module_dir.is_dir():
        return []
    return sorted(
        p for p in module_dir.rglob("*.py") if not p.name.startswith("test_") and not p.name.endswith("_test.py")
    )


SERVER_URLS = {"local": "http://localhost:8000", "prod": "https://app.codeflash.ai"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- Top-level flags (no subcommand needed) ------------------------------
    if args.version:
        from importlib.metadata import version

        print(f"cfnext {version('codeflash-core')}")
        return 0

    # -- Config (needed for --show-config and optimize) ----------------------
    from codeflash_core.config import CoreConfig

    start_dir = getattr(args, "project_root", None) or Path.cwd()
    config = CoreConfig.find_and_load(start_dir)

    if args.show_config:
        import json

        info = {
            "project_root": str(config.project_root),
            "module_root": config.module_root,
            "tests_root": config.tests_root,
            "benchmarks_root": config.benchmarks_root,
            "effort": config.effort,
            "formatter_cmds": config.formatter_cmds,
            "ignore_paths": config.ignore_paths,
            "ai_base_url": config.ai.base_url,
            "disable_telemetry": config.disable_telemetry,
        }
        print(json.dumps(info, indent=2))
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command != "optimize":
        parser.print_help()
        return 1

    # -- Logging -------------------------------------------------------------
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(name)s %(levelname)s: %(message)s"
    )

    # CLI overrides
    if args.project_root:
        config.project_root = args.project_root.resolve()
    if args.module_root is not None:
        config.module_root = args.module_root
    if args.tests_root is not None:
        config.tests_root = args.tests_root
    if args.benchmarks_root is not None:
        config.benchmarks_root = args.benchmarks_root
    if args.effort is not None:
        config.effort = args.effort
    if args.server is not None:
        config.ai.base_url = SERVER_URLS[args.server]
    if args.formatter_cmds is not None:
        config.formatter_cmds = args.formatter_cmds
    if args.disable_telemetry:
        config.disable_telemetry = True

    # API key: CLI flag > env var > config file
    api_key = args.api_key or os.environ.get("CODEFLASH_API_KEY", "") or config.ai.api_key
    if not api_key:
        print("Error: No API key provided. Set CODEFLASH_API_KEY or pass --api-key.")
        return 1
    config.ai.api_key = api_key

    # -- Telemetry -----------------------------------------------------------
    from codeflash_core.telemetry import PostHogClient, init_sentry

    if not config.disable_telemetry:
        PostHogClient.initialize(config.telemetry.posthog_api_key, enabled=config.telemetry.enabled)
        init_sentry(config.telemetry.sentry_dsn, enabled=config.telemetry.enabled)

    # -- Resolve files -------------------------------------------------------
    if args.target_file:
        files = [Path(args.target_file).resolve()]
    elif args.optimize_all:
        files = collect_files(config.project_root, config.module_root)
    elif args.files:
        files = [Path(f).resolve() for f in args.files]
    else:
        print("Error: Provide files to optimize, use --file, or use --all.")
        return 1

    if not files:
        print("Error: No Python files found.")
        return 1

    # -- Build plugin & validate environment ---------------------------------
    from codeflash_python.plugin import PythonPlugin

    from codeflash_core.optimizer import Optimizer

    plugin = PythonPlugin(config.project_root)

    if hasattr(plugin, "validate_environment") and not plugin.validate_environment(config):
        return 1

    # -- Resolve strategy ----------------------------------------------------
    strategy = _resolve_strategy(args.strategy)

    optimizer = Optimizer(config, plugin, strategy=strategy)
    results = optimizer.run(files, function_filter=args.target_function)

    # -- Shutdown telemetry --------------------------------------------------
    if PostHogClient.instance is not None:
        PostHogClient.instance.shutdown()

    return 0 if results else 2


def _resolve_strategy(name: str) -> OptimizationStrategy:
    """Return the strategy instance for the given CLI name."""
    from codeflash_core.strategy import DefaultStrategy

    if name == "k8bot":
        from codeflash_core.k8bot import K8BotStrategy

        return K8BotStrategy()
    return DefaultStrategy()


if __name__ == "__main__":
    sys.exit(main())
