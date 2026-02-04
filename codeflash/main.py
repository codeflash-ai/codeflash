"""Thanks for being curious about how codeflash works!.

If you might want to work with us on finally making performance a
solved problem, please reach out to us at careers@codeflash.ai. We're hiring!
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO, ask_run_end_to_end_test
from codeflash.cli_cmds.console import paneled_text
from codeflash.code_utils import env_utils
from codeflash.code_utils.checkpoint import ask_should_use_checkpoint_get_functions
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.version_check import check_for_newer_minor_version
from codeflash.telemetry import posthog_cf
from codeflash.telemetry.sentry import init_sentry

if TYPE_CHECKING:
    from argparse import Namespace


def main() -> None:
    """Entry point for the codeflash command-line interface."""
    args = parse_args()
    print_codeflash_banner()

    # Check for newer version for all commands
    check_for_newer_minor_version()

    if args.command:
        disable_telemetry = False
        if args.config_file and Path.exists(args.config_file):
            pyproject_config, _ = parse_config_file(args.config_file)
            disable_telemetry = pyproject_config.get("disable_telemetry", False)
        init_sentry(enabled=not disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(enabled=not disable_telemetry)
        args.func()
    elif args.verify_setup:
        args = process_pyproject_config(args)
        init_sentry(enabled=not args.disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(enabled=not args.disable_telemetry)
        ask_run_end_to_end_test(args)
    else:
        # Check for first-run experience (no config exists)
        loaded_args = _handle_config_loading(args)
        if loaded_args is None:
            sys.exit(0)
        args = loaded_args

        if not env_utils.check_formatter_installed(args.formatter_cmds):
            return
        args.previous_checkpoint_functions = ask_should_use_checkpoint_get_functions(args)
        init_sentry(enabled=not args.disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(enabled=not args.disable_telemetry)

        if getattr(args, "agentic", False):
            from codeflash.optimization.agentic_optimizer import run_agentic_with_args

            run_agentic_with_args(args)
        else:
            from codeflash.optimization import optimizer

            optimizer.run_with_args(args)


def _handle_config_loading(args: Namespace) -> Namespace | None:
    """Handle config loading with first-run experience support.

    If no config exists and not in CI, triggers the first-run experience.
    Otherwise, loads config normally.

    Args:
        args: CLI args namespace.

    Returns:
        Updated args with config loaded, or None if user cancelled first-run.

    """
    from codeflash.setup.first_run import handle_first_run, is_first_run

    # Check if we're in CI environment
    is_ci = any(
        var in ("true", "1", "True") for var in [os.environ.get("CI", ""), os.environ.get("GITHUB_ACTIONS", "")]
    )

    # Check if first run (no config exists)
    if is_first_run() and not is_ci:
        # Skip API key check if already set
        skip_api_key = bool(os.environ.get("CODEFLASH_API_KEY"))

        # Handle first-run experience
        result = handle_first_run(args=args, skip_confirm=getattr(args, "yes", False), skip_api_key=skip_api_key)

        if result is None:
            return None

        # Merge first-run results with any CLI overrides
        args = result
        # Still need to process some config values
        # Config might not exist yet if first run just saved it - that's OK
        import contextlib

        with contextlib.suppress(ValueError):
            args = process_pyproject_config(args)

        return args

    # Normal config loading
    return process_pyproject_config(args)


def print_codeflash_banner() -> None:
    paneled_text(
        CODEFLASH_LOGO, panel_args={"title": "https://codeflash.ai", "expand": False}, text_args={"style": "bold gold3"}
    )


if __name__ == "__main__":
    main()
