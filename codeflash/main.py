"""Thanks for being curious about how codeflash works!.

If you might want to work with us on finally making performance a
solved problem, please reach out to us at careers@codeflash.ai. We're hiring!
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if "--subagent" in sys.argv:
    os.environ["CODEFLASH_SUBAGENT_MODE"] = "true"
    import warnings

    warnings.filterwarnings("ignore")

from codeflash.cli_cmds.cli import (
    apply_language_config,
    handle_optimize_all_arg_parsing,
    parse_args,
    process_pyproject_config,
)
from codeflash.cli_cmds.console import paneled_text
from codeflash.code_utils import env_utils
from codeflash.code_utils.checkpoint import ask_should_use_checkpoint_get_functions
from codeflash.code_utils.config_parser import find_all_config_files, parse_config_file
from codeflash.code_utils.version_check import check_for_newer_minor_version
from codeflash.languages.registry import UnsupportedLanguageError, get_language_support

if TYPE_CHECKING:
    from argparse import Namespace


def main() -> None:
    """Entry point for the codeflash command-line interface."""
    from codeflash.telemetry import posthog_cf
    from codeflash.telemetry.sentry import init_sentry

    args = parse_args()
    if args.command != "auth":
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

        if args.command == "auth":
            from codeflash.cli_cmds.cmd_auth import auth_login, auth_status

            if args.auth_command == "login":
                auth_login()
            elif args.auth_command == "status":
                auth_status()
            else:
                from codeflash.code_utils.code_utils import exit_with_message

                exit_with_message("Usage: codeflash auth {login,status}", error_on_exit=True)
        elif args.command == "init":
            from codeflash.cli_cmds.cmd_init import init_codeflash

            init_codeflash()
        elif args.command == "init-actions":
            from codeflash.cli_cmds.github_workflow import install_github_actions

            install_github_actions()
        elif args.command == "vscode-install":
            from codeflash.cli_cmds.extension import install_vscode_extension

            install_vscode_extension()
        elif args.command == "compare":
            from codeflash.cli_cmds.cmd_compare import run_compare

            run_compare(args)
        elif args.command == "optimize":
            from codeflash.tracer import main as tracer_main

            tracer_main(args)
    elif args.verify_setup:
        args = process_pyproject_config(args)
        init_sentry(enabled=not args.disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(enabled=not args.disable_telemetry)

        from codeflash.cli_cmds.cmd_init import ask_run_end_to_end_test

        ask_run_end_to_end_test(args)
    else:
        language_configs = find_all_config_files()

        logger = logging.getLogger("codeflash")

        if not language_configs:
            # Fallback: no multi-config found, use existing single-config path
            loaded_args = _handle_config_loading(args)
            if loaded_args is None:
                sys.exit(0)
            args = loaded_args

            if not env_utils.check_formatter_installed(args.formatter_cmds):
                return
            args.previous_checkpoint_functions = ask_should_use_checkpoint_get_functions(args)
            init_sentry(enabled=not args.disable_telemetry, exclude_errors=True)
            posthog_cf.initialize_posthog(enabled=not args.disable_telemetry)

            from codeflash.optimization import optimizer

            optimizer.run_with_args(args)
            return

        # Filter to single language when --file is specified
        if hasattr(args, "file") and args.file:
            try:
                file_lang_support = get_language_support(Path(args.file))
                file_language = file_lang_support.language
                matching_configs = [lc for lc in language_configs if lc.language == file_language]
                if matching_configs:
                    language_configs = matching_configs
            except UnsupportedLanguageError:
                pass  # Unknown extension, let all configs run

        # Save the raw --all value before handle_optimize_all_arg_parsing mutates it.
        # In multi-language mode, module_root is None at this point so the resolution
        # produces None for the default case; we re-resolve per language inside the loop.
        original_all = getattr(args, "all", None) if hasattr(args, "all") else None
        optimize_all_requested = hasattr(args, "all") and original_all is not None

        # Multi-language path: run git/GitHub checks ONCE before the loop
        args = handle_optimize_all_arg_parsing(args)

        results: dict[str, str] = {}
        for lang_config in language_configs:
            lang_name = lang_config.language.value
            try:
                pass_args = copy.deepcopy(args)
                pass_args = apply_language_config(pass_args, lang_config)

                if optimize_all_requested:
                    if original_all == "":
                        # --all with no path: use this language's module_root
                        pass_args.all = pass_args.module_root
                    else:
                        # --all /specific/path: preserve the user's path
                        pass_args.all = Path(original_all).resolve()

                if not env_utils.check_formatter_installed(pass_args.formatter_cmds):
                    logger.info("Skipping %s: formatter not installed", lang_name)
                    results[lang_name] = "skipped"
                    continue

                pass_args.previous_checkpoint_functions = ask_should_use_checkpoint_get_functions(pass_args)
                init_sentry(enabled=not pass_args.disable_telemetry, exclude_errors=True)
                posthog_cf.initialize_posthog(enabled=not pass_args.disable_telemetry)

                logger.info("Processing %s (config: %s)", lang_name, lang_config.config_path)

                from codeflash.optimization import optimizer

                optimizer.run_with_args(pass_args)
                results[lang_name] = "success"
            except Exception:
                logger.exception("Error processing %s, continuing with remaining languages", lang_name)
                results[lang_name] = "failed"

        _log_orchestration_summary(logger, results)


def _log_orchestration_summary(logger: logging.Logger, results: dict[str, str]) -> None:
    if not results:
        return
    parts = [f"{lang}: {status}" for lang, status in results.items()]
    logger.info("Multi-language orchestration complete: %s", ", ".join(parts))


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
    """Print the Codeflash banner with the branded styling.

    Renders the Codeflash ASCII logo inside a non-expanding panel titled with
    https://codeflash.ai, using bold gold text for visual emphasis.
    """
    from codeflash.cli_cmds.console_constants import CODEFLASH_LOGO

    paneled_text(
        CODEFLASH_LOGO, panel_args={"title": "https://codeflash.ai", "expand": False}, text_args={"style": "bold gold3"}
    )


if __name__ == "__main__":
    main()
