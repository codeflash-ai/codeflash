"""Thanks for being curious about how codeflash works!.

If you might want to work with us on finally making performance a
solved problem, please reach out to us at careers@codeflash.ai. We're hiring!
"""

from pathlib import Path

from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO, ask_run_end_to_end_test
from codeflash.cli_cmds.console import paneled_text
from codeflash.code_utils.checkpoint import ask_should_use_checkpoint_get_functions
from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.version_check import check_for_newer_minor_version
from codeflash.telemetry import posthog_cf
from codeflash.telemetry.sentry import init_sentry


def main() -> None:
    """Entry point for the codeflash command-line interface."""
    paneled_text(
        CODEFLASH_LOGO, panel_args={"title": "https://codeflash.ai", "expand": False}, text_args={"style": "bold gold3"}
    )
    args = parse_args()
    
    # Check for newer version (skip for version command to avoid confusion)
    if not args.version:
        # Check if version check is disabled in config
        disable_version_check = False
        if args.config_file and Path.exists(args.config_file):
            pyproject_config, _ = parse_config_file(args.config_file)
            disable_version_check = pyproject_config.get("disable_version_check", False)
        
        check_for_newer_minor_version(disable_check=disable_version_check)
    
    if args.command:
        if args.config_file and Path.exists(args.config_file):
            pyproject_config, _ = parse_config_file(args.config_file)
            disable_telemetry = pyproject_config.get("disable_telemetry", False)
        else:
            disable_telemetry = False
        init_sentry(not disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(not disable_telemetry)
        args.func()
    elif args.verify_setup:
        args = process_pyproject_config(args)
        init_sentry(not args.disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(not args.disable_telemetry)
        ask_run_end_to_end_test(args)
    else:
        args = process_pyproject_config(args)
        args.previous_checkpoint_functions = ask_should_use_checkpoint_get_functions(args)
        init_sentry(not args.disable_telemetry, exclude_errors=True)
        posthog_cf.initialize_posthog(not args.disable_telemetry)

        from codeflash.optimization import optimizer

        optimizer.run_with_args(args)


if __name__ == "__main__":
    main()
