"""Thanks for being curious about how codeflash works! If you might want to work with us on finally making performance a
solved problem, please reach out to us at careers@codeflash.ai. We're hiring!
"""

import logging
import sys

from codeflash.cli_cmds.cli import parse_args, process_pyproject_config
from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
from codeflash.cli_cmds.logging_config import LOGGING_FORMAT
from codeflash.optimization import optimizer
from codeflash.telemetry import posthog
from codeflash.telemetry.sentry import init_sentry


def main() -> None:
    """Entry point for the codeflash command-line interface."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, stream=sys.stdout)
    logging.info(CODEFLASH_LOGO)
    args = parse_args()
    if args.command:
        disable_telemetry = args.disable_telemetry if hasattr(args, "disable_telemetry") else False
        init_sentry(not disable_telemetry, exclude_errors=True)
        posthog.initialize_posthog(not disable_telemetry)
        args.func()
    else:
        args = process_pyproject_config(args)
        init_sentry(not args.disable_telemetry, exclude_errors=True)
        posthog.initialize_posthog(not args.disable_telemetry)
        optimizer.run_with_args(args)


if __name__ == "__main__":
    main()
