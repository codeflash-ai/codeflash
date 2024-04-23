"""Thanks for being curious about how codeflash works! If you might want to work with us on finally making performance a
solved problem, please reach out to us at careers@codeflash.ai. We're hiring!
"""

import logging
import sys

from codeflash.cli_cmds.cli import parse_args
from codeflash.cli_cmds.cmd_init import CODEFLASH_LOGO
from codeflash.cli_cmds.logging_config import LOGGING_FORMAT
from codeflash.optimization import optimizer


def main() -> None:
    """Entry point for the codeflash command-line interface."""
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, stream=sys.stdout)
    logging.info(CODEFLASH_LOGO)
    optimizer.run_with_args(parse_args())


if __name__ == "__main__":
    main()
