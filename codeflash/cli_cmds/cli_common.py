from __future__ import annotations

import sys
from typing import NoReturn

from codeflash.cli_cmds.console import console, logger


def apologize_and_exit() -> NoReturn:
    console.rule()
    logger.info(
        "💡 If you're having trouble, see https://docs.codeflash.ai/getting-started/local-installation for further help getting started with Codeflash!"
    )
    console.rule()
    logger.info("👋 Exiting...")
    sys.exit(1)
