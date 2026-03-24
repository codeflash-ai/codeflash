from __future__ import annotations

import logging
import sys
from typing import NoReturn

logger = logging.getLogger("codeflash_python")


def apologize_and_exit() -> NoReturn:
    logger.info(
        "\U0001f4a1 If you're having trouble, see https://docs.codeflash.ai/getting-started/local-installation for further help getting started with Codeflash!"
    )
    logger.info("\U0001f44b Exiting...")
    sys.exit(1)
