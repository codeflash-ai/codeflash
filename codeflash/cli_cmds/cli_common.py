import sys
from typing import NoReturn

import click


def apologize_and_exit() -> NoReturn:
    click.echo(
        "💡 If you're having trouble, see https://docs.codeflash.ai/getting-started/local-installation for further help getting started with Codeflash!",
    )
    click.echo("👋 Exiting...")
    sys.exit(1)
