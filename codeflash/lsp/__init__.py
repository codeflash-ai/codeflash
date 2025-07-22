# Silence the console module to prevent stdout pollution
from codeflash.cli_cmds.console import console

console.quiet = True
