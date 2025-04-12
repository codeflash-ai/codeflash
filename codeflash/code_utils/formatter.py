from __future__ import annotations

import os
import shlex
import subprocess
from functools import partial
from typing import TYPE_CHECKING

import black
import isort

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path

imports_sort = partial(isort.code, float_to_top=True)


def format_code(formatter_cmds: list[str], path: Path) -> str:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = formatter_cmds[0].lower()
    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    if formatter_name == "disabled":
        return path.read_text(encoding="utf8")
    file_token = "$file"  # noqa: S105
    for command in set(formatter_cmds):
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [path.as_posix() if chunk == file_token else chunk for chunk in formatter_cmd_list]
        try:
            result = subprocess.run(formatter_cmd_list, capture_output=True, check=False)
            if result.returncode == 0:
                console.rule(f"Formatted Successfully with: {formatter_name.replace('$file', path.name)}")
            else:
                logger.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")
        except FileNotFoundError as e:
            from rich.panel import Panel
            from rich.text import Text

            panel = Panel(
                Text.from_markup(f"⚠️  Formatter command not found: {' '.join(formatter_cmd_list)}", style="bold red"),
                expand=False,
            )
            console.print(panel)

            raise e from None

    return path.read_text(encoding="utf8")


def format_code_in_memory(code: str, *, imports_only: bool = False) -> str:
    if imports_only:
        try:
            sorted_code = imports_sort(code)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to sort imports with isort.")
            return code
        return sorted_code
    try:
        formatted_code = black.format_str(code, mode=black.FileMode())
        formatted_code = imports_sort(formatted_code)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to format code with black.")
        return code

    return formatted_code
